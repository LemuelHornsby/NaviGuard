"""NaviSense simulator connector.

Supports two transport modes:

disk
    NaviSense writes rendered frames as JPEG/PNG files and own-ship state /
    AIS tracks as JSON files to known paths.  NaviGuard polls these files.
    Simple and reliable for early development.

websocket
    NaviSense streams data in real time over a WebSocket connection.
    Messages are JSON-encoded dicts with a ``"type"`` field:
        - ``"frame"``     → base64-encoded camera image
        - ``"own_ship"``  → own-ship state dict
        - ``"ais"``       → list of AIS-like dicts
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures shared with the rest of NaviGuard
# ---------------------------------------------------------------------------
@dataclass
class OwnShipState:
    lat: float = 0.0
    lon: float = 0.0
    heading_deg: float = 0.0
    speed_kn: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SyntheticAISTrack:
    mmsi: int = 0
    lat: float = 0.0
    lon: float = 0.0
    cog: float = 0.0   # course over ground (degrees)
    sog: float = 0.0   # speed over ground (knots)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class NaviSenseClientBase:
    """Abstract base – concrete implementations below."""

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest camera frame as a BGR numpy array, or None."""
        raise NotImplementedError

    def get_own_ship(self) -> OwnShipState:
        """Return the current own-ship state."""
        raise NotImplementedError

    def get_ais_tracks(self) -> List[SyntheticAISTrack]:
        """Return the current list of simulated AIS tracks."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources."""


# ---------------------------------------------------------------------------
# Disk-polling client
# ---------------------------------------------------------------------------
class DiskNaviSenseClient(NaviSenseClientBase):
    """Polls frame / state / AIS files written by NaviSense to disk.

    Parameters
    ----------
    frame_dir : str | Path
        Directory where NaviSense writes frame files, named
        ``frame_<timestamp>.jpg`` (or ``.png``).
    state_file : str | Path
        JSON file with the latest own-ship state::
            {"lat": …, "lon": …, "heading_deg": …, "speed_kn": …}
    ais_file : str | Path
        JSON file with a list of AIS-like dicts::
            [{"mmsi": …, "lat": …, "lon": …, "cog": …, "sog": …}, …]
    """

    def __init__(
        self,
        frame_dir: str | Path,
        state_file: str | Path,
        ais_file: str | Path,
    ) -> None:
        import cv2  # local import to keep the module importable without cv2

        self._cv2 = cv2
        self.frame_dir = Path(frame_dir)
        self.state_file = Path(state_file)
        self.ais_file = Path(ais_file)
        self._last_frame_path: Optional[Path] = None

    def get_frame(self) -> Optional[np.ndarray]:
        candidates = sorted(
            list(self.frame_dir.glob("*.jpg")) + list(self.frame_dir.glob("*.png"))
        )
        if not candidates:
            return None
        latest = candidates[-1]
        if latest == self._last_frame_path:
            return None  # no new frame yet
        self._last_frame_path = latest
        frame = self._cv2.imread(str(latest))
        return frame  # BGR numpy array or None if imread failed

    def get_own_ship(self) -> OwnShipState:
        if not self.state_file.exists():
            return OwnShipState()
        try:
            with open(self.state_file) as f:
                d = json.load(f)
            return OwnShipState(
                lat=float(d.get("lat", 0)),
                lon=float(d.get("lon", 0)),
                heading_deg=float(d.get("heading_deg", 0)),
                speed_kn=float(d.get("speed_kn", 0)),
                timestamp=float(d.get("timestamp", time.time())),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Could not parse own-ship state: %s", exc)
            return OwnShipState()

    def get_ais_tracks(self) -> List[SyntheticAISTrack]:
        if not self.ais_file.exists():
            return []
        try:
            with open(self.ais_file) as f:
                data = json.load(f)
            tracks = []
            for d in data:
                tracks.append(
                    SyntheticAISTrack(
                        mmsi=int(d.get("mmsi", 0)),
                        lat=float(d.get("lat", 0)),
                        lon=float(d.get("lon", 0)),
                        cog=float(d.get("cog", 0)),
                        sog=float(d.get("sog", 0)),
                        timestamp=float(d.get("timestamp", time.time())),
                    )
                )
            return tracks
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Could not parse AIS tracks: %s", exc)
            return []


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------
class WebSocketNaviSenseClient(NaviSenseClientBase):
    """Streams data from NaviSense over a WebSocket connection.

    Runs an asyncio event loop in the background.  Calling ``connect()``
    starts the background thread; ``close()`` shuts it down.

    Parameters
    ----------
    host : str
        NaviSense WebSocket server host.
    port : int
        NaviSense WebSocket server port.
    """

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        import threading

        self.uri = f"ws://{host}:{port}"
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_own_ship: OwnShipState = OwnShipState()
        self._latest_ais: List[SyntheticAISTrack] = []
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    def connect(self) -> None:
        """Start the background streaming thread."""
        import threading

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest_frame

    def get_own_ship(self) -> OwnShipState:
        with self._lock:
            return self._latest_own_ship

    def get_ais_tracks(self) -> List[SyntheticAISTrack]:
        with self._lock:
            return list(self._latest_ais)

    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._stream())
        finally:
            self._loop.close()

    async def _stream(self) -> None:
        try:
            import websockets  # type: ignore[import]
        except ImportError:
            logger.error("websockets library not installed; cannot use WebSocket mode.")
            return

        while self._running:
            try:
                async with websockets.connect(self.uri) as ws:
                    logger.info("Connected to NaviSense at %s", self.uri)
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        self._handle_message(raw_msg)
            except Exception as exc:
                logger.warning("NaviSense WebSocket error: %s – retrying in 2 s", exc)
                await asyncio.sleep(2)

    def _handle_message(self, raw_msg: str) -> None:
        try:
            msg = json.loads(raw_msg)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type")
        if msg_type == "frame":
            frame = self._decode_frame(msg.get("data", ""))
            with self._lock:
                self._latest_frame = frame
        elif msg_type == "own_ship":
            state = OwnShipState(
                lat=float(msg.get("lat", 0)),
                lon=float(msg.get("lon", 0)),
                heading_deg=float(msg.get("heading_deg", 0)),
                speed_kn=float(msg.get("speed_kn", 0)),
                timestamp=float(msg.get("timestamp", time.time())),
            )
            with self._lock:
                self._latest_own_ship = state
        elif msg_type == "ais":
            tracks = []
            for d in msg.get("tracks", []):
                tracks.append(
                    SyntheticAISTrack(
                        mmsi=int(d.get("mmsi", 0)),
                        lat=float(d.get("lat", 0)),
                        lon=float(d.get("lon", 0)),
                        cog=float(d.get("cog", 0)),
                        sog=float(d.get("sog", 0)),
                        timestamp=float(d.get("timestamp", time.time())),
                    )
                )
            with self._lock:
                self._latest_ais = tracks

    @staticmethod
    def _decode_frame(b64_data: str) -> Optional[np.ndarray]:
        try:
            import cv2

            raw = base64.b64decode(b64_data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.warning("Frame decode error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_client(config: dict) -> NaviSenseClientBase:
    """Instantiate the correct NaviSense client from config."""
    ns_cfg = config.get("navisense", {})
    mode = ns_cfg.get("mode", "disk")

    if mode == "websocket":
        ws_cfg = ns_cfg.get("websocket", {})
        client = WebSocketNaviSenseClient(
            host=ws_cfg.get("host", "localhost"),
            port=int(ws_cfg.get("port", 8765)),
        )
        client.connect()
        return client

    # Default: disk mode
    disk_cfg = ns_cfg.get("disk", {})
    return DiskNaviSenseClient(
        frame_dir=disk_cfg.get("frame_dir", "data/navisense/frames"),
        state_file=disk_cfg.get("state_file", "data/navisense/own_ship.json"),
        ais_file=disk_cfg.get("ais_file", "data/navisense/ais_tracks.json"),
    )
