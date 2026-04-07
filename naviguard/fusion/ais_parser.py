"""AIS parser for NaviGuard.

Provides an :class:`AISTrack` dataclass and an :class:`AISParser` that can
read AIS-like messages from:

- A JSON file on disk (written by NaviSense or a pre-recorded log)
- A UDP socket receiving JSON datagrams
- (future) raw NMEA sentences via a TCP stream

All backends return the same list of :class:`AISTrack` objects so that the
rest of the pipeline does not need to know the source.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AISTrack:
    """One AIS vessel report."""

    mmsi: int
    lat: float
    lon: float
    cog: float       # course over ground, degrees [0, 360)
    sog: float       # speed over ground, knots
    timestamp: float = field(default_factory=time.time)
    name: str = ""
    ship_type: int = 0


# ---------------------------------------------------------------------------
# Parser implementations
# ---------------------------------------------------------------------------

class JSONFileAISParser:
    """Read AIS tracks from a JSON file.

    The file is expected to contain a JSON array of objects::

        [
            {"mmsi": 123456789, "lat": 51.5, "lon": -0.09,
             "cog": 270.0, "sog": 5.0},
            …
        ]

    The file may be updated at any time; each call to ``poll()`` re-reads it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def poll(self) -> List[AISTrack]:
        if not self.path.exists():
            return []
        try:
            with open(self.path) as f:
                data = json.load(f)
            return [_dict_to_track(d) for d in data]
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("AIS JSON parse error (%s): %s", self.path, exc)
            return []


class UDPAISParser:
    """Receive AIS tracks as JSON datagrams over UDP.

    Each datagram must be either a single AIS dict or a JSON array of dicts.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5005, timeout: float = 0.01) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def open(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(self.timeout)
        logger.info("UDPAISParser listening on %s:%d", self.host, self.port)

    def poll(self) -> List[AISTrack]:
        if self._sock is None:
            return []
        tracks: List[AISTrack] = []
        while True:
            try:
                raw, _ = self._sock.recvfrom(65536)
                data = json.loads(raw.decode())
                if isinstance(data, list):
                    tracks.extend(_dict_to_track(d) for d in data)
                elif isinstance(data, dict):
                    tracks.append(_dict_to_track(data))
            except socket.timeout:
                break
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("AIS UDP parse error: %s", exc)
        return tracks

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class AISParser:
    """Unified AIS parser used by the NaviGuard pipeline.

    Delegates to the appropriate backend based on the source identifier.

    Parameters
    ----------
    source : str | Path | dict
        - A file path (str or Path) → :class:`JSONFileAISParser`
        - A dict with keys ``host`` / ``port`` → :class:`UDPAISParser`
        - ``None`` or empty string → returns an empty list every poll
    """

    def __init__(self, source) -> None:
        if isinstance(source, (str, Path)) and str(source).strip():
            self._backend = JSONFileAISParser(source)
        elif isinstance(source, dict):
            self._backend = UDPAISParser(
                host=source.get("host", "0.0.0.0"),
                port=int(source.get("port", 5005)),
            )
            self._backend.open()
        else:
            self._backend = _NullAISParser()

    def poll(self) -> List[AISTrack]:
        """Return the latest list of AIS tracks."""
        return self._backend.poll()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_track(d: dict) -> AISTrack:
    return AISTrack(
        mmsi=int(d.get("mmsi", 0)),
        lat=float(d.get("lat", 0.0)),
        lon=float(d.get("lon", 0.0)),
        cog=float(d.get("cog", 0.0)),
        sog=float(d.get("sog", 0.0)),
        timestamp=float(d.get("timestamp", time.time())),
        name=str(d.get("name", "")),
        ship_type=int(d.get("ship_type", 0)),
    )


class _NullAISParser:
    def poll(self) -> List[AISTrack]:
        return []
