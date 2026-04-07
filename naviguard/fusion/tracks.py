"""Unified track manager for NaviGuard.

Maintains a table of :class:`UnifiedTrack` objects that may be:
- ``AIS_ONLY``    – visible in AIS but not yet confirmed by camera
- ``VISION_ONLY`` – detected by camera but no AIS match
- ``FUSED``       – matched in both AIS and camera

Each track keeps an ENU-frame Kalman filter for state estimation and a
configurable history buffer for playback / logging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from naviguard.fusion.ais_parser import AISTrack
from naviguard.fusion.data_association import (
    AssociationResult,
    AISTrackView,
    VisionTrackView,
    ais_to_bearing_range,
    associate,
)
from naviguard.fusion.geometry import (
    bbox_height_to_range,
    heading_to_velocity,
    knots_to_ms,
    latlon_to_enu,
    pixel_to_bearing,
    polar_to_enu,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track kind
# ---------------------------------------------------------------------------

class TrackKind(Enum):
    AIS_ONLY = "AIS_ONLY"
    VISION_ONLY = "VISION_ONLY"
    FUSED = "FUSED"


# ---------------------------------------------------------------------------
# Minimal Kalman filter (constant velocity, 2D ENU)
# ---------------------------------------------------------------------------

class _CVKalman:
    """2D constant-velocity Kalman filter for ENU position tracking."""

    def __init__(self) -> None:
        from filterpy.kalman import KalmanFilter  # type: ignore[import]

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        # State: [x, y, vx, vy]
        self.kf.F = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]], dtype=float
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.kf.R = np.eye(2) * 25.0           # measurement noise (5 m std)
        self.kf.Q = np.eye(4) * 0.1
        self.kf.P *= 500.0
        self._initialized = False

    def init(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        self.kf.x = np.array([[x], [y], [vx], [vy]], dtype=float)
        self._initialized = True

    def predict(self) -> np.ndarray:
        self.kf.predict()
        return self.kf.x[:2].flatten()

    def update(self, x: float, y: float) -> None:
        self.kf.update(np.array([[x], [y]], dtype=float))
        self._initialized = True

    @property
    def state(self) -> np.ndarray:
        """Return [x, y, vx, vy]."""
        return self.kf.x.flatten()


# ---------------------------------------------------------------------------
# UnifiedTrack
# ---------------------------------------------------------------------------

@dataclass
class UnifiedTrack:
    """A single track in the unified world frame."""

    id: int
    kind: TrackKind
    cls_label: int               # internal class id
    mmsi: Optional[int] = None   # set if AIS data is available
    vision_id: Optional[int] = None  # tracker-assigned vision ID

    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # state[0..1] = ENU position (m), state[2..3] = ENU velocity (m/s)

    last_update: float = field(default_factory=time.time)
    history: List[Tuple[float, np.ndarray]] = field(default_factory=list)
    hits: int = 0

    # Kalman filter (lazy-init)
    _kf: Optional[_CVKalman] = field(default=None, repr=False)

    def update_position(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        ts: Optional[float] = None,
    ) -> None:
        if self._kf is None:
            self._kf = _CVKalman()
            self._kf.init(x, y, vx, vy)
        else:
            self._kf.predict()
            self._kf.update(x, y)

        self.state = self._kf.state.copy()
        self.last_update = ts if ts is not None else time.time()
        self.hits += 1
        self.history.append((self.last_update, self.state.copy()))

    def predict_position(self) -> np.ndarray:
        if self._kf is None:
            return self.state[:2]
        return self._kf.predict()


# ---------------------------------------------------------------------------
# TrackManager
# ---------------------------------------------------------------------------

class TrackManager:
    """Manages the unified track table.

    Parameters
    ----------
    track_timeout_s : float
        Seconds without update before a track is removed.
    history_max_len : int
        Maximum number of state snapshots kept per track.
    bearing_gate_rad, range_gate_m : float
        Gating thresholds for vision↔AIS association.
    """

    def __init__(
        self,
        track_timeout_s: float = 15.0,
        history_max_len: int = 100,
        bearing_gate_rad: float = 0.087,
        range_gate_m: float = 200.0,
    ) -> None:
        self.track_timeout_s = track_timeout_s
        self.history_max_len = history_max_len
        self.bearing_gate_rad = bearing_gate_rad
        self.range_gate_m = range_gate_m

        self._tracks: Dict[int, UnifiedTrack] = {}
        self._next_id = 1
        self._mmsi_to_id: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Main update entry point
    # ------------------------------------------------------------------

    def update(
        self,
        vision_tracks: list,                  # list of tracker.Track dicts
        ais_tracks: List[AISTrack],
        own_lat: float,
        own_lon: float,
        own_heading_deg: float,
        own_speed_kn: float,
        cam_fov_deg: float = 90.0,
        img_width: int = 1280,
        img_height: int = 720,
        camera_height_m: float = 3.0,
        ts: Optional[float] = None,
    ) -> List[UnifiedTrack]:
        """Fuse vision tracks + AIS tracks and return the active track list.

        Parameters
        ----------
        vision_tracks : list
            Output of MultiObjectTracker.update() – each item is a dict
            with keys ``id``, ``bbox``, ``cls``, ``score``.
        ais_tracks : list of AISTrack
        own_* : own-ship state fields
        cam_fov_deg, img_width, img_height, camera_height_m : camera params
        ts : timestamp (defaults to now)
        """
        ts = ts if ts is not None else time.time()

        # ── Step 1: project vision tracks to bearing/range ──────────────
        vis_views: List[VisionTrackView] = []
        vis_enu: List[Tuple[float, float]] = []   # (east, north) per vision track
        for vt in vision_tracks:
            x1, y1, x2, y2 = vt["bbox"]
            cx = (x1 + x2) / 2.0
            bbox_h = y2 - y1
            bearing = pixel_to_bearing(cx, img_width, cam_fov_deg)
            range_m = bbox_height_to_range(bbox_h, img_height, camera_height_m)
            east, north = polar_to_enu(range_m, bearing)
            vis_views.append(VisionTrackView(vt["id"], bearing, range_m))
            vis_enu.append((east, north))

        # ── Step 2: project AIS tracks to bearing/range ──────────────────
        ais_views: List[AISTrackView] = []
        for at in ais_tracks:
            brg, rng = ais_to_bearing_range(at, own_lat, own_lon, own_heading_deg)
            ais_views.append(AISTrackView(at.mmsi, brg, rng))

        # ── Step 3: associate ────────────────────────────────────────────
        result: AssociationResult = associate(
            vis_views,
            ais_views,
            bearing_gate_rad=self.bearing_gate_rad,
            range_gate_m=self.range_gate_m,
        )

        # ── Step 4: update/create tracks ─────────────────────────────────
        own_vx, own_vy = heading_to_velocity(
            own_heading_deg, knots_to_ms(own_speed_kn)
        )

        # FUSED tracks
        for vi, ai in result.matches:
            vt = vision_tracks[vi]
            at = ais_tracks[ai]
            east, north = latlon_to_enu(at.lat, at.lon, own_lat, own_lon)
            vx, vy = heading_to_velocity(at.cog, knots_to_ms(at.sog))
            tid = self._get_or_create_ais_track(at.mmsi, vt["cls"])
            track = self._tracks[tid]
            track.kind = TrackKind.FUSED
            track.vision_id = vt["id"]
            track.update_position(east, north, vx, vy, ts)
            self._trim_history(track)

        # VISION_ONLY tracks
        for vi in result.unmatched_vision:
            vt = vision_tracks[vi]
            east, north = vis_enu[vi]
            tid = self._get_or_create_vision_track(vt["id"], vt["cls"])
            track = self._tracks[tid]
            track.update_position(east, north, ts=ts)
            self._trim_history(track)

        # AIS_ONLY tracks
        for ai in result.unmatched_ais:
            at = ais_tracks[ai]
            east, north = latlon_to_enu(at.lat, at.lon, own_lat, own_lon)
            vx, vy = heading_to_velocity(at.cog, knots_to_ms(at.sog))
            tid = self._get_or_create_ais_track(at.mmsi, cls_label=0)
            track = self._tracks[tid]
            track.kind = TrackKind.AIS_ONLY
            track.update_position(east, north, vx, vy, ts)
            self._trim_history(track)

        # ── Step 5: age out stale tracks ─────────────────────────────────
        self._expire_tracks(ts)

        return list(self._tracks.values())

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_all(self) -> List[UnifiedTrack]:
        return list(self._tracks.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def _get_or_create_ais_track(self, mmsi: int, cls_label: int) -> int:
        if mmsi in self._mmsi_to_id:
            return self._mmsi_to_id[mmsi]
        tid = self._new_id()
        self._tracks[tid] = UnifiedTrack(
            id=tid, kind=TrackKind.AIS_ONLY, cls_label=cls_label, mmsi=mmsi
        )
        self._mmsi_to_id[mmsi] = tid
        return tid

    def _get_or_create_vision_track(self, vision_id: int, cls_label: int) -> int:
        for tid, track in self._tracks.items():
            if track.vision_id == vision_id and track.kind in (
                TrackKind.VISION_ONLY, TrackKind.FUSED
            ):
                return tid
        tid = self._new_id()
        self._tracks[tid] = UnifiedTrack(
            id=tid,
            kind=TrackKind.VISION_ONLY,
            cls_label=cls_label,
            vision_id=vision_id,
        )
        return tid

    def _expire_tracks(self, ts: float) -> None:
        expired = [
            tid
            for tid, t in self._tracks.items()
            if ts - t.last_update > self.track_timeout_s
        ]
        for tid in expired:
            t = self._tracks.pop(tid)
            if t.mmsi is not None and self._mmsi_to_id.get(t.mmsi) == tid:
                del self._mmsi_to_id[t.mmsi]
            logger.debug("Track %d expired (kind=%s)", tid, t.kind.value)

    def _trim_history(self, track: UnifiedTrack) -> None:
        if len(track.history) > self.history_max_len:
            track.history = track.history[-self.history_max_len :]

    @classmethod
    def from_config(cls, config: dict) -> "TrackManager":
        f = config.get("fusion", {})
        return cls(
            track_timeout_s=float(f.get("track_timeout_s", 15.0)),
            history_max_len=int(f.get("history_max_len", 100)),
            bearing_gate_rad=float(f.get("bearing_gate_rad", 0.087)),
            range_gate_m=float(f.get("range_gate_m", 200.0)),
        )
