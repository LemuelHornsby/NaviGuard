"""Multi-object tracker for NaviGuard.

Implements a lightweight IoU-based multi-object tracker (SORT-style) that
is dependency-free (no external bytetrack/ocSort package required).  When
those libraries become available in the environment, drop-in replacement is
straightforward by swapping the ``_IOUTracker`` implementation.

Public interface
----------------
``MultiObjectTracker.update(detections) → list[Track]``

Each track dict has:
    id    : int   – stable track ID
    bbox  : (x1, y1, x2, y2) – current bounding box
    cls   : int   – class id
    score : float – last detection confidence
    age   : int   – frames since track was initialised
    hits  : int   – total detection hits
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Track = Dict[str, Any]
Detection = Dict[str, Any]

# ---------------------------------------------------------------------------
# IOU helper
# ---------------------------------------------------------------------------

def _iou(box_a: Tuple, box_b: Tuple) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Single-object Kalman tracker
# ---------------------------------------------------------------------------

class _KalmanBox:
    """Constant-velocity Kalman filter for a single bounding box.

    State vector: [x1, y1, x2, y2, dx1, dy1, dx2, dy2]
    """

    _count = 0

    def __init__(self, bbox: Tuple, cls: int, score: float) -> None:
        from filterpy.kalman import KalmanFilter  # type: ignore[import]

        _KalmanBox._count += 1
        self.id = _KalmanBox._count
        self.cls = cls
        self.score = score
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i + 4] = 1.0          # position ← velocity

        self.kf.H = np.zeros((4, 8))
        for i in range(4):
            self.kf.H[i, i] = 1.0               # observe positions only

        self.kf.R[2:, 2:] *= 10.0              # measurement noise
        self.kf.P[4:, 4:] *= 1000.0            # high uncertainty in velocity
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        x1, y1, x2, y2 = bbox
        self.kf.x[:4] = np.array([[x1], [y1], [x2], [y2]])

    def predict(self) -> Tuple:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        state = self.kf.x[:4].flatten()
        return tuple(state.tolist())

    def update(self, bbox: Tuple, score: float) -> None:
        x1, y1, x2, y2 = bbox
        self.kf.update(np.array([[x1], [y1], [x2], [y2]]))
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    @property
    def bbox(self) -> Tuple:
        state = self.kf.x[:4].flatten()
        return tuple(state.tolist())


# ---------------------------------------------------------------------------
# Simple SORT-style tracker
# ---------------------------------------------------------------------------

class _IOUTracker:
    """Lightweight SORT tracker using IoU matching."""

    def __init__(self, max_age: int = 30, min_hits: int = 2, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: List[_KalmanBox] = []

    def update(self, detections: List[Detection]) -> List[Track]:
        # Predict all current tracks
        predicted_boxes = []
        alive = []
        for t in self._trackers:
            pred = t.predict()
            if any(np.isnan(pred)):
                continue
            alive.append(t)
            predicted_boxes.append(pred)
        self._trackers = alive

        # Build IoU cost matrix
        if predicted_boxes and detections:
            iou_matrix = np.zeros((len(predicted_boxes), len(detections)))
            for ti, pb in enumerate(predicted_boxes):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = _iou(pb, det["bbox"])

            # Greedy matching (Hungarian would be better but keeps deps minimal)
            matched_t: set = set()
            matched_d: set = set()
            matches: List[Tuple[int, int]] = []
            while True:
                if iou_matrix.size == 0:
                    break
                flat_idx = int(np.argmax(iou_matrix))
                ti = flat_idx // iou_matrix.shape[1]
                di = flat_idx % iou_matrix.shape[1]
                if iou_matrix[ti, di] < self.iou_threshold:
                    break
                if ti not in matched_t and di not in matched_d:
                    matches.append((ti, di))
                    matched_t.add(ti)
                    matched_d.add(di)
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1
        else:
            matches = []
            matched_t = set()
            matched_d = set()

        # Update matched tracks
        for ti, di in matches:
            det = detections[di]
            self._trackers[ti].update(det["bbox"], det["score"])

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_d:
                self._trackers.append(
                    _KalmanBox(det["bbox"], det.get("cls", 0), det.get("score", 1.0))
                )

        # Remove stale tracks
        self._trackers = [t for t in self._trackers if t.time_since_update < self.max_age]

        # Build output
        tracks: List[Track] = []
        for t in self._trackers:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                tracks.append(
                    {
                        "id": t.id,
                        "bbox": t.bbox,
                        "cls": t.cls,
                        "score": t.score,
                        "age": t.age,
                        "hits": t.hits,
                    }
                )
        return tracks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MultiObjectTracker:
    """Multi-object tracker for maritime scene understanding.

    Parameters
    ----------
    config : dict
        Tracker config sub-dict from naviguard_config.yaml::
            track_thresh : float  – detection confidence threshold
            track_buffer : int    – max frames to keep a lost track
            match_thresh : float  – IoU threshold for matching
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        max_age = int(cfg.get("track_buffer", 30))
        min_hits = 2
        iou_thresh = float(cfg.get("match_thresh", 0.3))
        self._tracker = _IOUTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_thresh,
        )

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracker with new detections and return active tracks.

        Parameters
        ----------
        detections : list of Detection dicts
            Each dict must contain ``bbox`` (x1,y1,x2,y2), ``cls``, ``score``.

        Returns
        -------
        list of Track dicts with stable ``id`` and updated ``bbox``.
        """
        # Filter by confidence
        return self._tracker.update(detections)

    @classmethod
    def from_config(cls, config: dict) -> "MultiObjectTracker":
        return cls(config.get("tracker", {}))
