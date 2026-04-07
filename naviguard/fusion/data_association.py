"""Vision ↔ AIS data association for NaviGuard.

Associates vision tracks (which have a bearing and estimated range derived
from pixel position + bbox size) with AIS tracks (which have a position
derived from GPS broadcasts).

The association is gated by:
- bearing difference (radians)
- range difference (metres)

A simple greedy O(N·M) matching is used.  For large traffic scenes a
Hungarian algorithm (scipy.optimize.linear_sum_assignment) would be better.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Set, Tuple

import numpy as np

from naviguard.fusion.ais_parser import AISTrack


# ---------------------------------------------------------------------------
# Input types expected by the associator
# ---------------------------------------------------------------------------

class VisionTrackView(NamedTuple):
    """Minimal projection of a vision track needed for association."""
    track_id: int
    bearing_rad: float      # relative bearing from camera centre
    range_m: float          # estimated range


class AISTrackView(NamedTuple):
    """Minimal projection of an AIS track needed for association."""
    mmsi: int
    bearing_rad: float      # predicted bearing from own ship
    range_m: float          # predicted range from own ship


# ---------------------------------------------------------------------------
# Association result
# ---------------------------------------------------------------------------

class AssociationResult(NamedTuple):
    """Output of :func:`associate`."""
    matches: List[Tuple[int, int]]  # (vision_idx, ais_idx)
    unmatched_vision: Set[int]
    unmatched_ais: Set[int]


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def associate(
    vision_tracks: List[VisionTrackView],
    ais_tracks: List[AISTrackView],
    bearing_gate_rad: float = 0.087,    # ≈ 5°
    range_gate_m: float = 200.0,
    w_bearing: float = 1.0,
    w_range: float = 0.005,             # scale range to similar magnitude as bearing
) -> AssociationResult:
    """Associate vision tracks with AIS tracks using greedy matching.

    Parameters
    ----------
    vision_tracks : list of VisionTrackView
    ais_tracks : list of AISTrackView
    bearing_gate_rad : float
        Maximum bearing difference (radians) for a valid match.
    range_gate_m : float
        Maximum range difference (metres) for a valid match.
    w_bearing : float
        Score weight for bearing error.
    w_range : float
        Score weight for range error (per metre).

    Returns
    -------
    AssociationResult
        Named tuple with (matches, unmatched_vision, unmatched_ais).
        ``matches`` is a list of ``(vision_idx, ais_idx)`` pairs.
    """
    nv = len(vision_tracks)
    na = len(ais_tracks)

    unmatched_vision: Set[int] = set(range(nv))
    unmatched_ais: Set[int] = set(range(na))
    matches: List[Tuple[int, int]] = []

    if nv == 0 or na == 0:
        return AssociationResult(matches, unmatched_vision, unmatched_ais)

    # Build score matrix (lower = better)
    score_matrix = np.full((nv, na), np.inf)
    for vi, vt in enumerate(vision_tracks):
        for ai, at in enumerate(ais_tracks):
            bearing_diff = abs(_angle_diff(vt.bearing_rad, at.bearing_rad))
            range_diff = abs(vt.range_m - at.range_m)
            if bearing_diff <= bearing_gate_rad and range_diff <= range_gate_m:
                score_matrix[vi, ai] = (
                    w_bearing * bearing_diff + w_range * range_diff
                )

    # Greedy matching
    matched_v: Set[int] = set()
    matched_a: Set[int] = set()

    while True:
        if np.all(np.isinf(score_matrix)):
            break
        flat_idx = int(np.argmin(score_matrix))
        vi = flat_idx // na
        ai = flat_idx % na
        if np.isinf(score_matrix[vi, ai]):
            break
        matches.append((vi, ai))
        matched_v.add(vi)
        matched_a.add(ai)
        score_matrix[vi, :] = np.inf
        score_matrix[:, ai] = np.inf

    unmatched_vision -= matched_v
    unmatched_ais -= matched_a
    return AssociationResult(matches, unmatched_vision, unmatched_ais)


# ---------------------------------------------------------------------------
# Helper: project AIS track to own-ship-relative bearing/range
# ---------------------------------------------------------------------------

def ais_to_bearing_range(
    ais: AISTrack,
    own_lat: float,
    own_lon: float,
    own_heading_deg: float,
) -> Tuple[float, float]:
    """Compute the bearing (relative to bow) and range of an AIS target.

    Parameters
    ----------
    ais : AISTrack
    own_lat, own_lon : float  – own ship position in decimal degrees
    own_heading_deg : float   – own ship heading in degrees

    Returns
    -------
    (bearing_rad, range_m)
        bearing_rad is relative to the ship's bow (positive = starboard).
    """
    from naviguard.fusion.geometry import latlon_to_enu

    east, north = latlon_to_enu(ais.lat, ais.lon, own_lat, own_lon)
    range_m = math.hypot(east, north)

    # Absolute bearing to target (0 = North)
    abs_bearing_deg = math.degrees(math.atan2(east, north)) % 360.0
    # Relative to ship bow
    rel_bearing_deg = (abs_bearing_deg - own_heading_deg) % 360.0
    # Wrap to [-180, 180]
    if rel_bearing_deg > 180.0:
        rel_bearing_deg -= 360.0
    return math.radians(rel_bearing_deg), range_m


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference (a − b) wrapped to (−π, π]."""
    diff = a - b
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff <= -math.pi:
        diff += 2 * math.pi
    return diff
