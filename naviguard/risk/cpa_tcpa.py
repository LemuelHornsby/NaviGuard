"""CPA / TCPA computation for NaviGuard.

Closest Point of Approach (CPA) and Time to Closest Point of Approach (TCPA)
are the two key scalars used to quantify collision risk between two vessels.

All positions and velocities are in ENU metres / metres-per-second.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def cpa_tcpa(
    own_pos: np.ndarray,
    own_vel: np.ndarray,
    tgt_pos: np.ndarray,
    tgt_vel: np.ndarray,
) -> Tuple[float, float]:
    """Compute CPA distance and TCPA between own ship and a target.

    Parameters
    ----------
    own_pos : np.ndarray shape (2,)
        Own ship position in ENU frame (metres).
    own_vel : np.ndarray shape (2,)
        Own ship velocity in ENU frame (m/s).
    tgt_pos : np.ndarray shape (2,)
        Target position in ENU frame (metres).
    tgt_vel : np.ndarray shape (2,)
        Target velocity in ENU frame (m/s).

    Returns
    -------
    (d_cpa, t_cpa) : Tuple[float, float]
        d_cpa – distance at closest point of approach (metres, >= 0).
        t_cpa – time to CPA (seconds).  Positive means CPA is in the future;
                negative means it already occurred.  ``math.inf`` if the
                vessels are stationary relative to each other.
    """
    own_pos = np.asarray(own_pos, dtype=float)
    own_vel = np.asarray(own_vel, dtype=float)
    tgt_pos = np.asarray(tgt_pos, dtype=float)
    tgt_vel = np.asarray(tgt_vel, dtype=float)

    r = tgt_pos - own_pos          # relative position
    v = tgt_vel - own_vel          # relative velocity

    v_norm2 = float(np.dot(v, v))
    if v_norm2 < 1e-9:             # vessels essentially stationary relative to each other
        return float(np.linalg.norm(r)), math.inf

    t_cpa = float(-np.dot(r, v) / v_norm2)
    r_cpa = r + t_cpa * v
    d_cpa = float(np.linalg.norm(r_cpa))
    return d_cpa, t_cpa


def current_range(own_pos: np.ndarray, tgt_pos: np.ndarray) -> float:
    """Return the current Euclidean range between own ship and target."""
    return float(np.linalg.norm(np.asarray(tgt_pos) - np.asarray(own_pos)))


def relative_bearing_deg(
    own_pos: np.ndarray,
    own_heading_deg: float,
    tgt_pos: np.ndarray,
) -> float:
    """Compute the bearing to the target relative to own ship's bow.

    Parameters
    ----------
    own_pos : np.ndarray (2,)  – own ship ENU position
    own_heading_deg : float    – own ship heading (0 = North, 90 = East)
    tgt_pos : np.ndarray (2,)  – target ENU position

    Returns
    -------
    float : bearing in degrees, wrapped to [0, 360).
        0 = dead ahead, 90 = starboard, 180 = astern, 270 = port.
    """
    delta = np.asarray(tgt_pos, dtype=float) - np.asarray(own_pos, dtype=float)
    abs_bearing = math.degrees(math.atan2(float(delta[0]), float(delta[1]))) % 360.0
    rel = (abs_bearing - own_heading_deg) % 360.0
    return rel
