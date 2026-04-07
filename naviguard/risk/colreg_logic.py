"""COLREG-aware encounter classification and evasion advice for NaviGuard.

Classifies encounters as head-on, crossing (give way / stand on), or
overtaking based on relative bearing and target aspect, then produces
a plain-English advisory consistent with the COLREGs (IRPCS).

Rules implemented (simplified advisory, not full legal compliance):
- Rule 13 Overtaking
- Rule 14 Head-on
- Rule 15 Crossing (starboard ↔ port give-way)

References
----------
COLREGS Part B, Section II (rules 12–18)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Encounter types
# ---------------------------------------------------------------------------

class EncounterType(Enum):
    HEAD_ON = "HEAD_ON"
    CROSSING_GIVE_WAY = "CROSSING_GIVE_WAY"   # we are the give-way vessel
    CROSSING_STAND_ON = "CROSSING_STAND_ON"   # we are the stand-on vessel
    OVERTAKING_US = "OVERTAKING_US"            # target is overtaking us
    OVERTAKING_THEM = "OVERTAKING_THEM"        # we are overtaking the target
    SAFE = "SAFE"                              # no significant encounter
    UNKNOWN = "UNKNOWN"


@dataclass
class COLREGAdvice:
    encounter: EncounterType
    advice: str
    urgency: str    # "routine" | "caution" | "immediate"


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

_HEAD_ON_SECTOR = 22.5       # ±22.5° = ±1 point ahead
_OVERTAKING_SECTOR = 135.0   # > 112.5° from target's stern = overtaking sector
_SAFE_CRI = 0.1              # below this CRI → SAFE, no classification needed


def classify_encounter(
    own_pos: np.ndarray,
    own_vel: np.ndarray,
    own_heading_deg: float,
    tgt_pos: np.ndarray,
    tgt_vel: np.ndarray,
    tgt_heading_deg: float,
    cri: float = 0.0,
) -> COLREGAdvice:
    """Classify a bilateral encounter and return COLREG-aware advice.

    Parameters
    ----------
    own_pos, own_vel : np.ndarray (2,)
        Own ship ENU position (m) and velocity (m/s).
    own_heading_deg : float
        Own ship heading (degrees, 0 = N).
    tgt_pos, tgt_vel : np.ndarray (2,)
        Target ENU position (m) and velocity (m/s).
    tgt_heading_deg : float
        Target heading (degrees, 0 = N).
    cri : float
        Collision Risk Index; if below ``_SAFE_CRI`` the encounter is
        immediately classified as SAFE.

    Returns
    -------
    COLREGAdvice
    """
    if cri < _SAFE_CRI:
        return COLREGAdvice(
            encounter=EncounterType.SAFE,
            advice="No action required.",
            urgency="routine",
        )

    # Bearing from own to target (relative to own bow, 0–360)
    bearing_own_to_tgt = _relative_bearing(own_pos, own_heading_deg, tgt_pos)
    # Bearing from target to own (relative to target bow, 0–360)
    bearing_tgt_to_own = _relative_bearing(tgt_pos, tgt_heading_deg, own_pos)

    # ── Rule 14: Head-on ──────────────────────────────────────────────────
    if (
        bearing_own_to_tgt <= _HEAD_ON_SECTOR or bearing_own_to_tgt >= 360 - _HEAD_ON_SECTOR
    ) and (
        bearing_tgt_to_own <= _HEAD_ON_SECTOR or bearing_tgt_to_own >= 360 - _HEAD_ON_SECTOR
    ):
        return COLREGAdvice(
            encounter=EncounterType.HEAD_ON,
            advice=(
                "HEAD-ON encounter (Rule 14): Both vessels must alter course to "
                "starboard. Alter course to starboard now."
            ),
            urgency=_urgency(cri),
        )

    # ── Rule 13: Overtaking ───────────────────────────────────────────────
    # Target is overtaking us if it approaches from > 112.5° abaft our beam
    if bearing_own_to_tgt > (180.0 + _OVERTAKING_SECTOR / 2) or \
       bearing_own_to_tgt < (180.0 - _OVERTAKING_SECTOR / 2) and bearing_own_to_tgt > 112.5:
        # approximate: target is coming from our stern sector
        pass  # fall through to overtaking check below

    own_speed = float(np.linalg.norm(own_vel))
    tgt_speed = float(np.linalg.norm(tgt_vel))
    closing = own_speed > 0 and tgt_speed > 0

    # Simpler overtaking test: target is abaft our beam and faster
    abaft_of_us = 112.5 < bearing_own_to_tgt < (360.0 - 112.5)
    # (wrapping check)
    abaft_of_us = bearing_own_to_tgt > 112.5 and bearing_own_to_tgt < 247.5

    if abaft_of_us and tgt_speed >= own_speed * 0.8:
        return COLREGAdvice(
            encounter=EncounterType.OVERTAKING_US,
            advice=(
                "OVERTAKING (Rule 13): Target is overtaking from astern. "
                "The overtaking vessel must keep clear. Monitor closely."
            ),
            urgency=_urgency(cri),
        )

    # ── Rule 15: Crossing ─────────────────────────────────────────────────
    # Target is to our starboard → we are give-way
    if 0 < bearing_own_to_tgt < 180.0:
        return COLREGAdvice(
            encounter=EncounterType.CROSSING_GIVE_WAY,
            advice=(
                "CROSSING (Rule 15): Target is on your starboard side – you are "
                "the give-way vessel. Alter course to starboard or reduce speed."
            ),
            urgency=_urgency(cri),
        )

    # Target is to our port → we are stand-on
    return COLREGAdvice(
        encounter=EncounterType.CROSSING_STAND_ON,
        advice=(
            "CROSSING (Rule 15): Target is on your port side – you are the "
            "stand-on vessel. Maintain course and speed, but be prepared to "
            "manoeuvre if the give-way vessel fails to act."
        ),
        urgency=_urgency(cri),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relative_bearing(
    from_pos: np.ndarray, from_heading_deg: float, to_pos: np.ndarray
) -> float:
    """Bearing from *from_pos* to *to_pos* relative to *from_heading_deg*.

    Returns bearing in [0, 360) degrees.
    """
    delta = np.asarray(to_pos, dtype=float) - np.asarray(from_pos, dtype=float)
    abs_bearing = math.degrees(math.atan2(float(delta[0]), float(delta[1]))) % 360.0
    rel = (abs_bearing - from_heading_deg) % 360.0
    return rel


def _urgency(cri: float) -> str:
    if cri >= 0.6:
        return "immediate"
    if cri >= 0.3:
        return "caution"
    return "routine"
