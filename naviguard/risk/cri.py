"""Collision Risk Index (CRI) for NaviGuard.

The CRI combines normalised DCPA and TCPA into a single [0, 1] scalar:

    CRI = w_d · (1 − DCPA/d₀)⁺  +  w_t · (1 − |TCPA|/t₀)⁺

where  (·)⁺ = max(·, 0)  (clipped to [0, 1]).

Thresholds map CRI to traffic-light risk levels:
    CRI < yellow_threshold  → GREEN   (low risk)
    yellow ≤ CRI < red      → YELLOW  (moderate risk)
    CRI ≥ red_threshold     → RED     (high risk – action required)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Risk level
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class RiskAssessment(NamedTuple):
    """Output of :func:`assess_risk`."""
    cri: float           # Collision Risk Index in [0, 1]
    level: RiskLevel
    d_cpa: float         # DCPA in metres
    t_cpa: float         # TCPA in seconds


# ---------------------------------------------------------------------------
# CRI computation
# ---------------------------------------------------------------------------

def risk_index(
    d_cpa: float,
    t_cpa: float,
    critical_distance_m: float = 100.0,
    critical_time_s: float = 180.0,
    w_d: float = 0.6,
    w_t: float = 0.4,
) -> float:
    """Compute the Collision Risk Index (CRI).

    Parameters
    ----------
    d_cpa : float
        Distance at Closest Point of Approach (metres).
    t_cpa : float
        Time to CPA (seconds).  Can be ``math.inf`` if vessels are diverging.
    critical_distance_m : float
        Distance at which CRI distance term reaches 1.0.
    critical_time_s : float
        Time window inside which CRI time term reaches 1.0.
    w_d, w_t : float
        Weights for distance and time terms respectively (sum to 1).

    Returns
    -------
    float : CRI in [0, 1].  Higher = more dangerous.
    """
    # Distance term: larger DCPA → smaller risk
    d_term = max(0.0, 1.0 - d_cpa / critical_distance_m)
    d_term = min(d_term, 1.0)

    # Time term: TCPA in the future and small → higher risk
    if math.isinf(t_cpa) or t_cpa < 0:
        t_term = 0.0
    else:
        t_term = max(0.0, 1.0 - t_cpa / critical_time_s)
        t_term = min(t_term, 1.0)

    return w_d * d_term + w_t * t_term


def assess_risk(
    d_cpa: float,
    t_cpa: float,
    params: dict,
) -> RiskAssessment:
    """Compute CRI and map to a risk level.

    Parameters
    ----------
    d_cpa, t_cpa : float – from :func:`~naviguard.risk.cpa_tcpa.cpa_tcpa`.
    params : dict
        Sub-dict from ``naviguard_config.yaml``::
            critical_distance_m, critical_time_s, w_d, w_t,
            alert_thresholds.yellow, alert_thresholds.red

    Returns
    -------
    RiskAssessment
    """
    cri = risk_index(
        d_cpa=d_cpa,
        t_cpa=t_cpa,
        critical_distance_m=float(params.get("critical_distance_m", 100.0)),
        critical_time_s=float(params.get("critical_time_s", 180.0)),
        w_d=float(params.get("w_d", 0.6)),
        w_t=float(params.get("w_t", 0.4)),
    )

    thresholds = params.get("alert_thresholds", {})
    yellow_th = float(thresholds.get("yellow", 0.3))
    red_th = float(thresholds.get("red", 0.6))

    if cri >= red_th:
        level = RiskLevel.RED
    elif cri >= yellow_th:
        level = RiskLevel.YELLOW
    else:
        level = RiskLevel.GREEN

    return RiskAssessment(cri=cri, level=level, d_cpa=d_cpa, t_cpa=t_cpa)
