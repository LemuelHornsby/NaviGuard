"""Tests for naviguard.risk.cri"""

import math

import pytest

from naviguard.risk.cri import RiskLevel, assess_risk, risk_index


class TestRiskIndex:
    def test_zero_distance_gives_max_d_term(self):
        """d_cpa=0 with recent tcpa → CRI should be at maximum distance contribution."""
        cri = risk_index(d_cpa=0, t_cpa=0, critical_distance_m=100, critical_time_s=180)
        assert cri == pytest.approx(1.0, abs=1e-6)

    def test_safe_scenario(self):
        """Very large DCPA and TCPA → CRI near 0."""
        cri = risk_index(d_cpa=5000, t_cpa=3600, critical_distance_m=100, critical_time_s=180)
        assert cri == pytest.approx(0.0, abs=1e-6)

    def test_d_term_only(self):
        """TCPA = inf → only distance term contributes."""
        cri = risk_index(d_cpa=50, t_cpa=math.inf, critical_distance_m=100, critical_time_s=180,
                         w_d=0.6, w_t=0.4)
        expected = 0.6 * 0.5   # d_term = 1 - 50/100 = 0.5
        assert cri == pytest.approx(expected, rel=1e-6)

    def test_negative_tcpa_gives_zero_t_term(self):
        """Negative TCPA (CPA already passed) → t_term = 0."""
        cri = risk_index(d_cpa=50, t_cpa=-60, critical_distance_m=100, critical_time_s=180,
                         w_d=0.6, w_t=0.4)
        expected = 0.6 * 0.5  # t_term = 0 because t_cpa < 0
        assert cri == pytest.approx(expected, rel=1e-6)

    def test_weights_sum_to_one(self):
        """With d_cpa=0, t_cpa=0, CRI = w_d + w_t = 1."""
        cri = risk_index(d_cpa=0, t_cpa=0, w_d=0.5, w_t=0.5)
        assert cri == pytest.approx(1.0, abs=1e-6)


class TestAssessRisk:
    _params = {
        "critical_distance_m": 100.0,
        "critical_time_s": 180.0,
        "w_d": 0.6,
        "w_t": 0.4,
        "alert_thresholds": {"yellow": 0.3, "red": 0.6},
    }

    def test_green_level(self):
        result = assess_risk(500, math.inf, self._params)
        assert result.level == RiskLevel.GREEN
        assert result.cri < 0.3

    def test_yellow_level(self):
        # Half the critical distance, large tcpa → moderate risk
        result = assess_risk(50, 3600, self._params)
        assert result.level in (RiskLevel.GREEN, RiskLevel.YELLOW)

    def test_red_level(self):
        result = assess_risk(0, 0, self._params)
        assert result.level == RiskLevel.RED
        assert result.cri == pytest.approx(1.0, abs=1e-6)

    def test_result_contains_inputs(self):
        result = assess_risk(42.0, 90.0, self._params)
        assert result.d_cpa == pytest.approx(42.0)
        assert result.t_cpa == pytest.approx(90.0)
