"""Tests for naviguard.risk.cpa_tcpa"""

import math

import numpy as np
import pytest

from naviguard.risk.cpa_tcpa import cpa_tcpa, current_range, relative_bearing_deg


class TestCpaTcpa:
    def test_head_on_same_speed(self):
        """Two vessels approaching head-on at equal speed should have d_cpa ≈ 0."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([0.0, 5.0])   # heading north at 5 m/s
        tgt_pos = np.array([0.0, 100.0])
        tgt_vel = np.array([0.0, -5.0])  # heading south at 5 m/s

        d_cpa, t_cpa = cpa_tcpa(own_pos, own_vel, tgt_pos, tgt_vel)
        assert d_cpa == pytest.approx(0.0, abs=1e-6)
        assert t_cpa == pytest.approx(10.0, rel=1e-3)  # 100 m / 10 m/s = 10 s

    def test_parallel_tracks(self):
        """Parallel tracks have constant separation → d_cpa equals current range."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([0.0, 5.0])
        tgt_pos = np.array([50.0, 0.0])
        tgt_vel = np.array([0.0, 5.0])

        d_cpa, t_cpa = cpa_tcpa(own_pos, own_vel, tgt_pos, tgt_vel)
        assert d_cpa == pytest.approx(50.0, rel=1e-3)
        assert math.isinf(t_cpa)

    def test_diverging_vessels(self):
        """When vessels move apart, t_cpa should be negative (already past CPA)."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([0.0, -5.0])  # south
        tgt_pos = np.array([0.0, 50.0])
        tgt_vel = np.array([0.0, 5.0])   # north

        d_cpa, t_cpa = cpa_tcpa(own_pos, own_vel, tgt_pos, tgt_vel)
        assert t_cpa < 0

    def test_stationary_relative(self):
        """If both vessels have identical velocity, t_cpa is inf."""
        own_pos = np.array([0.0, 0.0])
        vel = np.array([3.0, 4.0])
        d_cpa, t_cpa = cpa_tcpa(own_pos, vel, np.array([10.0, 10.0]), vel)
        assert math.isinf(t_cpa)

    def test_current_range(self):
        own_pos = np.array([0.0, 0.0])
        tgt_pos = np.array([3.0, 4.0])
        assert current_range(own_pos, tgt_pos) == pytest.approx(5.0)

    def test_relative_bearing_ahead(self):
        own_pos = np.array([0.0, 0.0])
        own_heading = 0.0  # facing North
        tgt_pos = np.array([0.0, 100.0])  # directly ahead
        bearing = relative_bearing_deg(own_pos, own_heading, tgt_pos)
        assert bearing == pytest.approx(0.0, abs=1e-6)

    def test_relative_bearing_starboard(self):
        own_pos = np.array([0.0, 0.0])
        own_heading = 0.0
        tgt_pos = np.array([100.0, 0.0])  # due east = starboard
        bearing = relative_bearing_deg(own_pos, own_heading, tgt_pos)
        assert bearing == pytest.approx(90.0, abs=1e-3)
