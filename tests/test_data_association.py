"""Tests for naviguard.fusion.data_association"""

import math

import pytest

from naviguard.fusion.data_association import (
    AISTrackView,
    VisionTrackView,
    _angle_diff,
    associate,
)


class TestAngleDiff:
    def test_no_diff(self):
        assert _angle_diff(0.0, 0.0) == pytest.approx(0.0)

    def test_positive(self):
        assert _angle_diff(0.1, 0.0) == pytest.approx(0.1, abs=1e-9)

    def test_wrap_positive(self):
        diff = _angle_diff(math.pi - 0.01, -(math.pi - 0.01))
        assert abs(diff) <= math.pi

    def test_wrap_negative(self):
        diff = _angle_diff(-math.pi + 0.01, math.pi - 0.01)
        assert abs(diff) <= math.pi


class TestAssociate:
    def _vis(self, track_id, bearing, range_m):
        return VisionTrackView(track_id=track_id, bearing_rad=bearing, range_m=range_m)

    def _ais(self, mmsi, bearing, range_m):
        return AISTrackView(mmsi=mmsi, bearing_rad=bearing, range_m=range_m)

    def test_perfect_match(self):
        vis = [self._vis(1, 0.0, 100.0)]
        ais = [self._ais(1000, 0.0, 100.0)]
        result = associate(vis, ais, bearing_gate_rad=0.1, range_gate_m=50)
        assert len(result.matches) == 1
        assert result.matches[0] == (0, 0)
        assert len(result.unmatched_vision) == 0
        assert len(result.unmatched_ais) == 0

    def test_no_match_bearing_outside_gate(self):
        vis = [self._vis(1, 0.0, 100.0)]
        ais = [self._ais(1000, 1.0, 100.0)]  # 1 rad ≈ 57° difference
        result = associate(vis, ais, bearing_gate_rad=0.1, range_gate_m=50)
        assert len(result.matches) == 0

    def test_no_match_range_outside_gate(self):
        vis = [self._vis(1, 0.0, 100.0)]
        ais = [self._ais(1000, 0.0, 500.0)]   # 400 m range difference
        result = associate(vis, ais, bearing_gate_rad=0.1, range_gate_m=50)
        assert len(result.matches) == 0

    def test_empty_inputs(self):
        result = associate([], [], bearing_gate_rad=0.1, range_gate_m=50)
        assert result.matches == []
        assert len(result.unmatched_vision) == 0
        assert len(result.unmatched_ais) == 0

    def test_multiple_candidates_best_wins(self):
        vis = [self._vis(1, 0.0, 100.0)]
        ais = [
            self._ais(1000, 0.05, 100.0),  # slight bearing diff
            self._ais(2000, 0.01, 100.0),  # closer bearing – should win
        ]
        result = associate(vis, ais, bearing_gate_rad=0.1, range_gate_m=50)
        assert len(result.matches) == 1
        assert result.matches[0] == (0, 1)  # paired with ais index 1 (mmsi 2000)

    def test_unmatched_sets_correct(self):
        vis = [self._vis(1, 0.0, 100.0), self._vis(2, 0.5, 200.0)]
        ais = [self._ais(1000, 0.0, 100.0)]   # only one AIS track
        result = associate(vis, ais, bearing_gate_rad=0.1, range_gate_m=50)
        assert len(result.matches) == 1
        assert 1 in result.unmatched_vision   # vis index 1 (track id 2) is unmatched
