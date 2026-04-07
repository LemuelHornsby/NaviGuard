"""Tests for naviguard.risk.colreg_logic"""

import math

import numpy as np
import pytest

from naviguard.risk.colreg_logic import (
    COLREGAdvice,
    EncounterType,
    classify_encounter,
)

_OWN_POS = np.array([0.0, 0.0])


def _vel(heading_deg: float, speed: float = 5.0) -> np.ndarray:
    """Build a velocity vector from a compass heading (degrees)."""
    rad = math.radians(heading_deg)
    return np.array([speed * math.sin(rad), speed * math.cos(rad)])


class TestClassifyEncounter:
    def test_safe_when_cri_low(self):
        own_vel = _vel(0)
        tgt_pos = np.array([100.0, 200.0])
        tgt_vel = _vel(180)
        result = classify_encounter(
            _OWN_POS, own_vel, 0.0, tgt_pos, tgt_vel, 180.0, cri=0.05
        )
        assert result.encounter == EncounterType.SAFE

    def test_head_on(self):
        """Two vessels heading directly toward each other should be HEAD_ON."""
        own_vel = _vel(0)   # own heading North
        tgt_pos = np.array([0.0, 100.0])
        tgt_vel = _vel(180)  # target heading South (toward own)
        result = classify_encounter(
            _OWN_POS, own_vel, 0.0, tgt_pos, tgt_vel, 180.0, cri=0.7
        )
        assert result.encounter == EncounterType.HEAD_ON
        assert "starboard" in result.advice.lower()

    def test_crossing_starboard_give_way(self):
        """Target crossing from starboard → own ship is give-way."""
        own_vel = _vel(0)    # own heading North
        tgt_pos = np.array([100.0, 0.0])   # target to starboard (east)
        tgt_vel = _vel(270)   # target heading West – crossing
        result = classify_encounter(
            _OWN_POS, own_vel, 0.0, tgt_pos, tgt_vel, 270.0, cri=0.5
        )
        assert result.encounter == EncounterType.CROSSING_GIVE_WAY

    def test_crossing_port_stand_on(self):
        """Target crossing from port → own ship is stand-on."""
        own_vel = _vel(0)
        tgt_pos = np.array([-100.0, 0.0])   # port (west)
        tgt_vel = _vel(90)    # heading east
        result = classify_encounter(
            _OWN_POS, own_vel, 0.0, tgt_pos, tgt_vel, 90.0, cri=0.5
        )
        assert result.encounter == EncounterType.CROSSING_STAND_ON

    def test_urgency_levels(self):
        own_vel = _vel(0)
        tgt_pos = np.array([100.0, 0.0])
        tgt_vel = _vel(270)
        low = classify_encounter(
            _OWN_POS, own_vel, 0.0, tgt_pos, tgt_vel, 270.0, cri=0.2
        )
        med = classify_encounter(
            _OWN_POS, own_vel, 0.0, tgt_pos, tgt_vel, 270.0, cri=0.4
        )
        high = classify_encounter(
            _OWN_POS, own_vel, 0.0, tgt_pos, tgt_vel, 270.0, cri=0.8
        )
        assert low.urgency == "routine"
        assert med.urgency == "caution"
        assert high.urgency == "immediate"
