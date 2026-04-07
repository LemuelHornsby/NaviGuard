"""Tests for naviguard.fusion.tracks (TrackManager)"""

import time

import numpy as np
import pytest

from naviguard.fusion.ais_parser import AISTrack
from naviguard.fusion.tracks import TrackKind, TrackManager, UnifiedTrack


def _make_ais(mmsi, lat=51.5, lon=-0.09, cog=0.0, sog=0.0):
    return AISTrack(mmsi=mmsi, lat=lat, lon=lon, cog=cog, sog=sog)


def _make_vis(track_id, x1=500, y1=300, x2=600, y2=400, cls=1, score=0.9):
    return {"id": track_id, "bbox": (x1, y1, x2, y2), "cls": cls, "score": score}


class TestTrackManager:
    def test_ais_only_track_created(self):
        tm = TrackManager()
        ais = [_make_ais(12345)]
        tracks = tm.update(
            vision_tracks=[], ais_tracks=ais,
            own_lat=51.5, own_lon=-0.09, own_heading_deg=0, own_speed_kn=0,
        )
        assert len(tracks) == 1
        assert tracks[0].kind == TrackKind.AIS_ONLY
        assert tracks[0].mmsi == 12345

    def test_vision_only_track_created(self):
        tm = TrackManager()
        vis = [_make_vis(1)]
        tracks = tm.update(
            vision_tracks=vis, ais_tracks=[],
            own_lat=51.5, own_lon=-0.09, own_heading_deg=0, own_speed_kn=0,
        )
        assert len(tracks) == 1
        assert tracks[0].kind == TrackKind.VISION_ONLY

    def test_track_id_is_stable(self):
        """The same AIS MMSI should map to the same track ID across updates."""
        tm = TrackManager()
        ais = [_make_ais(99999)]
        tracks1 = tm.update(
            vision_tracks=[], ais_tracks=ais,
            own_lat=51.5, own_lon=-0.09, own_heading_deg=0, own_speed_kn=0,
        )
        tracks2 = tm.update(
            vision_tracks=[], ais_tracks=ais,
            own_lat=51.5, own_lon=-0.09, own_heading_deg=0, own_speed_kn=0,
        )
        assert tracks1[0].id == tracks2[0].id

    def test_track_expires(self):
        tm = TrackManager(track_timeout_s=0.1)
        ais = [_make_ais(77777)]
        tm.update(
            vision_tracks=[], ais_tracks=ais,
            own_lat=51.5, own_lon=-0.09, own_heading_deg=0, own_speed_kn=0,
            ts=0.0,
        )
        # Update with no tracks after timeout
        tracks = tm.update(
            vision_tracks=[], ais_tracks=[],
            own_lat=51.5, own_lon=-0.09, own_heading_deg=0, own_speed_kn=0,
            ts=1.0,  # 1 s later – well past 0.1 s timeout
        )
        assert len(tracks) == 0

    def test_hits_increment(self):
        tm = TrackManager()
        ais = [_make_ais(11111)]
        for _ in range(3):
            tracks = tm.update(
                vision_tracks=[], ais_tracks=ais,
                own_lat=51.5, own_lon=-0.09, own_heading_deg=0, own_speed_kn=0,
            )
        assert tracks[0].hits == 3

    def test_history_bounded(self):
        tm = TrackManager(history_max_len=5)
        ais = [_make_ais(22222)]
        for i in range(20):
            tm.update(
                vision_tracks=[], ais_tracks=ais,
                own_lat=51.5, own_lon=-0.09, own_heading_deg=0, own_speed_kn=0,
            )
        track = tm.get_all()[0]
        assert len(track.history) <= 5
