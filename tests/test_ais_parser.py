"""Tests for naviguard.fusion.ais_parser"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from naviguard.fusion.ais_parser import AISParser, AISTrack, JSONFileAISParser, _dict_to_track


class TestAISTrack:
    def test_default_values(self):
        t = AISTrack(mmsi=0, lat=0, lon=0, cog=0, sog=0)
        assert t.mmsi == 0
        assert t.name == ""

    def test_dict_to_track(self):
        d = {"mmsi": 123456789, "lat": 51.5, "lon": -0.09, "cog": 270.0, "sog": 5.0}
        t = _dict_to_track(d)
        assert t.mmsi == 123456789
        assert t.lat == pytest.approx(51.5)
        assert t.sog == pytest.approx(5.0)


class TestJSONFileAISParser:
    def test_empty_file(self, tmp_path):
        p = tmp_path / "ais.json"
        p.write_text("[]")
        parser = JSONFileAISParser(p)
        assert parser.poll() == []

    def test_valid_records(self, tmp_path):
        records = [
            {"mmsi": 1, "lat": 51.5, "lon": -0.09, "cog": 0.0, "sog": 3.0},
            {"mmsi": 2, "lat": 51.6, "lon": -0.10, "cog": 90.0, "sog": 5.0},
        ]
        p = tmp_path / "ais.json"
        p.write_text(json.dumps(records))
        parser = JSONFileAISParser(p)
        tracks = parser.poll()
        assert len(tracks) == 2
        assert tracks[0].mmsi == 1
        assert tracks[1].cog == pytest.approx(90.0)

    def test_missing_file(self, tmp_path):
        parser = JSONFileAISParser(tmp_path / "nonexistent.json")
        assert parser.poll() == []

    def test_corrupt_file(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not: valid json")
        parser = JSONFileAISParser(p)
        assert parser.poll() == []


class TestAISParser:
    def test_file_source(self, tmp_path):
        records = [{"mmsi": 99, "lat": 52.0, "lon": 0.0, "cog": 180.0, "sog": 2.0}]
        p = tmp_path / "ais.json"
        p.write_text(json.dumps(records))
        parser = AISParser(str(p))
        tracks = parser.poll()
        assert len(tracks) == 1
        assert tracks[0].mmsi == 99

    def test_none_source_returns_empty(self):
        parser = AISParser(None)
        assert parser.poll() == []

    def test_empty_string_source_returns_empty(self):
        parser = AISParser("")
        assert parser.poll() == []
