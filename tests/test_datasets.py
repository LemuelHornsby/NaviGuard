"""Tests for naviguard.data_sources.datasets"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from naviguard.data_sources.datasets import CLASS_NAMES, INTERNAL_CLASSES, DatasetMapper, load_config


@pytest.fixture
def minimal_config(tmp_path):
    config = {
        "paths": {"datasets_root": str(tmp_path)},
        "label_map": {
            "seadronessee": {"boat": "vessel_small", "buoy": "buoy"},
            "modd": {"boat": "vessel_small"},
            "tisd": {"ship": "vessel_large"},
        },
    }
    return config


class TestInternalClasses:
    def test_class_ids_unique(self):
        ids = list(INTERNAL_CLASSES.values())
        assert len(ids) == len(set(ids))

    def test_class_names_inverse(self):
        for name, cid in INTERNAL_CLASSES.items():
            assert CLASS_NAMES[cid] == name


class TestDatasetMapper:
    def test_iter_samples_empty_dir(self, minimal_config):
        mapper = DatasetMapper(minimal_config)
        samples = list(mapper.iter_samples())
        assert samples == []

    def test_seadronessee_reader(self, minimal_config, tmp_path):
        """Create a minimal SeaDronesSee layout and verify it is read correctly."""
        root = tmp_path / "seadronessee"
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)

        # Write classes.txt
        (root / "classes.txt").write_text("boat\nbuoy\n")

        # Write a dummy image (1 pixel PNG)
        import struct, zlib
        def _minimal_png():
            sig = b'\x89PNG\r\n\x1a\n'
            ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
            raw = b'\x00\xff\x00\x00'  # filter byte + 1 RGB pixel
            compressed = zlib.compress(raw)
            idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
            idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
            iend_crc = zlib.crc32(b'IEND') & 0xffffffff
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
            return sig + ihdr + idat + iend

        img_path = root / "images" / "train" / "frame_001.png"
        img_path.write_bytes(_minimal_png())

        lbl_path = root / "labels" / "train" / "frame_001.txt"
        lbl_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

        mapper = DatasetMapper(minimal_config)
        samples = list(mapper.iter_samples())
        assert len(samples) == 1

        img, anns = samples[0]
        assert img.name == "frame_001.png"
        assert len(anns) == 2
        cls_ids = {a[0] for a in anns}
        assert INTERNAL_CLASSES["vessel_small"] in cls_ids
        assert INTERNAL_CLASSES["buoy"] in cls_ids

    def test_export_yolo(self, minimal_config, tmp_path):
        mapper = DatasetMapper(minimal_config)
        out = tmp_path / "yolo_out"
        # No samples – should still create directories without error
        mapper.export_yolo(out, split="train")
        assert (out / "train" / "images").is_dir()
        assert (out / "train" / "labels").is_dir()


class TestLoadConfig:
    def test_loads_yaml(self, tmp_path):
        cfg = {"paths": {"datasets_root": "data"}, "label_map": {}}
        p = tmp_path / "test_config.yaml"
        with open(p, "w") as f:
            yaml.dump(cfg, f)
        result = load_config(p)
        assert result["paths"]["datasets_root"] == "data"
