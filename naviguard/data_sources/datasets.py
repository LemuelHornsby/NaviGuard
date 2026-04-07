"""Dataset readers and label mappers for NaviGuard.

Supports:
- SeaDronesSee (RGB maritime object detection/tracking dataset)
- MODD / MODD2 (Marine Obstacle Detection Dataset)
- TISD (Thermal Infrared Ship Dataset)

All datasets are mapped to a unified internal class schema:
    0  vessel_large
    1  vessel_small
    2  buoy
    3  person
    4  shoreline
    5  floating_object
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Internal class IDs  (source of truth: configs/naviguard_config.yaml)
# ---------------------------------------------------------------------------
INTERNAL_CLASSES = {
    "vessel_large": 0,
    "vessel_small": 1,
    "buoy": 2,
    "person": 3,
    "shoreline": 4,
    "floating_object": 5,
}

# (class_id → class_name)
CLASS_NAMES = {v: k for k, v in INTERNAL_CLASSES.items()}

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# bbox is always (x_center, y_center, width, height) normalised to [0, 1]
BBox = Tuple[float, float, float, float]
Annotation = List[Tuple[int, BBox]]            # [(class_id, bbox), ...]
Sample = Tuple[Path, Annotation]               # (image_path, annotations)


# ---------------------------------------------------------------------------
# DatasetMapper
# ---------------------------------------------------------------------------
class DatasetMapper:
    """Unified reader for maritime datasets.

    Parameters
    ----------
    config : dict
        Parsed content of ``naviguard_config.yaml``.  At minimum, the
        following keys must be present::

            paths.datasets_root  – root folder that contains the datasets
            label_map            – per-source label → internal class name
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self._datasets_root = Path(config["paths"]["datasets_root"])
        self._label_map: dict[str, dict[str, str]] = config.get("label_map", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def iter_samples(self) -> Generator[Sample, None, None]:
        """Iterate over all samples from all configured datasets."""
        yield from self._iter_seadronessee()
        yield from self._iter_modd()
        yield from self._iter_tisd()

    def export_yolo(self, output_dir: str | Path, split: str = "train") -> None:
        """Write a YOLO-format dataset to *output_dir/split*.

        Creates::
            <output_dir>/<split>/images/<img_name>   (symlink or copy)
            <output_dir>/<split>/labels/<img_name>.txt
        """
        output_dir = Path(output_dir)
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, annotations in self.iter_samples():
            dst_img = img_dir / img_path.name
            # Prefer a symlink to avoid duplicating large image files.
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path.resolve())
                except OSError:
                    import shutil
                    shutil.copy2(img_path, dst_img)

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            with open(lbl_path, "w") as f:
                for cls_id, (cx, cy, w, h) in annotations:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _map_to_internal(
        self,
        img_path: Path,
        raw_annotations: list[tuple[str, BBox]],
        source: str,
    ) -> Sample:
        """Map source-specific labels → internal class IDs."""
        src_map = self._label_map.get(source, {})
        mapped: Annotation = []
        for cname, bbox in raw_annotations:
            internal_name = src_map.get(cname)
            if internal_name is None:
                continue
            cls_id = INTERNAL_CLASSES.get(internal_name)
            if cls_id is None:
                continue
            mapped.append((cls_id, bbox))
        return img_path, mapped

    # ------------------------------------------------------------------
    # SeaDronesSee reader
    # ------------------------------------------------------------------
    def _iter_seadronessee(self) -> Iterator[Sample]:
        """Iterate SeaDronesSee samples.

        Expected layout::
            <datasets_root>/seadronessee/
                images/<subset>/<img>.jpg
                labels/<subset>/<img>.txt   # YOLO format with SeaDronesSee labels
                classes.txt

        If the dataset is not present, iteration silently yields nothing.
        """
        root = self._datasets_root / "seadronessee"
        if not root.exists():
            return

        classes_file = root / "classes.txt"
        idx_to_name: dict[int, str] = {}
        if classes_file.exists():
            with open(classes_file) as f:
                for i, line in enumerate(f):
                    idx_to_name[i] = line.strip()

        labels_root = root / "labels"
        images_root = root / "images"

        for lbl_file in sorted(labels_root.rglob("*.txt")):
            rel = lbl_file.relative_to(labels_root)
            img_file = images_root / rel.with_suffix(".jpg")
            if not img_file.exists():
                img_file = images_root / rel.with_suffix(".png")
            if not img_file.exists():
                continue

            raw: list[tuple[str, BBox]] = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_idx = int(parts[0])
                    cname = idx_to_name.get(cls_idx, str(cls_idx))
                    bbox = tuple(float(p) for p in parts[1:5])
                    raw.append((cname, bbox))  # type: ignore[arg-type]

            yield self._map_to_internal(img_file, raw, "seadronessee")

    # ------------------------------------------------------------------
    # MODD / MODD2 reader
    # ------------------------------------------------------------------
    def _iter_modd(self) -> Iterator[Sample]:
        """Iterate MODD / MODD2 samples.

        Expected layout::
            <datasets_root>/modd/
                sequences/<seq_name>/
                    frames/<frame>.png
                    annotations/<frame>.json   # {objects: [{label, bbox_xyxy}]}
        """
        root = self._datasets_root / "modd"
        if not root.exists():
            return

        for ann_file in sorted(root.rglob("annotations/*.json")):
            seq_dir = ann_file.parent.parent
            frame_name = ann_file.stem
            img_file = seq_dir / "frames" / (frame_name + ".png")
            if not img_file.exists():
                img_file = seq_dir / "frames" / (frame_name + ".jpg")
            if not img_file.exists():
                continue

            with open(ann_file) as f:
                data = json.load(f)

            img_w = data.get("width", 1)
            img_h = data.get("height", 1)

            raw: list[tuple[str, BBox]] = []
            for obj in data.get("objects", []):
                cname: str = obj.get("label", "obstacle")
                x1, y1, x2, y2 = (
                    obj["bbox"][0],
                    obj["bbox"][1],
                    obj["bbox"][2],
                    obj["bbox"][3],
                )
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                raw.append((cname, (cx, cy, w, h)))

            yield self._map_to_internal(img_file, raw, "modd")

    # ------------------------------------------------------------------
    # TISD (thermal) reader
    # ------------------------------------------------------------------
    def _iter_tisd(self) -> Iterator[Sample]:
        """Iterate TISD (thermal ship detection dataset) samples.

        Expected layout::
            <datasets_root>/tisd/
                images/<img>.png
                labels/<img>.txt   # YOLO format with TISD labels
                classes.txt
        """
        root = self._datasets_root / "tisd"
        if not root.exists():
            return

        classes_file = root / "classes.txt"
        idx_to_name: dict[int, str] = {}
        if classes_file.exists():
            with open(classes_file) as f:
                for i, line in enumerate(f):
                    idx_to_name[i] = line.strip()

        for lbl_file in sorted((root / "labels").rglob("*.txt")):
            img_file = root / "images" / lbl_file.with_suffix(".png").name
            if not img_file.exists():
                img_file = root / "images" / lbl_file.with_suffix(".jpg").name
            if not img_file.exists():
                continue

            raw: list[tuple[str, BBox]] = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_idx = int(parts[0])
                    cname = idx_to_name.get(cls_idx, str(cls_idx))
                    bbox = tuple(float(p) for p in parts[1:5])
                    raw.append((cname, bbox))  # type: ignore[arg-type]

            yield self._map_to_internal(img_file, raw, "tisd")


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path = "configs/naviguard_config.yaml") -> dict:
    """Load and return the NaviGuard YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)
