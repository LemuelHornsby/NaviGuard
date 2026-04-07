#!/usr/bin/env python3
"""Train a YOLO detector on the unified NaviGuard maritime dataset.

Usage
-----
    python scripts/train_detector.py \\
        --config configs/naviguard_config.yaml \\
        --modality rgb \\
        --epochs 100 \\
        --output-dir data/yolo_dataset

The script:
1. Uses DatasetMapper to export a YOLO-format dataset to *output_dir*.
2. Writes a ``dataset.yaml`` compatible with Ultralytics.
3. Launches YOLO training with strong maritime augmentations.
4. Exports the best checkpoint to ONNX.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allow running from repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NaviGuard YOLO detector")
    p.add_argument("--config", default="configs/naviguard_config.yaml")
    p.add_argument(
        "--modality", choices=["rgb", "thermal"], default="rgb",
        help="Which detector variant to train."
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output-dir", default="data/yolo_dataset",
                   help="Directory to write the YOLO dataset layout.")
    p.add_argument("--model", default="yolov8s.pt",
                   help="Ultralytics model name or path to start from.")
    p.add_argument("--device", default="", help="cuda device or 'cpu'.")
    p.add_argument("--no-export", action="store_true",
                   help="Skip ONNX export after training.")
    return p.parse_args()


def build_dataset_yaml(output_dir: Path, class_names: dict) -> Path:
    """Write a dataset.yaml for Ultralytics training."""
    nc = len(class_names)
    names = [class_names[i] for i in range(nc)]
    ds_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": nc,
        "names": names,
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(ds_yaml, f, default_flow_style=False)
    logger.info("Dataset YAML written to %s", yaml_path)
    return yaml_path


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    from naviguard.data_sources.datasets import DatasetMapper, CLASS_NAMES

    output_dir = Path(args.output_dir)

    # ── Export dataset ────────────────────────────────────────────────
    logger.info("Exporting unified YOLO dataset to %s …", output_dir)
    mapper = DatasetMapper(config)
    mapper.export_yolo(output_dir, split="train")
    # For a real run you would export a separate val split from held-out data.
    # Here we symlink train as val for demonstration purposes.
    val_dir = output_dir / "val"
    if not val_dir.exists():
        import shutil
        shutil.copytree(output_dir / "train", val_dir, symlinks=True)
        logger.info("val/ created as a copy of train/ – replace with real val data.")

    # ── Write dataset.yaml ────────────────────────────────────────────
    dataset_yaml = build_dataset_yaml(output_dir, CLASS_NAMES)

    # ── YOLO training ─────────────────────────────────────────────────
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        logger.error("Ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)
    model_suffix = args.modality

    augment_kwargs = dict(
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
        # Simulate fog / haze: use albumentations in data pipeline or custom
    )

    project_name = f"naviguard-yolo-{model_suffix}"
    logger.info("Starting training: %s", project_name)

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        project="runs/detect",
        name=project_name,
        exist_ok=True,
        **augment_kwargs,
    )

    # ── ONNX export ──────────────────────────────────────────────────
    if not args.no_export:
        best_weights = Path(f"runs/detect/{project_name}/weights/best.pt")
        if best_weights.exists():
            export_model = YOLO(str(best_weights))
            export_path = export_model.export(format="onnx", imgsz=args.imgsz)
            # Copy into models/
            models_dir = Path(config["paths"]["models_dir"])
            models_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(export_path, models_dir / f"naviguard-yolo-{model_suffix}.onnx")
            shutil.copy(str(best_weights), models_dir / f"naviguard-yolo-{model_suffix}.pt")
            logger.info("Model exported to %s/", models_dir)
        else:
            logger.warning("best.pt not found; skipping export.")


if __name__ == "__main__":
    main()
