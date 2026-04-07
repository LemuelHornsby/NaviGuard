"""Thermal (infrared) object detector for NaviGuard.

Identical interface to :class:`~naviguard.perception.detector_rgb.RGBDetector`
but intended for thermal/FLIR-style single-channel or 3-channel IR images.
Thermal images are normalised to uint8 before inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


class ThermalDetector:
    """YOLO-based detector for thermal (IR) camera frames.

    Parameters
    ----------
    weights_path : str | Path
        Path to ``.pt`` or ``.onnx`` YOLO weights trained on thermal data.
    conf_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        IoU threshold for NMS.
    img_size : int
        Inference image size (square).
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self._model: Optional[Any] = None

    def load(self) -> None:
        """Load weights into memory."""
        try:
            from ultralytics import YOLO  # type: ignore[import]

            self._model = YOLO(str(self.weights_path))
            logger.info("ThermalDetector: loaded weights from %s", self.weights_path)
        except Exception as exc:
            logger.error("ThermalDetector: failed to load weights: %s", exc)
            raise

    @staticmethod
    def _prepare_frame(frame: np.ndarray) -> np.ndarray:
        """Normalise a thermal frame to uint8 BGR for YOLO inference.

        Handles:
        - Single-channel (H×W) → replicated to 3-channel BGR
        - 16-bit raw thermal → scaled to uint8
        - Already uint8 3-channel → returned as-is
        """
        if frame.dtype != np.uint8:
            # Scale to [0, 255]
            f_min, f_max = float(frame.min()), float(frame.max())
            if f_max > f_min:
                frame = ((frame.astype(np.float32) - f_min) / (f_max - f_min) * 255).astype(
                    np.uint8
                )
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)

        if frame.ndim == 2:
            # Single channel → 3-channel
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a thermal frame.

        Parameters
        ----------
        frame : np.ndarray
            Thermal image.  Any of: H×W (grey), H×W×1, or H×W×3.
            May be uint8 or uint16 raw thermal counts.

        Returns
        -------
        list of Detection dicts (same format as RGBDetector).
        """
        if self._model is None:
            raise RuntimeError("Call ThermalDetector.load() before infer().")

        prepared = self._prepare_frame(frame)
        results = self._model.predict(
            prepared,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
        )
        dets: List[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls)
                score = float(box.conf)
                dets.append({"bbox": (x1, y1, x2, y2), "cls": cls_id, "score": score})
        return dets

    @classmethod
    def from_config(cls, config: dict) -> "ThermalDetector":
        """Instantiate from a parsed NaviGuard config dict."""
        det_cfg = config["detector"]["thermal"]
        return cls(
            weights_path=det_cfg["weights"],
            conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
            iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
            img_size=int(det_cfg.get("img_size", 640)),
        )
