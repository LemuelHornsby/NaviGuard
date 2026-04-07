"""RGB object detector for NaviGuard.

Wraps a YOLOv8 model (via the Ultralytics library) and provides a clean
``infer()`` interface that returns a list of detection dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]
# Each detection dict has keys:
#   bbox  : (x1, y1, x2, y2)  – pixel coordinates, absolute
#   cls   : int                – internal class id
#   score : float              – confidence in [0, 1]


class RGBDetector:
    """YOLO-based detector for RGB camera frames.

    Parameters
    ----------
    weights_path : str | Path
        Path to a ``.pt`` or ``.onnx`` YOLO weights file.
    conf_threshold : float
        Minimum confidence score to keep a detection.
    iou_threshold : float
        IoU threshold for NMS.
    img_size : int
        Inference image size (square).
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        img_size: int = 640,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self._model: Optional[Any] = None

    def load(self) -> None:
        """Load weights into memory.  Call once before using ``infer()``."""
        try:
            from ultralytics import YOLO  # type: ignore[import]

            self._model = YOLO(str(self.weights_path))
            logger.info("RGBDetector: loaded weights from %s", self.weights_path)
        except Exception as exc:
            logger.error("RGBDetector: failed to load weights: %s", exc)
            raise

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            H×W×3 uint8 image in BGR colour order (OpenCV convention).

        Returns
        -------
        list of Detection dicts.
        """
        if self._model is None:
            raise RuntimeError("Call RGBDetector.load() before infer().")

        results = self._model.predict(
            frame_bgr,
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
    def from_config(cls, config: dict) -> "RGBDetector":
        """Instantiate from a parsed NaviGuard config dict."""
        det_cfg = config["detector"]["rgb"]
        return cls(
            weights_path=det_cfg["weights"],
            conf_threshold=float(det_cfg.get("conf_threshold", 0.30)),
            iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
            img_size=int(det_cfg.get("img_size", 640)),
        )
