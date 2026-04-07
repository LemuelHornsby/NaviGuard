"""OpenCV-based visualiser for NaviGuard.

Draws annotated camera frames with:
- Bounding boxes colour-coded by risk level (green / yellow / red)
- Track ID, class label, CRI value
- Side panel listing active tracks with CPA/TCPA and advisory text
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from naviguard.data_sources.datasets import CLASS_NAMES
from naviguard.fusion.tracks import TrackKind, UnifiedTrack
from naviguard.risk.cri import RiskAssessment, RiskLevel
from naviguard.risk.colreg_logic import COLREGAdvice

logger = logging.getLogger(__name__)

# Colour map: risk level → BGR
_COLOURS: Dict[RiskLevel, Tuple[int, int, int]] = {
    RiskLevel.GREEN: (0, 200, 0),
    RiskLevel.YELLOW: (0, 200, 255),
    RiskLevel.RED: (0, 0, 220),
}

_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2
_PANEL_W = 340    # width of the side-panel in pixels


class NaviGuardViewer:
    """OpenCV window viewer for NaviGuard.

    Parameters
    ----------
    window_name : str
        Name of the OpenCV window.
    panel_width : int
        Width in pixels of the right-hand info panel.
    """

    def __init__(self, window_name: str = "NaviGuard", panel_width: int = _PANEL_W) -> None:
        import cv2  # local import

        self._cv2 = cv2
        self.window_name = window_name
        self.panel_width = panel_width
        self._window_created = False

    def show(
        self,
        frame_bgr: np.ndarray,
        tracks: List[UnifiedTrack],
        risk_map: Optional[Dict[int, RiskAssessment]] = None,
        advice_map: Optional[Dict[int, COLREGAdvice]] = None,
        vision_dets: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Render one frame and return True if the user pressed 'q' to quit.

        Parameters
        ----------
        frame_bgr : np.ndarray   H×W×3 BGR camera frame
        tracks    : list of UnifiedTrack
        risk_map  : {track_id → RiskAssessment} (optional)
        advice_map: {track_id → COLREGAdvice}   (optional)
        vision_dets: raw detector outputs (optional, drawn as thin boxes)

        Returns
        -------
        bool : True if user wants to quit (pressed 'q').
        """
        if not self._window_created:
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
            self._window_created = True

        risk_map = risk_map or {}
        advice_map = advice_map or {}
        canvas = frame_bgr.copy()

        # Raw detector boxes (optional, thin grey)
        if vision_dets:
            for det in vision_dets:
                x1, y1, x2, y2 = (int(v) for v in det["bbox"])
                self._cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 120), 1)

        # Track boxes
        for track in tracks:
            risk = risk_map.get(track.id)
            colour = _COLOURS.get(risk.level if risk else RiskLevel.GREEN, (0, 200, 0))

            # Draw box if we have a vision bbox approximation
            # (reconstruct bbox from ENU state – rough pixel reprojection)
            x_px, y_px = self._enu_to_pixel(track.state[:2], canvas.shape)
            if x_px is not None:
                r = max(5, 30 - int(math.hypot(*track.state[:2]) / 50))
                self._cv2.circle(canvas, (x_px, y_px), r, colour, _BOX_THICKNESS)
                label = self._build_label(track, risk)
                self._cv2.putText(
                    canvas, label,
                    (x_px + r + 2, y_px),
                    self._cv2.FONT_HERSHEY_SIMPLEX,
                    _FONT_SCALE, colour, _FONT_THICKNESS,
                )

        # Side panel
        panel = self._build_panel(canvas.shape[0], tracks, risk_map, advice_map)
        composite = np.hstack([canvas, panel])
        self._cv2.imshow(self.window_name, composite)
        key = self._cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def close(self) -> None:
        self._cv2.destroyWindow(self.window_name)
        self._window_created = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enu_to_pixel(
        self, enu_xy: np.ndarray, frame_shape: Tuple
    ) -> Tuple[Optional[int], Optional[int]]:
        """Very rough ENU → pixel mapping for visualisation purposes only.

        Treats the image centre as own-ship position, maps ± 500 m to
        the frame width/height.
        """
        h, w = frame_shape[:2]
        east, north = float(enu_xy[0]), float(enu_xy[1])
        scale = w / 1000.0   # 1000 m → full image width
        px = int(w / 2 + east * scale)
        py = int(h / 2 - north * scale)   # north is up
        if 0 <= px < w and 0 <= py < h:
            return px, py
        return None, None

    def _build_label(self, track: UnifiedTrack, risk: Optional[RiskAssessment]) -> str:
        cls_name = CLASS_NAMES.get(track.cls_label, str(track.cls_label))
        parts = [f"#{track.id}", cls_name, track.kind.value[0]]
        if risk:
            parts.append(f"CRI:{risk.cri:.2f}")
        return " ".join(parts)

    def _build_panel(
        self,
        height: int,
        tracks: List[UnifiedTrack],
        risk_map: Dict[int, RiskAssessment],
        advice_map: Dict[int, COLREGAdvice],
    ) -> np.ndarray:
        import cv2

        panel = np.zeros((height, self.panel_width, 3), dtype=np.uint8)
        # Header
        cv2.putText(panel, "NaviGuard – Tracks", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.line(panel, (0, 26), (self.panel_width, 26), (80, 80, 80), 1)

        y = 45
        line_h = 18
        for track in sorted(tracks, key=lambda t: t.id):
            risk = risk_map.get(track.id)
            adv = advice_map.get(track.id)
            colour = _COLOURS.get(risk.level if risk else RiskLevel.GREEN, (0, 200, 0))
            cls_name = CLASS_NAMES.get(track.cls_label, str(track.cls_label))

            cri_str = f"{risk.cri:.2f}" if risk else "n/a"
            dcpa_str = f"{risk.d_cpa:.0f}m" if risk else "n/a"
            tcpa_str = (
                f"{risk.t_cpa:.0f}s"
                if risk and not math.isinf(risk.t_cpa)
                else "∞"
            )

            line1 = f"#{track.id} {cls_name} [{track.kind.value}]"
            line2 = f"  CRI:{cri_str}  DCPA:{dcpa_str}  TCPA:{tcpa_str}"

            cv2.putText(panel, line1, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        _FONT_SCALE, colour, _FONT_THICKNESS)
            y += line_h
            cv2.putText(panel, line2, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        _FONT_SCALE, (180, 180, 180), _FONT_THICKNESS)
            y += line_h

            if adv and adv.urgency != "routine":
                # Wrap advice text
                words = adv.advice.split()
                line_buf = "  "
                for word in words:
                    if len(line_buf) + len(word) > 38:
                        cv2.putText(panel, line_buf, (8, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                    (200, 200, 50), 1)
                        y += 14
                        line_buf = "  " + word + " "
                    else:
                        line_buf += word + " "
                if line_buf.strip():
                    cv2.putText(panel, line_buf, (8, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                                (200, 200, 50), 1)
                    y += 14

            y += 4  # spacer between tracks
            if y > height - 10:
                break

        return panel
