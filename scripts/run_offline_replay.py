#!/usr/bin/env python3
"""Offline replay of a video + synthetic AIS log through the NaviGuard pipeline.

Usage
-----
    python scripts/run_offline_replay.py \\
        --video path/to/video.mp4 \\
        --ais   path/to/ais_log.json \\
        --config configs/naviguard_config.yaml \\
        [--speed 1.0] \\
        [--no-gui]

AIS log format (JSON array)::

    [
        {"timestamp": 0.0, "mmsi": 123, "lat": 51.5, "lon": -0.09,
         "cog": 270, "sog": 5},
        …
    ]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NaviGuard offline replay")
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--ais", default=None, help="Path to AIS log JSON (optional)")
    p.add_argument("--config", default="configs/naviguard_config.yaml")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    p.add_argument("--no-gui", action="store_true", help="Disable OpenCV display")
    p.add_argument("--output-video", default=None, help="Save annotated video to file")
    p.add_argument(
        "--own-lat", type=float, default=None,
        help="Own ship latitude (overrides config default)"
    )
    p.add_argument(
        "--own-lon", type=float, default=None,
        help="Own ship longitude (overrides config default)"
    )
    p.add_argument(
        "--own-heading", type=float, default=None,
        help="Own ship heading degrees (overrides config default)"
    )
    p.add_argument(
        "--own-speed-kn", type=float, default=None,
        help="Own ship speed in knots (overrides config default)"
    )
    return p.parse_args()


def load_ais_log(path: Optional[str]) -> List[dict]:
    if path is None:
        return []
    import json
    with open(path) as f:
        return json.load(f)


def get_ais_at_time(ais_log: List[dict], t: float, window: float = 5.0) -> List[dict]:
    """Return AIS records within *window* seconds of time *t*."""
    return [r for r in ais_log if abs(r.get("timestamp", 0) - t) <= window]


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    own_cfg = config.get("own_ship", {})
    own_lat = args.own_lat if args.own_lat is not None else float(own_cfg.get("lat", 51.5))
    own_lon = args.own_lon if args.own_lon is not None else float(own_cfg.get("lon", -0.09))
    own_heading = args.own_heading if args.own_heading is not None else float(
        own_cfg.get("heading_deg", 0.0)
    )
    own_speed_kn = args.own_speed_kn if args.own_speed_kn is not None else float(
        own_cfg.get("speed_kn", 5.0)
    )

    # ── Imports ──────────────────────────────────────────────────────
    import cv2

    from naviguard.perception.detector_rgb import RGBDetector
    from naviguard.perception.tracker import MultiObjectTracker
    from naviguard.fusion.ais_parser import AISTrack
    from naviguard.fusion.tracks import TrackManager
    from naviguard.risk.cpa_tcpa import cpa_tcpa
    from naviguard.risk.cri import assess_risk
    from naviguard.risk.colreg_logic import classify_encounter
    from naviguard.fusion.geometry import heading_to_velocity, knots_to_ms

    # ── Initialise components ────────────────────────────────────────
    detector = RGBDetector.from_config(config)
    try:
        detector.load()
        logger.info("Detector loaded.")
    except Exception as exc:
        logger.warning("Could not load detector weights (%s) – running without detection.", exc)
        detector = None  # type: ignore[assignment]

    tracker = MultiObjectTracker.from_config(config)
    track_manager = TrackManager.from_config(config)

    cam_cfg = config.get("camera", {})
    cam_fov = float(cam_cfg.get("fov_deg", 90.0))
    img_w = int(cam_cfg.get("img_width", 1280))
    img_h = int(cam_cfg.get("img_height", 720))
    cam_h = float(cam_cfg.get("mount_height_m", 3.0))
    risk_params = config.get("risk", {})

    if not args.no_gui:
        from naviguard.ui.viewer import NaviGuardViewer
        viewer = NaviGuardViewer()
    else:
        viewer = None

    # ── Load AIS log ─────────────────────────────────────────────────
    ais_log = load_ais_log(args.ais)
    logger.info("AIS log: %d records loaded.", len(ais_log))

    # ── Open video ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = 1.0 / (fps * args.speed)

    writer: Optional[cv2.VideoWriter] = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = img_w + (340 if viewer else 0)
        writer = cv2.VideoWriter(args.output_video, fourcc, fps, (out_w, img_h))

    logger.info("Starting offline replay at %.1f× speed …", args.speed)
    frame_idx = 0
    t_start = time.time()

    own_pos = np.zeros(2)
    own_vel = np.array(heading_to_velocity(own_heading, knots_to_ms(own_speed_kn)))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video.")
                break

            t_video = frame_idx / fps

            # ── Detection ────────────────────────────────────────────
            dets = detector.infer(frame) if detector else []

            # ── Tracking ─────────────────────────────────────────────
            vision_tracks = tracker.update(dets)

            # ── AIS ──────────────────────────────────────────────────
            raw_ais = get_ais_at_time(ais_log, t_video)
            ais_tracks = [
                AISTrack(
                    mmsi=int(r.get("mmsi", 0)),
                    lat=float(r.get("lat", own_lat)),
                    lon=float(r.get("lon", own_lon)),
                    cog=float(r.get("cog", 0)),
                    sog=float(r.get("sog", 0)),
                    timestamp=float(r.get("timestamp", t_video)),
                )
                for r in raw_ais
            ]

            # ── Fusion ───────────────────────────────────────────────
            unified = track_manager.update(
                vision_tracks=vision_tracks,
                ais_tracks=ais_tracks,
                own_lat=own_lat,
                own_lon=own_lon,
                own_heading_deg=own_heading,
                own_speed_kn=own_speed_kn,
                cam_fov_deg=cam_fov,
                img_width=img_w,
                img_height=img_h,
                camera_height_m=cam_h,
                ts=t_video,
            )

            # ── Risk assessment ───────────────────────────────────────
            risk_map: Dict = {}
            advice_map: Dict = {}
            for track in unified:
                tgt_pos = track.state[:2]
                tgt_vel = track.state[2:4]
                d_cpa, t_cpa_val = cpa_tcpa(own_pos, own_vel, tgt_pos, tgt_vel)
                risk = assess_risk(d_cpa, t_cpa_val, risk_params)
                risk_map[track.id] = risk

                tgt_heading = (
                    np.degrees(np.arctan2(tgt_vel[0], tgt_vel[1])) % 360.0
                    if np.linalg.norm(tgt_vel) > 0.01
                    else 0.0
                )
                adv = classify_encounter(
                    own_pos, own_vel, own_heading,
                    tgt_pos, tgt_vel, tgt_heading,
                    cri=risk.cri,
                )
                advice_map[track.id] = adv

            # ── UI ───────────────────────────────────────────────────
            if viewer:
                quit_requested = viewer.show(
                    frame, unified, risk_map, advice_map, dets
                )
                if quit_requested:
                    logger.info("User requested quit.")
                    break

            if writer:
                # Write just the frame for now (full panel would need resize logic)
                writer.write(frame)

            # ── Timing ───────────────────────────────────────────────
            elapsed = time.time() - t_start
            expected = frame_idx * frame_delay
            sleep_time = expected - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if viewer:
            viewer.close()
        logger.info("Replay finished. Processed %d frames.", frame_idx)


if __name__ == "__main__":
    main()
