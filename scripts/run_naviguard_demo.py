#!/usr/bin/env python3
"""Live NaviGuard demo connected to the NaviSense Unity simulator.

Usage
-----
    # Disk mode (NaviSense writes files to disk):
    python scripts/run_naviguard_demo.py --mode disk \\
        --frame-dir data/navisense/frames \\
        --state-file data/navisense/own_ship.json \\
        --ais-file  data/navisense/ais_tracks.json

    # WebSocket mode (NaviSense streams data in real time):
    python scripts/run_naviguard_demo.py --mode websocket \\
        --ws-host localhost --ws-port 8765
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NaviGuard live demo / NaviSense connector")
    p.add_argument("--config", default="configs/naviguard_config.yaml")
    p.add_argument("--mode", choices=["disk", "websocket"], default="disk")
    p.add_argument("--frame-dir", default="data/navisense/frames")
    p.add_argument("--state-file", default="data/navisense/own_ship.json")
    p.add_argument("--ais-file", default="data/navisense/ais_tracks.json")
    p.add_argument("--ws-host", default="localhost")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--fps-limit", type=float, default=25.0,
                   help="Maximum frames per second to process.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override transport config from CLI
    config["navisense"]["mode"] = args.mode
    if args.mode == "disk":
        config["navisense"]["disk"]["frame_dir"] = args.frame_dir
        config["navisense"]["disk"]["state_file"] = args.state_file
        config["navisense"]["disk"]["ais_file"] = args.ais_file
    else:
        config["navisense"]["websocket"]["host"] = args.ws_host
        config["navisense"]["websocket"]["port"] = args.ws_port

    # ── Imports ──────────────────────────────────────────────────────
    from naviguard.data_sources.navisense_client import create_client
    from naviguard.fusion.ais_parser import AISTrack
    from naviguard.fusion.geometry import heading_to_velocity, knots_to_ms
    from naviguard.fusion.tracks import TrackManager
    from naviguard.perception.detector_rgb import RGBDetector
    from naviguard.perception.tracker import MultiObjectTracker
    from naviguard.risk.colreg_logic import classify_encounter
    from naviguard.risk.cpa_tcpa import cpa_tcpa
    from naviguard.risk.cri import assess_risk

    # ── Initialise NaviSense client ──────────────────────────────────
    client = create_client(config)
    logger.info("NaviSense client ready (mode=%s).", args.mode)

    # ── Initialise pipeline components ───────────────────────────────
    detector = RGBDetector.from_config(config)
    try:
        detector.load()
        logger.info("Detector loaded.")
    except Exception as exc:
        logger.warning("Detector weights not available (%s) – skipping detection.", exc)
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

    frame_period = 1.0 / args.fps_limit
    logger.info("NaviGuard demo running.  Press 'q' to quit.")

    try:
        while True:
            t0 = time.time()

            # ── Get data from NaviSense ────────────────────────────
            frame = client.get_frame()
            own_ship = client.get_own_ship()
            sim_ais = client.get_ais_tracks()

            if frame is None:
                time.sleep(0.05)
                continue

            own_pos = np.zeros(2)
            own_vel = np.array(
                heading_to_velocity(own_ship.heading_deg, knots_to_ms(own_ship.speed_kn))
            )

            # ── Detection ────────────────────────────────────────────
            dets = detector.infer(frame) if detector else []

            # ── Tracking ─────────────────────────────────────────────
            vision_tracks = tracker.update(dets)

            # ── Convert sim AIS to AISTrack ──────────────────────────
            ais_tracks = [
                AISTrack(
                    mmsi=a.mmsi,
                    lat=a.lat,
                    lon=a.lon,
                    cog=a.cog,
                    sog=a.sog,
                    timestamp=a.timestamp,
                )
                for a in sim_ais
            ]

            # ── Fusion ───────────────────────────────────────────────
            unified = track_manager.update(
                vision_tracks=vision_tracks,
                ais_tracks=ais_tracks,
                own_lat=own_ship.lat,
                own_lon=own_ship.lon,
                own_heading_deg=own_ship.heading_deg,
                own_speed_kn=own_ship.speed_kn,
                cam_fov_deg=cam_fov,
                img_width=img_w,
                img_height=img_h,
                camera_height_m=cam_h,
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
                    float(np.degrees(np.arctan2(tgt_vel[0], tgt_vel[1]))) % 360.0
                    if np.linalg.norm(tgt_vel) > 0.01
                    else 0.0
                )
                adv = classify_encounter(
                    own_pos, own_vel, own_ship.heading_deg,
                    tgt_pos, tgt_vel, tgt_heading,
                    cri=risk.cri,
                )
                advice_map[track.id] = adv

            # ── UI ───────────────────────────────────────────────────
            if viewer:
                quit_req = viewer.show(frame, unified, risk_map, advice_map, dets)
                if quit_req:
                    break

            # ── Frame-rate limiter ────────────────────────────────────
            elapsed = time.time() - t0
            sleep = frame_period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    finally:
        client.close()
        if viewer:
            viewer.close()
        logger.info("NaviGuard demo stopped.")


if __name__ == "__main__":
    main()
