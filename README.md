# NaviGuard

**Vision-based collision avoidance for yachts** — a maritime safety system that fuses camera detections, multi-object tracking, and AIS vessel data to compute collision risk and provide COLREG-aware evasion advice.

NaviGuard can be run in **offline-replay mode** (datasets / recorded videos), in **simulation mode** connected to the [NaviSense](https://github.com/LemuelHornsby/NaviSense) Unity simulator, or eventually with **live sensors** (IP cameras, AIS receivers, GPS).

---

## Repository layout

```
naviguard/
  data_sources/
    datasets.py          # SeaDronesSee / MODD / TISD readers + YOLO export
    navisense_client.py  # NaviSense Unity connector (disk polling or WebSocket)
    sensors_live.py      # Stubs for real cameras, AIS receiver, GPS
  perception/
    detector_rgb.py      # YOLO-based RGB object detector
    detector_thermal.py  # YOLO-based thermal object detector
    tracker.py           # IoU-based multi-object tracker (SORT-style)
  fusion/
    ais_parser.py        # AIS track reader (JSON file / UDP)
    geometry.py          # pixel->bearing, bbox->range, ENU conversions
    data_association.py  # Vision <-> AIS gated association
    tracks.py            # Unified track manager with Kalman filtering
  risk/
    cpa_tcpa.py          # Closest Point of Approach computation
    cri.py               # Collision Risk Index (CRI)
    colreg_logic.py      # COLREG encounter classification + advice
  ui/
    viewer.py            # OpenCV visualiser (bounding boxes + side panel)
configs/
  naviguard_config.yaml  # All configuration parameters
scripts/
  train_detector.py      # Train YOLOv8 on unified maritime dataset
  run_offline_replay.py  # Replay a video + AIS log through the pipeline
  run_naviguard_demo.py  # Live demo connected to NaviSense
tests/                   # Unit tests (pytest)
models/                  # Trained weights (place .pt / .onnx files here)
data/                    # Datasets and NaviSense output (not committed)
```

---

## Internal class schema

| ID | Name             | Description                          |
|----|------------------|--------------------------------------|
|  0 | vessel_large     | Ships, ferries, large power boats    |
|  1 | vessel_small     | Sailboats, kayaks, RIBs              |
|  2 | buoy             | Navigation buoys and markers         |
|  3 | person           | Swimmer / person overboard           |
|  4 | shoreline        | Quay wall, pier, shoreline           |
|  5 | floating_object  | Unknown floating hazard              |

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or editable install:
pip install -e .
```

### 2. Run tests

```bash
pytest tests/ -v
```

### 3. Offline replay (video + AIS log)

```bash
python scripts/run_offline_replay.py \
    --video  path/to/clip.mp4 \
    --ais    path/to/ais_log.json \
    --config configs/naviguard_config.yaml \
    --speed  1.0
```

Press **q** in the OpenCV window to quit.

**AIS log format** (JSON array):

```json
[
  {"timestamp": 0.0, "mmsi": 123456789, "lat": 51.5001, "lon": -0.089,
   "cog": 270.0, "sog": 5.0}
]
```

### 4. NaviSense live demo

```bash
# Disk mode:
python scripts/run_naviguard_demo.py \
    --mode disk \
    --frame-dir  data/navisense/frames \
    --state-file data/navisense/own_ship.json \
    --ais-file   data/navisense/ais_tracks.json

# WebSocket mode:
python scripts/run_naviguard_demo.py \
    --mode websocket \
    --ws-host localhost \
    --ws-port 8765
```

---

## Training a detector

```bash
python scripts/train_detector.py \
    --config configs/naviguard_config.yaml \
    --modality rgb \
    --epochs 100 \
    --model yolov8s.pt
```

Trained weights are saved to `models/naviguard-yolo-rgb.pt` and `models/naviguard-yolo-rgb.onnx`.

---

## Architecture

```
Camera frames --> RGBDetector --> MultiObjectTracker -->+
                                                        +--> TrackManager (fusion)
AIS log / socket --> AISParser ------------------------->
                                                        |
                                                        v
                                             UnifiedTrack table
                                                        |
                                       +----------------+---------------+
                                       v                                v
                                 cpa_tcpa()                  classify_encounter()
                                       |                                |
                                       v                                v
                                  risk_index()                  COLREGAdvice
                                       |
                                       v
                                NaviGuardViewer (OpenCV)
```

---

## Connecting to NaviSense

**Disk mode** - NaviSense writes files:
- `data/navisense/frames/frame_<ts>.jpg`
- `data/navisense/own_ship.json` -- `{"lat":...,"lon":...,"heading_deg":...,"speed_kn":...}`
- `data/navisense/ais_tracks.json` -- `[{"mmsi":...,"lat":...,"lon":...,"cog":...,"sog":...}]`

**WebSocket mode** - NaviSense streams JSON:

```json
{"type": "frame", "data": "<base64-JPEG>"}
{"type": "own_ship", "lat": 51.5, "lon": -0.09, "heading_deg": 45, "speed_kn": 5}
{"type": "ais", "tracks": [{"mmsi": 123, "lat": 51.51, "lon": -0.08, "cog": 270, "sog": 4}]}
```

---

## Datasets

| Dataset | Modality | URL |
|---------|----------|-----|
| SeaDronesSee | RGB (drone/USV) | https://seadronessee.cs.uni-tuebingen.de |
| MODD / MODD2 | RGB (USV) | https://www.vicos.si/resources/modd2/ |
| TISD | Thermal | arXiv |

---

## Roadmap

- [x] Phase 1: Repo skeleton, config, dataset readers
- [x] Phase 2: RGB/Thermal detectors, multi-object tracker
- [x] Phase 3: AIS parser, NaviSense connector
- [x] Phase 4: Geometry utilities, fusion, track manager
- [x] Phase 5: CPA/TCPA, CRI, COLREG logic
- [x] Phase 6: OpenCV UI, offline replay script
- [ ] Phase 7: NaviSense scenario manager, closed-loop demo
- [ ] Phase 8: Evaluation scripts (mAP, MOT, risk, COLREG score)
- [ ] Phase 9: Docker packaging, Jetson deployment
