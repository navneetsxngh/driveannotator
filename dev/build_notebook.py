# Generates 01_build_dataset.ipynb. Cells tagged "# [DEFS]" are pure definitions that
# test_notebook.py executes locally against synthetic data.
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []


def md(src):
    cells.append(("markdown", src.strip("\n")))


def code(src):
    cells.append(("code", src.strip("\n")))


# ---------------------------------------------------------------------------
md(r'''
# DriveAnnotator — Step 1: Build the dataset

## What this notebook does

- It downloads real driving clips from NVIDIA's `PhysicalAI-Autonomous-Vehicles` dataset on Hugging Face.
- From every clip it takes 5 moments in time.
- For each moment it saves 24 camera images (6 cameras × 4 frames).
- This is exactly the input that `Alpamayo2-Edge` expects.
- It creates a label for each moment automatically.
- No training happens here. Training is in `02_train_lora.ipynb`.

## How the labels are made

- The dataset comes with 3D boxes around cars, people and other road users (made with LiDAR).
- We project those boxes into each camera image.
- Simple rules turn this into a label: lighting, visible road users, which camera sees them, and hazards.
- Later, the model learns to produce the same label from the camera images alone.

## Steps

| Step | What happens |
|---|---|
| 1 | Install and import libraries |
| 2 | Settings |
| 3 | Folders and config lock |
| 4 | Log in to Hugging Face |
| 5 | Load the dataset index |
| 6 | Pick the clips |
| 7 | Estimate the download size |
| 8 | Read the 3D box file |
| 9 | Place the boxes on the images and check them by eye |
| 10 | Label rules |
| 11 | Try one clip first |
| 12 | Build all clips |
| 13 | Write the train / val / test files |
| 14 | Check the numbers |
| 15–16 | Review test labels by hand (optional) |
| 17 | Write the dataset card |

## Before you start

1. Open https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles, log in, and accept the license.
2. Create a **Read** token at https://huggingface.co/settings/tokens.
3. In Colab, add the token in the 🔑 **Secrets** panel as `HF_TOKEN`. Locally, put it in the `.env` file.
4. Use a **CPU runtime**. This notebook does not need a GPU.
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 1 — Install and import libraries

- In Colab, this cell installs the libraries.
- Locally, run `pip install -r requirements.txt` once before opening the notebook.
- `physical_ai_av` is NVIDIA's helper library for this dataset. It downloads clips, decodes videos and reads calibration.
- `scipy` must be version 1.16 or newer, because the helper library needs a feature added in 1.16.
- The cell stops with a clear message if a version is wrong.
''')

code(r'''
# ==== Cell 1: Environment setup + imports ====
import sys, os, subprocess, re, shutil

IN_COLAB = "google.colab" in sys.modules
REQUIREMENTS = [
    "physical_ai_av==0.2.3", "huggingface_hub>=1.0,<2", "av>=14.4.0",
    "numpy>=2.0", "pandas>=2.2", "pyarrow>=17.0", "scipy>=1.16",
    "pillow>=11.0", "tqdm>=4.66", "ipywidgets>=8.1", "matplotlib>=3.9",
]
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS], check=True)

# Local: read .env (HF_TOKEN, paths) BEFORE huggingface_hub import, so HF_* settings apply.
ENV_FILE = os.path.join(os.getcwd(), ".env")
ENV_LOADED = False
if not IN_COLAB:
    from dotenv import load_dotenv
    ENV_LOADED = load_dotenv(ENV_FILE, override=False)

import json, math, time, io, zipfile, getpass, traceback, threading, random, datetime, logging, hashlib, warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.transform import Rotation, RigidTransform
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import huggingface_hub
import physical_ai_av


def _ver(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


assert sys.version_info >= (3, 11), f"Python 3.11 or newer is needed, found {sys.version}"
assert physical_ai_av.__version__ == "0.2.3", f"physical_ai_av 0.2.3 is needed, found {physical_ai_av.__version__}"
assert _ver(scipy.__version__) >= (1, 16, 0), f"scipy 1.16 or newer is needed, found {scipy.__version__}"
assert _ver(huggingface_hub.__version__) >= (1, 0, 0), f"huggingface_hub 1.0 or newer is needed, found {huggingface_hub.__version__}"

logging.basicConfig(level=logging.WARNING)
logging.getLogger("physical_ai_av").setLevel(logging.INFO)
logging.getLogger("physical_ai_av.dataset").setLevel(logging.ERROR)   # "reader not implemented" = harmless

print(f"Python          {sys.version.split()[0]}")
print(f"physical_ai_av  {physical_ai_av.__version__}")
print(f"huggingface_hub {huggingface_hub.__version__}")
print(f"numpy {np.__version__} | pandas {pd.__version__} | scipy {scipy.__version__}")
print(f"Colab: {IN_COLAB} | CPU cores: {os.cpu_count()}")
if not IN_COLAB:
    print(f".env: {'loaded' if ENV_LOADED else 'NOT FOUND'} ({ENV_FILE}) | HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
print(f"Free disk (cwd): {shutil.disk_usage('.').free / 1e9:.1f} GB")
if shutil.which("nvidia-smi"):
    print("\n⚠️  A GPU runtime is connected. This notebook only needs a CPU. Switch to a CPU runtime to save compute units.")
print("\n✅ Step 1 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 2 — Settings

All settings are in one place.

### Which clips

| Setting | Value | Why |
|---|---|---|
| `N_CLIPS` | 240 / 30 / 30 | Clips for train / val / test. 5 moments per clip gives about 1,500 samples. Try 20 / 5 / 5 first. |
| `MAX_CLIPS_PER_CHUNK` | 6 | Clips stored together are often similar. This limit keeps the data varied. |
| `WINDOW_T0_US` | every 4 s | 5 moments per 20-second clip. 4 seconds apart, so the scenes are different. |

### Model input (do not change)

- 6 cameras, 4 frames each, at −0.3, −0.2, −0.1 and 0 seconds.
- These values come from NVIDIA's own loading code, so the model gets the input it was built for.
- `STORE_MAX_PIXELS` saves images at twice the size the model uses. Files stay small and still sharp.

### Label rules

| Setting | Value | Meaning |
|---|---|---|
| `PATH_HALF_WIDTH_M` | 1.8 m | Half a lane. Objects inside this corridor are "in our path". |
| `PATH_LOOKAHEAD_M` | 25 m | About 2 seconds ahead at city speed. |
| `LEAD_CLOSE_M` | 10 m | A vehicle closer than this in our path is "lead vehicle close". |
| `CUT_IN_MIN_LAT_SPEED_MPS` | 0.6 m/s | Sideways speed that clearly means a lane change. |
| `VRU_NEAR_RADIUS_M` | 6 m | A pedestrian, rider or animal this close is "near ego". |
| `MIN_BOX_HEIGHT_PX` | 18 px | Smaller objects are too tiny to see, so they are not labelled. |
| `MAX_RANGE_M` | 30–70 m | Objects farther away are not labelled. |

### Lighting

- Lighting comes from the height of the sun, using the clip's country, month and local hour.
- Sun above +6° = `daylight`. Below −6° = `night`. In between = `twilight`.
- Image brightness alone is not reliable, because cameras brighten night scenes automatically.

### Counts

- Counts are groups (`1`, `2-3`, `4-7`, `8+`), because exact counts are hard to see in images.

### Versions

- `PROMPT_VERSION`, `RULE_VERSION` and `LIGHTING_VERSION` record which rules made the labels.
''')

code(r'''
# ==== Cell 2: CONFIG ====
# [DEFS]
@dataclass
class Config:
    # ---------- 1. Paths ----------
    # Local defaults can be overridden in .env (DATA_ROOT, LOCAL_WORK, HF_CACHE_DIR)
    DATA_ROOT: str = "/content/drive/MyDrive/drive_annotator/dataset_v1" if IN_COLAB else os.environ.get("DATA_ROOT", str(Path.cwd() / "data" / "dataset_v1"))
    LOCAL_WORK: str = "/content/work" if IN_COLAB else os.environ.get("LOCAL_WORK", str(Path.cwd() / "data" / "_work"))   # fast local scratch
    HF_CACHE_DIR: str = "/content/hf_cache" if IN_COLAB else os.environ.get("HF_CACHE_DIR", str(Path.cwd() / "data" / "_hf_cache"))

    # ---------- 2. Source dataset ----------
    DATASET_REPO: str = "nvidia/PhysicalAI-Autonomous-Vehicles"
    DATASET_REVISION: str | None = None      # None -> use the current `main` version and save it in run_meta.json
    SPLIT_NAME: str = "default"              # NVIDIA ka session-isolated train/val/test split

    # ---------- 3. Clip sampling ----------
    N_CLIPS: dict = field(default_factory=lambda: {"train": 240, "val": 30, "test": 30})
    MAX_CLIPS_PER_CHUNK: int = 6             # clips in one chunk are often similar
    SEED: int = 42

    # ---------- 4. Time windows inside each 20 s clip (clip-relative microseconds) ----------
    WINDOW_T0_US: tuple = (2_000_000, 6_000_000, 10_000_000, 14_000_000, 18_000_000)

    # ---------- 5. Alpamayo2-Edge VQA input profile (DO NOT CHANGE) ----------
    # (camera_index used by the model, short name for labels, devkit feature attribute)
    CAMERAS: tuple = (
        (0, "cross_left",  "CAMERA_CROSS_LEFT_120FOV"),
        (1, "front_wide",  "CAMERA_FRONT_WIDE_120FOV"),
        (2, "cross_right", "CAMERA_CROSS_RIGHT_120FOV"),
        (3, "rear_left",   "CAMERA_REAR_LEFT_70FOV"),
        (5, "rear_right",  "CAMERA_REAR_RIGHT_70FOV"),
        (6, "front_tele",  "CAMERA_FRONT_TELE_30FOV"),
    )
    VQA_OFFSETS_US: tuple = (-300_000, -200_000, -100_000, 0)
    MODEL_MIN_PIXELS: int = 163_840
    MODEL_MAX_PIXELS: int = 196_608

    # ---------- 6. Image storage ----------
    STORE_MAX_PIXELS: int = 393_216          # 2x model max; model processor does the final resize
    JPEG_QUALITY: int = 92

    # ---------- 7. Label engine ----------
    # Dataset's 12 obstacle classes -> our schema classes (None = ignore)
    CLASS_MAP: dict = field(default_factory=lambda: {
        "automobile": "car", "heavy_truck": "truck", "trailer": "truck",
        "bus": "bus", "trolley_bus": "bus", "train_or_tram_car": "tram",
        "other_vehicle": "other_vehicle", "person": "pedestrian", "stroller": "pedestrian",
        "rider": "rider", "animal": "animal", "protruding_object": None,
    })
    MAX_RANGE_M: dict = field(default_factory=lambda: {
        "car": 60.0, "truck": 70.0, "bus": 70.0, "tram": 70.0, "other_vehicle": 50.0,
        "pedestrian": 40.0, "rider": 45.0, "animal": 30.0,
    })
    MIN_BOX_HEIGHT_PX: float = 18.0          # native-resolution pixels
    MIN_DEPTH_M: float = 0.5                 # object centre must be this far in front of the camera
    LABEL_TIME_TOL_US: int = 60_000          # nearest label must be within 60 ms of the frame time
    CUT_IN_LOOKBACK_US: int = 300_000        # = first VQA frame -> model sees the same motion
    PATH_HALF_WIDTH_M: float = 1.8
    PATH_LOOKAHEAD_M: float = 25.0
    LEAD_CLOSE_M: float = 10.0
    CUT_IN_MAX_X_M: float = 30.0
    CUT_IN_MIN_LAT_SPEED_MPS: float = 0.6
    VRU_NEAR_RADIUS_M: float = 6.0           # measured from rig origin (rear axle)
    COUNT_BUCKETS: tuple = ((1, "1"), (3, "2-3"), (7, "4-7"), (10**9, "8+"))
    DIST_BUCKETS: tuple = ((5, "<5m"), (10, "5-10m"), (20, "10-20m"), (40, "20-40m"), (10**9, "40m+"))
    # Lighting = sun elevation from (country, month, local hour). Brightness alone fails: camera
    # auto-exposure brightens night scenes and overcast days look "dim".
    SUN_DAYLIGHT_MIN_DEG: float = 6.0        # sun > +6 deg  -> daylight
    SUN_NIGHT_MAX_DEG: float = -6.0          # sun < -6 deg  -> night (civil twilight ends); between -> twilight
    SUN_DARK_OVERRIDE_MAX_DEG: float = 15.0  # low sun + very dark image (luma <= LUMA_NIGHT_MAX) -> night
    # Fallback only when country/hour metadata is missing:
    LUMA_DAYLIGHT_MIN: float = 0.30          # mean luma of front_wide t0 frame, 0..1
    LUMA_NIGHT_MAX: float = 0.12
    # country -> (latitude, longitude, time-zone meridian in deg); hour_of_day in the dataset is local clock time
    COUNTRY_GEO: dict = field(default_factory=lambda: {
        "Austria": (47.5, 14.5, 15), "Belgium": (50.8, 4.5, 15), "Bulgaria": (42.7, 25.5, 30),
        "Croatia": (45.1, 15.2, 15), "Czechia": (49.8, 15.5, 15), "Denmark": (56.0, 10.0, 15),
        "Estonia": (58.7, 25.0, 30), "Finland": (62.0, 25.7, 30), "France": (46.6, 2.4, 15),
        "Germany": (51.2, 10.4, 15), "Greece": (39.1, 22.0, 30), "Hungary": (47.2, 19.5, 15),
        "Italy": (42.8, 12.6, 15), "Latvia": (56.9, 24.6, 30), "Lithuania": (55.2, 23.9, 30),
        "Luxembourg": (49.8, 6.1, 15), "Netherlands": (52.1, 5.3, 15), "Poland": (52.0, 19.1, 15),
        "Portugal": (39.6, -8.0, 0), "Romania": (45.9, 25.0, 30), "Slovakia": (48.7, 19.7, 15),
        "Slovenia": (46.1, 14.8, 15), "Spain": (40.2, -3.7, 15), "Sweden": (60.1, 15.0, 15),
        "Switzerland": (46.8, 8.2, 15), "Norway": (61.0, 9.0, 15), "United Kingdom": (54.0, -2.0, 0),
        "Ireland": (53.4, -8.0, 0), "United States": (38.0, -95.0, -90),
    })
    LIGHTING_VERSION: str = "light-v3"
    DEDUPE_MIN_EGO_MOVE_M: float = 1.0       # ego barely moved + identical label -> duplicate window

    # ---------- 8. Obstacle-label schema (auto-detected in cell 8; override only if it fails) ----------
    OBSTACLE_FEATURE: str = "auto"           # e.g. "obstacle.offline"
    OBSTACLE_COLUMN_OVERRIDES: dict = field(default_factory=dict)   # e.g. {"track_id": "obj_id"}
    OBSTACLE_FRAME: str = "auto"             # "auto" (use reference_frame column, else detect) | "anchor" | "rig"
    OBSTACLE_SOURCE: str = "auto"            # label source to keep when a clip has several (auto = most common)
    OBSTACLE_CLASS_ID_MAP: dict | None = None   # only if classes are integers: {0: "automobile", ...}

    # ---------- 9. Build ----------
    NUM_WORKERS: int = 3                     # parallel clips; 1 if you see HF 429 errors
    MAX_RETRIES: int = 3
    PROMPT_VERSION: str = "annot-v3"
    RULE_VERSION: str = "rules-v2"
    REVIEW_LIMIT: int = 100
    REVIEWER_ID: str = "reviewer_1"


CFG = Config()

# ---- derived constants (do not edit; derived from CFG) ----
CAMERA_NAMES = [name for _, name, _ in CFG.CAMERAS]
CAMERA_ID = {name: cid for cid, name, _ in CFG.CAMERAS}
SCHEMA_CLASSES = ["car", "truck", "bus", "tram", "other_vehicle", "pedestrian", "rider", "animal"]
VEHICLE_CLASSES = {"car", "truck", "bus", "tram", "other_vehicle"}
VRU_CLASSES = {"pedestrian", "rider", "animal"}
EVENT_TYPES = ["pedestrian_in_path", "vehicle_cut_in", "lead_vehicle_close", "vru_near_ego"]
LIGHTING_VALUES = ["daylight", "twilight", "night"]
COUNT_LABELS = [b for _, b in CFG.COUNT_BUCKETS]
DIST_LABELS = [b for _, b in CFG.DIST_BUCKETS]
TARGET_KEYS = ["lighting", "road_users", "cameras", "events"]

# How Alpamayo2-Edge labels each camera in its prompt (alpamayo2_edge/chat_template.py CAMERA_DISPLAY_NAMES)
CAMERA_DISPLAY = {0: "Front left camera", 1: "Front camera", 2: "Front right camera",
                  3: "Rear left camera", 5: "Rear right camera", 6: "Front telephoto camera"}
PROMPT = (
    "You are a driving-scene annotator. Look only at the six camera views, four frames each covering the last 0.3 s. "
    "Camera names: " + ", ".join(f"{name} = {CAMERA_DISPLAY[cid]}" for cid, name, _ in CFG.CAMERAS) + ". "
    "Return exactly one compact JSON object with keys in this order: lighting, road_users, cameras, events. "
    f"lighting: one of {', '.join(LIGHTING_VALUES)}. "
    f"road_users: object mapping each visible class ({', '.join(SCHEMA_CLASSES)}) "
    f"to a count bucket ({', '.join(COUNT_LABELS)}). "
    "cameras: object mapping a camera name to the list of classes visible in it; omit cameras with none. "
    f"events: list with at most one entry per type ({', '.join(EVENT_TYPES)}); "
    f"each entry has type, camera, distance ({', '.join(DIST_LABELS)}). "
    "Use {} or [] when nothing applies. Output JSON only."
)

# ---- sanity asserts ----
assert set(CFG.N_CLIPS) == {"train", "val", "test"} and all(v >= 1 for v in CFG.N_CLIPS.values())
assert CFG.STORE_MAX_PIXELS >= CFG.MODEL_MAX_PIXELS, "stored frames must not be smaller than model input"
assert all(t0 + CFG.VQA_OFFSETS_US[0] >= 0 for t0 in CFG.WINDOW_T0_US)
assert all(b - a > (CFG.VQA_OFFSETS_US[-1] - CFG.VQA_OFFSETS_US[0]) for a, b in zip(CFG.WINDOW_T0_US, CFG.WINDOW_T0_US[1:])), \
    "windows overlap: spacing must exceed the 0.3 s frame span"
assert CFG.CUT_IN_LOOKBACK_US == -CFG.VQA_OFFSETS_US[0], "cut-in lookback must equal what the model sees"
assert sorted(set(v for v in CFG.CLASS_MAP.values() if v)) == sorted(SCHEMA_CLASSES)
assert set(CFG.MAX_RANGE_M) == set(SCHEMA_CLASSES)
assert CFG.LUMA_NIGHT_MAX < CFG.LUMA_DAYLIGHT_MIN
assert CFG.SUN_NIGHT_MAX_DEG < CFG.SUN_DAYLIGHT_MIN_DEG
assert len(set(CAMERA_NAMES)) == 6

# Fingerprint of everything baked into clips/*.json + frames/*.zip. Cell 3 refuses to mix fingerprints.
# Lighting and the prompt are applied when manifests are assembled (cell 13), so they can change freely.
FINGERPRINT_SCHEMA = 2
LABEL_FINGERPRINT_FIELDS = [
    "DATASET_REPO", "SPLIT_NAME", "WINDOW_T0_US", "CAMERAS", "VQA_OFFSETS_US", "STORE_MAX_PIXELS",
    "JPEG_QUALITY", "CLASS_MAP", "MAX_RANGE_M", "MIN_BOX_HEIGHT_PX", "MIN_DEPTH_M", "LABEL_TIME_TOL_US",
    "CUT_IN_LOOKBACK_US", "PATH_HALF_WIDTH_M", "PATH_LOOKAHEAD_M", "LEAD_CLOSE_M", "CUT_IN_MAX_X_M",
    "CUT_IN_MIN_LAT_SPEED_MPS", "VRU_NEAR_RADIUS_M", "COUNT_BUCKETS", "DIST_BUCKETS",
    "OBSTACLE_SOURCE", "RULE_VERSION",
]


def label_fingerprint(cfg):
    payload = json.dumps({k: getattr(cfg, k) for k in LABEL_FINGERPRINT_FIELDS}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


print(f"Clips: {CFG.N_CLIPS} x {len(CFG.WINDOW_T0_US)} windows -> up to "
      f"{sum(CFG.N_CLIPS.values()) * len(CFG.WINDOW_T0_US)} samples")
print(f"Label fingerprint: {label_fingerprint(CFG)}")
print(f"\nPROMPT ({len(PROMPT)} chars):\n{PROMPT}")
print("\n✅ Step 2 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 3 — Folders and config lock

- In Colab, Google Drive is connected, so nothing is lost if Colab disconnects.
- The cell creates these folders:

```
DATA_ROOT/
  run_meta.json        settings and dataset version
  selection.csv        the chosen clips
  frames/<clip>.zip    the camera images of one clip
  clips/<clip>.json    the labels of one clip (this file means "clip is done")
  manifests/           train / val / test files
  stats/, review/, logs/
```

- It also saves a fingerprint of the settings that change the images or the boxes.
- If you change one of those settings later, this cell stops. This prevents mixing old and new labels.
- To use new settings, pick a new `DATA_ROOT` (for example `dataset_v2`).
- Lighting and the prompt can be changed freely. They are applied later, in Step 13.
''')

code(r'''
# ==== Cell 3: Paths, Drive mount, config lock ====
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

ROOT = Path(CFG.DATA_ROOT)
DIRS = {k: ROOT / k for k in ["frames", "clips", "manifests", "stats", "review", "logs"]}
for d in [ROOT, *DIRS.values(), Path(CFG.LOCAL_WORK), Path(CFG.HF_CACHE_DIR)]:
    d.mkdir(parents=True, exist_ok=True)

def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


RUN_META_PATH = ROOT / "run_meta.json"
fp = label_fingerprint(CFG)
if RUN_META_PATH.exists():
    RUN_META = json.loads(RUN_META_PATH.read_text())
    if RUN_META.get("fingerprint_schema", 1) < FINGERPRINT_SCHEMA and RUN_META["label_fingerprint"] != fp:
        # Older builds hashed prompt/lighting too. Migrate only if every clip-affecting field is unchanged.
        cur = json.loads(json.dumps(asdict(CFG), default=str))
        old = RUN_META.get("config", {})
        changed = [k for k in LABEL_FINGERPRINT_FIELDS if old.get(k) != cur.get(k)]
        if not changed:
            old_fp, n_mig = RUN_META["label_fingerprint"], 0
            for mp in (ROOT / "clips").glob("*.json"):
                m = json.loads(mp.read_text())
                if m.get("label_fingerprint") == old_fp:
                    m["label_fingerprint"] = fp
                    atomic_write_text(mp, json.dumps(m)); n_mig += 1
            RUN_META.update(label_fingerprint=fp, fingerprint_schema=FINGERPRINT_SCHEMA)
            if RUN_META.get("visual_check", {}).get("label_fingerprint") == old_fp:
                RUN_META["visual_check"]["label_fingerprint"] = fp
            RUN_META_PATH.write_text(json.dumps(RUN_META, indent=2))
            print(f"Migrated fingerprint schema -> v{FINGERPRINT_SCHEMA} ({old_fp} -> {fp}, {n_mig} clip files); "
                  "clip content unchanged")
        else:
            print(f"Cannot migrate: clip-affecting fields changed: {changed}")
    if RUN_META["label_fingerprint"] != fp and not any((ROOT / "clips").glob("*.json")):
        print(f"Config changed ({RUN_META['label_fingerprint']} -> {fp}) but no clips built yet -> fingerprint refreshed")
        RUN_META["label_fingerprint"] = fp
        RUN_META["config"] = json.loads(json.dumps(asdict(CFG), default=str))
        RUN_META["prompt"] = PROMPT
        RUN_META_PATH.write_text(json.dumps(RUN_META, indent=2))
    if RUN_META["label_fingerprint"] != fp:
        raise RuntimeError(
            f"These settings change images or labels, and {ROOT} already holds data with a different fingerprint "
            f"({RUN_META['label_fingerprint']} vs {fp}).\n"
            "Fix: use a new DATA_ROOT (for example dataset_v2) or restore the old settings.")
    print(f"Resuming existing dataset at {ROOT}")
else:
    RUN_META = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label_fingerprint": fp,
        "dataset_revision": CFG.DATASET_REVISION,
        "config": json.loads(json.dumps(asdict(CFG), default=str)),
        "prompt": PROMPT,
        "fingerprint_schema": FINGERPRINT_SCHEMA,
    }
    RUN_META_PATH.write_text(json.dumps(RUN_META, indent=2))
    print(f"New dataset at {ROOT}")


def save_run_meta():
    tmp = RUN_META_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(RUN_META, indent=2))
    os.replace(tmp, RUN_META_PATH)


for k, v in DIRS.items():
    print(f"  {k:10s} {v}")
print(f"Free space at DATA_ROOT: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB")
print("\n✅ Step 3 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 4 — Log in to Hugging Face

- The token is read from Colab Secrets, from `.env`, or typed in.
- The cell checks that your account has access to the dataset.
- If access is still pending, it stops with a clear message.
- The current dataset version is saved, so a resumed build always uses the same data.
- The token is never printed.
''')

code(r'''
# ==== Cell 4: HF auth + gated-access check ====
from huggingface_hub import HfApi, login
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError

HF_TOKEN = None
if IN_COLAB:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        HF_TOKEN = None
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN") or getpass.getpass("HF read token: ").strip()
assert HF_TOKEN, "No Hugging Face token found"
assert not HF_TOKEN.startswith("hf_xxx"), "HF_TOKEN in .env is still the placeholder. Put your real token there, restart the kernel and run from Step 1."
if os.environ.get("HF_TOKEN") != HF_TOKEN:      # env var (.env) already authenticates huggingface_hub
    login(token=HF_TOKEN, add_to_git_credential=False)

api = HfApi(token=HF_TOKEN)
print(f"Logged in as: {api.whoami()['name']}")

try:
    api.auth_check(CFG.DATASET_REPO, repo_type="dataset")
except GatedRepoError:
    raise SystemExit(
        "\n❌ The dataset is gated and your access is not approved yet.\n"
        f"   1) https://huggingface.co/datasets/{CFG.DATASET_REPO} kholo\n"
        "   2) While logged in, accept the license agreement\n"
        "   3) After approval, run this cell again.")
except RepositoryNotFoundError:
    raise SystemExit("❌ Repository not found or invalid token. The token needs the 'Read' scope.")

if RUN_META.get("dataset_revision") is None:
    RUN_META["dataset_revision"] = api.dataset_info(CFG.DATASET_REPO).sha
    save_run_meta()
DATASET_REVISION = RUN_META["dataset_revision"]
print(f"Access OK. Dataset revision locked: {DATASET_REVISION}")
print("\n✅ Step 4 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 5 — Load the dataset index

This step downloads only small index files, not videos:

- `features.csv` — which file holds which data,
- `clip_index.parquet` — every clip, its storage chunk and its split,
- `feature_presence` — which sensors each clip has,
- `data_collection` — country, month and hour of each clip.

In the output, look at:

- the name of the 3D box feature (for example `obstacle.offline`),
- the train / val / test split.

NVIDIA keeps all clips from one drive in the same split. So test scenes are never seen during training.
''')

code(r'''
# ==== Cell 5: Dataset interface + metadata ====
assert "DATASET_REVISION" in globals(), (
    "Step 4 did not finish (access pending or the cell was skipped). Run Step 4 again and "
    "continue only after it prints 'Access OK'.")
def make_avdi():
    return physical_ai_av.PhysicalAIAVDatasetInterface(
        revision=DATASET_REVISION, repo_id=CFG.DATASET_REPO, token=HF_TOKEN,
        cache_dir=CFG.HF_CACHE_DIR, confirm_download_threshold_gb=float("inf"),
        split_name=CFG.SPLIT_NAME,
    )

avdi = make_avdi()
FEATURES_DF = avdi.features.features_df

print("Feature directories:", sorted(FEATURES_DF["directory"].unique()))
print("\nlabels/ + calibration/ features:")
print(FEATURES_DF[FEATURES_DF["directory"].isin(["labels", "calibration"])][["directory", "chunk_path"]].to_string())

CLIP_INDEX = avdi.clip_index.copy()
print(f"\nclip_index: {len(CLIP_INDEX):,} clips | columns: {list(CLIP_INDEX.columns)}")
if "split" in CLIP_INDEX.columns:
    print(CLIP_INDEX["split"].value_counts(dropna=False).to_string())

FEATURE_PRESENCE = avdi.feature_presence.copy()
print(f"\nfeature_presence columns ({len(FEATURE_PRESENCE.columns)}): {list(FEATURE_PRESENCE.columns)}")

DATA_COLLECTION = avdi.data_collection.copy()
if "clip_id" in DATA_COLLECTION.columns:
    DATA_COLLECTION = DATA_COLLECTION.set_index("clip_id")
print(f"\ndata_collection columns: {list(DATA_COLLECTION.columns)}")
print(DATA_COLLECTION.head(3).to_string())

# ---- resolve feature names we need ----
CAM_FEATURE = {name: getattr(avdi.features.CAMERA, attr) for _, name, attr in CFG.CAMERAS}
EGO_FEATURE = avdi.features.LABELS.EGOMOTION
EXTR_FEATURE = avdi.features.CALIBRATION.SENSOR_EXTRINSICS
INTR_FEATURE = avdi.features.CALIBRATION.CAMERA_INTRINSICS

if CFG.OBSTACLE_FEATURE == "auto":
    candidates = sorted(f for f in avdi.features.LABELS.ALL if "obstacle" in f.lower())
    assert candidates, f"No obstacle feature found in labels/: {sorted(avdi.features.LABELS.ALL)}"
    offline = [f for f in candidates if "offline" in f.lower()]
    OBS_FEATURE = (offline or candidates)[0]   # offline = optimised, more accurate labels
else:
    OBS_FEATURE = CFG.OBSTACLE_FEATURE
assert OBS_FEATURE in avdi.features.ALL, f"{OBS_FEATURE} not a dataset feature"

print(f"\nCamera features: {CAM_FEATURE}")
print(f"Egomotion: {EGO_FEATURE} | Obstacle: {OBS_FEATURE}")
print("\n✅ Step 5 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 6 — Pick the clips

- Only clips with all 6 cameras, motion data and 3D boxes are used.
- Clips are picked per split, in proportion to each country.
- At most `MAX_CLIPS_PER_CHUNK` clips come from the same storage chunk.
- The choice is fixed by `SEED` and saved to `selection.csv`.
- Running the cell again reuses `selection.csv`. Delete the file to pick again.

The first cell defines the helper functions. The second cell runs them.
''')

code(r'''
# ==== Cell 6: Clip selection ====
# [DEFS]
def proportional_quota(sizes, n):
    """Largest-remainder allocation of n across groups; every non-empty group gets >=1 when n allows."""
    total = sum(sizes.values())
    raw = {k: n * v / total for k, v in sizes.items()}
    quota = {k: int(math.floor(r)) for k, r in raw.items()}
    if n >= len(sizes):
        for k in quota:
            quota[k] = max(quota[k], 1)
    while sum(quota.values()) > n:                       # min-1 may overshoot: trim the biggest
        k = max(quota, key=lambda k: quota[k]); quota[k] -= 1
    for k in sorted(raw, key=lambda k: raw[k] - math.floor(raw[k]), reverse=True):
        if sum(quota.values()) >= n:
            break
        quota[k] += 1
    return {k: min(q, sizes[k]) for k, q in quota.items()}


def sample_clips(pool, n, seed, max_per_chunk, strat_col):
    """pool: DataFrame indexed by clip_id with columns 'chunk' and optional strat_col."""
    pool = pool.sample(frac=1.0, random_state=seed)
    key = pool[strat_col].fillna("unknown") if strat_col else pd.Series("all", index=pool.index)
    groups = {k: list(g.index) for k, g in pool.groupby(key, sort=True)}   # groupby keeps shuffled order
    quota = proportional_quota({k: len(v) for k, v in groups.items()}, min(n, len(pool)))
    chosen, per_chunk = [], Counter()

    def take(cands, limit, cap):
        got = 0
        for c in cands:
            if got >= limit:
                break
            if c in chosen_set or per_chunk[pool.at[c, "chunk"]] >= cap:
                continue
            chosen.append(c); chosen_set.add(c); per_chunk[pool.at[c, "chunk"]] += 1; got += 1
        return got

    chosen_set = set()
    for k, q in quota.items():
        take(groups[k], q, max_per_chunk)
    if len(chosen) < n:                                  # fill from anywhere, still capped
        take(list(pool.index), n - len(chosen), max_per_chunk)
    if len(chosen) < n:                                  # last resort: relax cap
        take(list(pool.index), n - len(chosen), 10**9)
    return chosen
''')

md(r'''
### Run the selection
''')

code(r'''
# ==== Cell 6b: run selection ====
SELECTION_PATH = ROOT / "selection.csv"

if SELECTION_PATH.exists():
    SELECTION = pd.read_csv(SELECTION_PATH)
    print(f"Loaded existing selection ({len(SELECTION)} clips). Delete {SELECTION_PATH.name} to reselect.")
else:
    required = list(CAM_FEATURE.values()) + [EGO_FEATURE, OBS_FEATURE, EXTR_FEATURE, INTR_FEATURE]
    present_cols = [f for f in required if f in FEATURE_PRESENCE.columns]
    missing_cols = sorted(set(required) - set(present_cols))
    if missing_cols:
        print(f"⚠️  Not in the presence table: {missing_cols}. They will be checked during the build instead.")
    ok = FEATURE_PRESENCE[present_cols].all(axis=1) if present_cols else pd.Series(True, index=FEATURE_PRESENCE.index)
    pool = CLIP_INDEX.loc[CLIP_INDEX.index.intersection(ok[ok].index)].copy()

    if "split" not in pool.columns:
        raise RuntimeError("clip_index has no 'split' column. Without NVIDIA's split, test scenes could leak into training. "
                           "Check CFG.SPLIT_NAME (see the names in the splits/ folder).")
    split_alias = {"train": "train", "val": "val", "validation": "val", "valid": "val", "test": "test"}
    pool["split_norm"] = pool["split"].astype(str).str.lower().map(split_alias)

    country_cols = [c for c in DATA_COLLECTION.columns if "country" in c.lower()]
    STRAT_COL = country_cols[0] if country_cols else None
    if STRAT_COL:
        pool[STRAT_COL] = DATA_COLLECTION.reindex(pool.index)[STRAT_COL]
    print(f"Eligible clips: {len(pool):,} | stratify by: {STRAT_COL}")

    rows = []
    for split, n in CFG.N_CLIPS.items():
        sp = pool[pool["split_norm"] == split]
        assert len(sp) >= n, f"{split}: only {len(sp)} usable clips, but {n} are needed"
        for c in sample_clips(sp, n, CFG.SEED, CFG.MAX_CLIPS_PER_CHUNK, STRAT_COL):
            rows.append({"clip_id": c, "split": split, "chunk": int(sp.at[c, "chunk"]),
                         "country": (sp.at[c, STRAT_COL] if STRAT_COL else None)})
    SELECTION = pd.DataFrame(rows)
    SELECTION.to_csv(SELECTION_PATH, index=False)
    print(f"Saved {SELECTION_PATH}")

assert SELECTION["clip_id"].is_unique, "duplicate clip in selection"
assert SELECTION.groupby("clip_id")["split"].nunique().max() == 1
print(SELECTION.groupby("split").agg(clips=("clip_id", "size"), chunks=("chunk", "nunique")).to_string())
print("\nCountry mix:\n", SELECTION.groupby(["split", "country"]).size().unstack(0).fillna(0).astype(int).to_string())
print("\n✅ Step 6 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 7 — Estimate the download size

- Before downloading, we ask Hugging Face for the exact file sizes.
- The dataset stores about 100 clips in one zip file.
- Downloading whole zip files would waste a lot of space.
- So we **stream** only the parts of each zip that we need.
- Only the final small JPEG images are saved to disk.
''')

code(r'''
# ==== Cell 7: Size estimate ====
def chunk_file_sizes(feature, chunks):
    paths = [avdi.features.get_chunk_feature_filename(int(c), feature) for c in chunks]
    sizes = {}
    for i in range(0, len(paths), 500):
        for info in avdi.api.get_paths_info(paths=paths[i:i + 500], **avdi.repo_snapshot_info):
            sizes[info.path] = info.size or 0
    return {int(c): sizes.get(p, 0) for c, p in zip(chunks, paths)}

clips_per_chunk = CLIP_INDEX.groupby("chunk").size()
chunks = sorted(SELECTION["chunk"].unique())
per_clip_bytes = pd.Series(0.0, index=SELECTION.index)
breakdown = {}
for feat in list(CAM_FEATURE.values()) + [EGO_FEATURE, OBS_FEATURE]:
    sizes = chunk_file_sizes(feat, chunks)
    est = SELECTION["chunk"].map(lambda c: sizes[int(c)] / clips_per_chunk.get(c, 100))
    per_clip_bytes += est
    breakdown[feat] = est.sum()

n_windows = len(SELECTION) * len(CFG.WINDOW_T0_US)
jpeg_bytes = n_windows * 24 * 75e3     # ~75 KB per 393k-px JPEG at q92 (measured precisely in cell 11)
print("Estimated streamed data:")
for k, v in breakdown.items():
    print(f"  {k:32s} {v / 1e9:7.2f} GB")
print(f"  {'TOTAL network':32s} {per_clip_bytes.sum() / 1e9:7.2f} GB  (~{per_clip_bytes.mean() / 1e6:.0f} MB/clip)")
print(f"Estimated disk for frames: {jpeg_bytes / 1e9:.1f} GB at DATA_ROOT "
      f"(free: {shutil.disk_usage(ROOT).free / 1e9:.1f} GB)")
assert jpeg_bytes * 1.3 < shutil.disk_usage(ROOT).free, "Not enough free space at DATA_ROOT. Lower N_CLIPS or free up space."
print("\n✅ Step 7 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 8 — Read the 3D box file

The dataset documentation does not list the column names of the box file.

So this step finds them automatically:

1. **Columns** — timestamp, track id, class, center, size and orientation.
2. **Timestamp unit** — seconds, milliseconds or microseconds.
3. **Classes** — the dataset's 12 classes are mapped to our 8 classes.
4. **Coordinate frame** — the file says which frame the boxes use (`reference_frame`).
5. **Label source** — if a clip has two label sources, only one is kept, so nothing is counted twice.

If something cannot be matched, the cell stops and prints the real column names.

Then set `OBSTACLE_COLUMN_OVERRIDES` in Step 2 and run Steps 2 and 8 again.

The first cell defines the helper functions. The second cell runs them on 3 clips.
''')

code(r'''
# ==== Cell 8: Obstacle schema helpers ====
# [DEFS]
ROLE_CANDIDATES = {
    "timestamp":   ["timestamp", "timestamp_us", "time_us", "ts", "time", "t", "label_timestamp"],
    "track_id":    ["track_id", "trackid", "tracking_id", "object_id", "obj_id", "instance_id", "track", "id"],
    "label_class": ["label_class", "class_name", "class", "category", "object_class", "obstacle_class",
                    "label", "type", "class_id", "category_id", "label_id"],
    "x": ["x", "center_x", "cx", "centroid_x", "position_x", "pos_x", "translation_x", "box_center_x", "bbox_center_x"],
    "y": ["y", "center_y", "cy", "centroid_y", "position_y", "pos_y", "translation_y", "box_center_y", "bbox_center_y"],
    "z": ["z", "center_z", "cz", "centroid_z", "position_z", "pos_z", "translation_z", "box_center_z", "bbox_center_z"],
    "length": ["length", "size_x", "dim_x", "dimensions_x", "extent_x", "box_length", "bbox_length", "l", "dx"],
    "width":  ["width", "size_y", "dim_y", "dimensions_y", "extent_y", "box_width", "bbox_width", "w", "dy"],
    "height": ["height", "size_z", "dim_z", "dimensions_z", "extent_z", "box_height", "bbox_height", "h", "dz"],
    "qx": ["qx", "quat_x", "orientation_x", "rotation_x", "quaternion_x", "q_x"],
    "qy": ["qy", "quat_y", "orientation_y", "rotation_y", "quaternion_y", "q_y"],
    "qz": ["qz", "quat_z", "orientation_z", "rotation_z", "quaternion_z", "q_z"],
    "qw": ["qw", "quat_w", "orientation_w", "rotation_w", "quaternion_w", "q_w"],
    "yaw": ["yaw", "heading", "theta", "yaw_rad", "heading_rad"],
    "source": ["source", "label_source", "annotation_source"],
    "reference_frame": ["reference_frame", "coordinate_frame", "frame_id", "frame"],
    "reference_frame_timestamp": ["reference_frame_timestamp_us", "reference_frame_timestamp", "frame_timestamp_us"],
}
FRAME_ALIASES = {"rig": "rig", "ego": "rig", "vehicle": "rig", "anchor": "anchor", "world": "anchor", "local": "anchor"}
REQUIRED_ROLES = ["timestamp", "track_id", "label_class", "x", "y", "z", "length", "width", "height"]


def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def raw_obstacle_to_df(raw):
    """devkit returns DataFrame (parquet features) or {alias: DataFrame|BytesIO} (zip features)."""
    if isinstance(raw, pd.Series):
        raw = raw.to_frame().T
    if isinstance(raw, dict):
        dfs = [v for v in raw.values() if isinstance(v, pd.DataFrame)]
        if not dfs:
            raise ValueError(f"obstacle feature has no parquet member: {list(raw)}")
        raw = max(dfs, key=len)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"unexpected obstacle payload type {type(raw)}")
    df = raw.reset_index()
    return df.drop(columns=[c for c in ["index", "level_0"] if c in df.columns])


def expand_vector_columns(df):
    """Split list/array columns of length 3 or 4 into <col>_x/_y/_z(/_w)."""
    out = df.copy()
    for col in df.columns:
        if df[col].dtype != object or len(df) == 0:
            continue
        first = df[col].dropna().iloc[0] if df[col].notna().any() else None
        if isinstance(first, (list, tuple, np.ndarray)) and len(first) in (3, 4):
            arr = np.stack(df[col].map(lambda v: np.asarray(v, dtype=float)).to_numpy())
            for i, s in enumerate("xyzw"[: arr.shape[1]]):
                out[f"{col}_{s}"] = arr[:, i]
            out = out.drop(columns=[col])
    return out


def resolve_obstacle_columns(df, overrides=None):
    overrides = dict(overrides or {})
    by_norm = {}
    for c in df.columns:
        by_norm.setdefault(_norm(c), c)
    colmap = {}
    for role, cands in ROLE_CANDIDATES.items():
        if role in overrides:
            if overrides[role] not in df.columns:
                raise KeyError(f"override {role}={overrides[role]!r} not in columns {list(df.columns)}")
            colmap[role] = overrides[role]
            continue
        for cand in cands:
            if cand in by_norm and by_norm[cand] not in colmap.values():
                colmap[role] = by_norm[cand]
                break
    missing = [r for r in REQUIRED_ROLES if r not in colmap]
    if missing:
        raise KeyError(
            f"Could not find these box columns: {missing}\nAvailable columns: {list(df.columns)}\n"
            "Fix: set CFG.OBSTACLE_COLUMN_OVERRIDES = {'<role>': '<column>', ...} and run Steps 2 and 8 again.")
    has_quat = all(r in colmap for r in ["qx", "qy", "qz", "qw"])
    if not has_quat:
        for r in ["qx", "qy", "qz", "qw"]:
            colmap.pop(r, None)
    return colmap


def detect_time_scale(raw_ts, ego_range):
    """Find multiplier that converts obstacle timestamps to the egomotion microsecond timeline."""
    # Score = (fraction inside ego range) x (span similarity). Span matters: seconds like 0.5..19
    # also "fit" inside a microsecond range at scale 1, but their span is a million times too small.
    lo, hi = float(ego_range[0]), float(ego_range[1])
    raw_ts = np.asarray(raw_ts, dtype=np.float64)
    best = None
    for scale in (1.0, 1e3, 1e6, 1e-3):
        t = raw_ts * scale
        frac = float(np.mean((t >= lo - 1e5) & (t <= hi + 1e5)))
        span_ratio = (t.max() - t.min()) / max(hi - lo, 1.0)
        score = frac * min(span_ratio, 1.0 / span_ratio) if span_ratio > 0 else 0.0
        if best is None or score > best[1]:
            best = (scale, score, frac)
    if best[2] < 0.8 or best[1] < 0.2:
        raise ValueError(f"Box timestamps [{raw_ts.min()}, {raw_ts.max()}] do not match the motion time range "
                         f"[{lo}, {hi}] us in any unit (best overlap {best[2]:.0%}, score {best[1]:.2f}).")
    return best[0]


def normalize_obstacles(df, colmap, time_scale, class_map, class_id_map=None, source=None):
    """-> DataFrame[t_us, track, cls, source, frame, frame_t_us, x, y, z, length, width, height, qx..qw].
    track = "<source>|<track_id>" (ids are only unique per source). frame/frame_t_us = coordinate frame of
    the box and the time of that frame's pose (None/t_us when the file has no reference_frame column).
    Unmapped classes dropped; if `source` given, other sources dropped."""
    raw_cls = df[colmap["label_class"]]
    if pd.api.types.is_numeric_dtype(raw_cls):
        if not class_id_map:
            raise ValueError(f"Classes are numbers: {sorted(raw_cls.unique().tolist())[:20]}. "
                             "Set CFG.OBSTACLE_CLASS_ID_MAP = {id: 'automobile', ...}.")
        raw_cls = raw_cls.map(class_id_map)
    names = raw_cls.astype(str).map(_norm)
    cls = names.map(lambda n: class_map.get(n, "__unknown__"))
    t_us = np.round(df[colmap["timestamp"]].to_numpy(np.float64) * time_scale).astype(np.int64)
    src = df[colmap["source"]].astype(str).to_numpy() if "source" in colmap else np.full(len(df), "default")
    if "reference_frame" in colmap:
        raw_frame = df[colmap["reference_frame"]].astype(str).map(_norm)
        bad = sorted(set(raw_frame) - set(FRAME_ALIASES))
        if bad:
            raise ValueError(f"Unknown reference_frame values {bad}. Add them to FRAME_ALIASES.")
        frame = raw_frame.map(FRAME_ALIASES).to_numpy(dtype=object)
    else:
        frame = np.full(len(df), None, dtype=object)
    if "reference_frame_timestamp" in colmap:
        frame_t = np.round(df[colmap["reference_frame_timestamp"]].to_numpy(np.float64) * time_scale).astype(np.int64)
    else:
        frame_t = t_us
    out = pd.DataFrame({
        "t_us": t_us,
        "track": np.char.add(np.char.add(src.astype(str), "|"), df[colmap["track_id"]].astype(str).to_numpy().astype(str)),
        "cls": cls.to_numpy(),
        "raw_cls": names.to_numpy(),
        "source": src,
        "frame": frame,
        "frame_t_us": frame_t,
    })
    for r in ["x", "y", "z", "length", "width", "height"]:
        out[r] = df[colmap[r]].to_numpy(np.float64)
    if "qx" in colmap:
        q = df[[colmap["qx"], colmap["qy"], colmap["qz"], colmap["qw"]]].to_numpy(np.float64)
    elif "yaw" in colmap:
        q = Rotation.from_euler("z", df[colmap["yaw"]].to_numpy(np.float64)[:, None]).as_quat()
    else:
        q = np.tile([0.0, 0.0, 0.0, 1.0], (len(df), 1))
    out[["qx", "qy", "qz", "qw"]] = q
    unknown = sorted(set(out.loc[out["cls"] == "__unknown__", "raw_cls"]))
    out = out[(out["cls"] != "__unknown__") & out["cls"].notna()].copy()
    if source is not None:
        out = out[out["source"] == source]
    out = out[np.isfinite(out[["x", "y", "z", "length", "width", "height"]]).all(axis=1)]
    return out.sort_values(["t_us", "track"]).reset_index(drop=True), unknown


def detect_obstacle_frame(obs, ego, min_track_rows=5):
    """Static objects stay put in the anchor (world) frame. Returns (frame|None, metrics)."""
    lo, hi = ego.time_range
    obs = obs[(obs["frame_t_us"] >= lo) & (obs["frame_t_us"] <= hi)]
    if obs.empty:
        return None, {"reason": "no obstacle rows within egomotion range"}
    ts = obs["frame_t_us"].to_numpy(np.int64)
    p = obs[["x", "y", "z"]].to_numpy(np.float64)
    poses = ego(ts).pose
    ego_xy = poses.translation[:, :2]
    travel = float(np.linalg.norm(ego_xy.max(axis=0) - ego_xy.min(axis=0)))
    metrics = {"ego_travel_m": round(travel, 2)}
    if travel < 5.0:
        metrics["reason"] = "ego barely moved; frames indistinguishable"
        return None, metrics
    world = {"anchor": p, "rig": poses.apply(p)}
    for hyp, pts in world.items():
        stds = []
        for _, idx in obs.groupby("track").indices.items():
            if len(idx) >= min_track_rows:
                stds.append(float(np.linalg.norm(pts[idx, :2].std(axis=0))))
        metrics[f"{hyp}_p25_track_std_m"] = round(float(np.percentile(stds, 25)), 3) if stds else None
    a, r = metrics["anchor_p25_track_std_m"], metrics["rig_p25_track_std_m"]
    if a is None or r is None:
        metrics["reason"] = "too few tracks"
        return None, metrics
    lo_v, hi_v = sorted([a, r])
    if hi_v < 1.5 * max(lo_v, 0.05):
        metrics["reason"] = "ambiguous (ratio < 1.5)"
        return None, metrics
    return ("anchor" if a < r else "rig"), metrics
''')

md(r'''
### Run the detection on 3 clips

Check the printed column map by eye.
''')

code(r'''
# ==== Cell 8b: run schema detection on probe clips ====
PROBE_CLIPS, PROBE_RAW = [], {}
for cid in SELECTION[SELECTION["split"] == "train"]["clip_id"]:
    try:
        PROBE_RAW[cid] = expand_vector_columns(raw_obstacle_to_df(avdi.get_clip_feature(cid, OBS_FEATURE, maybe_stream=True)))
    except (KeyError, FileNotFoundError) as exc:
        print(f"  {cid[:8]}: no obstacle labels ({exc}) -> next")
        continue
    if len(PROBE_RAW[cid]):
        PROBE_CLIPS.append(cid)
    if len(PROBE_CLIPS) == 3:
        break
assert PROBE_CLIPS, "No train clip has box labels"

probe_df = PROBE_RAW[PROBE_CLIPS[0]]
print(f"Probe clip {PROBE_CLIPS[0]}: {len(probe_df)} obstacle rows")
print(probe_df.dtypes.to_string())
print(probe_df.head(5).to_string())

OBS_COLMAP = resolve_obstacle_columns(probe_df, CFG.OBSTACLE_COLUMN_OVERRIDES)
print("\nResolved column map (CHECK THIS BY EYE):")
for k, v in OBS_COLMAP.items():
    print(f"  {k:12s} <- {v}")
if not any(k in OBS_COLMAP for k in ("qx", "yaw")):
    print("⚠️  No orientation column found. Boxes are treated as axis-aligned (visibility is slightly less exact).")

ego0 = avdi.get_clip_feature(PROBE_CLIPS[0], EGO_FEATURE, maybe_stream=True)
OBS_TIME_SCALE = detect_time_scale(probe_df[OBS_COLMAP["timestamp"]].to_numpy(), ego0.time_range)
print(f"\nTimestamp scale -> us: x{OBS_TIME_SCALE:g} | egomotion range {ego0.time_range}")

probe_obs, unknown = normalize_obstacles(probe_df, OBS_COLMAP, OBS_TIME_SCALE, CFG.CLASS_MAP, CFG.OBSTACLE_CLASS_ID_MAP)
print(f"Raw classes seen: {sorted(set(probe_df[OBS_COLMAP['label_class']].astype(str)))}")
assert not unknown, f"Classes missing from CLASS_MAP: {unknown}. Add them to CFG.CLASS_MAP (use None to ignore a class)."
per_track_dt = probe_obs.groupby("track")["t_us"].apply(
    lambda t: float(np.median(np.diff(np.sort(t.to_numpy())))) if len(t) > 1 else np.nan)
print(f"Label rate per track: median dt {np.nanmedian(per_track_dt) / 1e3:.0f} ms "
      f"(tracks are asynchronous -> nearest row picked per track) | classes kept: {probe_obs['cls'].value_counts().to_dict()}")
assert np.nanmedian(per_track_dt) <= 2 * CFG.LABEL_TIME_TOL_US, \
    f"Labels come every {np.nanmedian(per_track_dt) / 1e3:.0f} ms, more than 2x LABEL_TIME_TOL_US. Raise the tolerance."

# ---- label source: track ids are per source; keep exactly one source to avoid double counting
probe_all = pd.concat([normalize_obstacles(PROBE_RAW[c], OBS_COLMAP, OBS_TIME_SCALE, CFG.CLASS_MAP,
                                           CFG.OBSTACLE_CLASS_ID_MAP)[0] for c in PROBE_CLIPS])
src_counts = probe_all["source"].value_counts()
print(f"\nLabel sources in probe clips: {src_counts.to_dict()}")
if CFG.OBSTACLE_SOURCE != "auto":
    OBS_SOURCE = CFG.OBSTACLE_SOURCE
else:
    OBS_SOURCE = src_counts.index[0] if len(src_counts) > 1 else None
if OBS_SOURCE:
    print(f"  keeping only source {OBS_SOURCE!r} (set CFG.OBSTACLE_SOURCE to choose)")

# ---- coordinate frame: prefer the file's own reference_frame column
if CFG.OBSTACLE_FRAME != "auto":
    OBSTACLE_FRAME = CFG.OBSTACLE_FRAME
elif "reference_frame" in OBS_COLMAP:
    frames_seen = sorted(set(probe_all["frame"]))
    print(f"reference_frame values (normalised): {frames_seen} "
          f"| pose-time column: {OBS_COLMAP.get('reference_frame_timestamp', OBS_COLMAP['timestamp'])}")
    assert len(frames_seen) == 1, f"mixed frames {frames_seen} - per-row frames not supported yet"
    OBSTACLE_FRAME = frames_seen[0]
else:
    votes = []
    for cid in PROBE_CLIPS:
        obs_c, _ = normalize_obstacles(PROBE_RAW[cid], OBS_COLMAP, OBS_TIME_SCALE, CFG.CLASS_MAP, CFG.OBSTACLE_CLASS_ID_MAP)
        frame, m = detect_obstacle_frame(obs_c, avdi.get_clip_feature(cid, EGO_FEATURE, maybe_stream=True))
        print(f"  frame test {cid[:8]}: {frame} {m}")
        if frame:
            votes.append(frame)
    assert votes, "Could not detect the box frame (car not moving?). Set CFG.OBSTACLE_FRAME to 'anchor' or 'rig'."
    assert len(set(votes)) == 1, f"The probe clips disagree: {votes}. Use the visual check in Step 9 and set CFG.OBSTACLE_FRAME by hand."
    OBSTACLE_FRAME = votes[0]

RUN_META["obstacle_schema"] = {"feature": OBS_FEATURE, "colmap": OBS_COLMAP,
                               "time_scale": OBS_TIME_SCALE, "frame": OBSTACLE_FRAME, "source": OBS_SOURCE}
save_run_meta()
print(f"\nObstacle frame: {OBSTACLE_FRAME}")
print("\n✅ Step 8 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 9 — Place the boxes on the images and check them by eye

Each 3D box is moved step by step:

`box frame → car frame → camera frame → pixel`

The last step uses NVIDIA's fisheye lens model for each camera.

An object counts as **visible** in a camera when:

1. it is in front of the camera and inside the image,
2. it is at least `MIN_BOX_HEIGHT_PX` pixels tall,
3. it is closer than the range for its class.

Known limit: an object hidden behind a truck can still count as visible.

The first cell defines the geometry helpers. The second cell draws the boxes on a real clip.
''')

code(r'''
# ==== Cell 9: Geometry helpers ====
# [DEFS]
_BOX_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)


class ClipGeometry:
    def __init__(self, ego, extrinsics, intrinsics, obstacle_frame, cam_features):
        assert obstacle_frame in ("anchor", "rig")
        self.ego = ego
        self.frame = obstacle_frame
        self.cam_from_rig = {n: extrinsics.sensor_poses[f].inv() for n, f in cam_features.items()}
        self.models = {n: intrinsics.camera_models[f] for n, f in cam_features.items()}

    def rig_from_anchor(self, t_us):
        return self.ego(np.asarray([t_us], dtype=np.int64)).pose[0].inv()

    def to_rig_at(self, pts, pts_t_us, ref_t_us):
        """Points (N,3) in obstacle frame, observed at pts_t_us -> rig frame at ref_t_us."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if self.frame == "anchor":
            anchor = pts
        else:
            ts = np.broadcast_to(np.asarray(pts_t_us, dtype=np.int64), (len(pts),))
            anchor = self.ego(ts).pose.apply(pts)
        return self.rig_from_anchor(ref_t_us).apply(anchor)

    def project(self, cam, pts_rig):
        """-> uv (N,2), depth (N,), in_image (N,) bool."""
        p_cam = self.cam_from_rig[cam].apply(np.asarray(pts_rig, dtype=np.float64).reshape(-1, 3))
        depth = p_cam[:, 2]
        model = self.models[cam]
        uv = np.full((len(p_cam), 2), np.nan)
        front = depth > 1e-3
        if front.any():
            uv[front] = model.ray2pixel(p_cam[front])
        inside = front & ~model.is_out_of_bounds(np.nan_to_num(uv, nan=-1.0))
        return uv, depth, inside


def box_corners(df):
    """(N,8,3) box corners in the obstacle frame."""
    half = df[["length", "width", "height"]].to_numpy(np.float64) / 2.0
    local = _BOX_SIGNS[None, :, :] * half[:, None, :]
    rot = Rotation.from_quat(np.repeat(df[["qx", "qy", "qz", "qw"]].to_numpy(np.float64), 8, axis=0))
    return rot.apply(local.reshape(-1, 3)).reshape(-1, 8, 3) + df[["x", "y", "z"]].to_numpy(np.float64)[:, None, :]


def objects_at(obs, t_us, tol_us):
    """Per track, the label row nearest to t_us (within tol_us). Tracks have their own timestamps.
    None  -> t_us outside the clip's label coverage (treat window as unlabelled).
    empty -> covered, but no object near t_us (a genuinely empty scene)."""
    if obs.empty:
        return None
    t = obs["t_us"].to_numpy(np.int64)
    if t_us < t.min() - tol_us or t_us > t.max() + tol_us:
        return None
    d = np.abs(t - int(t_us))
    near = d <= tol_us
    if not near.any():
        return obs.iloc[0:0].reset_index(drop=True)
    sub = obs[near].assign(_d=d[near])
    idx = sub.groupby("track")["_d"].idxmin().to_numpy()
    return obs.loc[idx].sort_values("track").reset_index(drop=True)


def rig_boxes(geom, rows, t_ref_us):
    """Box centres (N,3) and corners (N,8,3) in rig@t_ref, using each row's frame pose time."""
    n = len(rows)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 8, 3))
    ft = rows["frame_t_us"].to_numpy(np.int64)
    centre = geom.to_rig_at(rows[["x", "y", "z"]].to_numpy(), ft, t_ref_us)
    corners = geom.to_rig_at(box_corners(rows).reshape(-1, 3), np.repeat(ft, 8), t_ref_us).reshape(n, 8, 3)
    return centre, corners


def observe(geom, obs, t_ref_us, cfg, cameras):
    """Objects at t_ref with rig@t_ref positions and per-camera visibility. None if no labels near t_ref."""
    now = objects_at(obs, t_ref_us, cfg.LABEL_TIME_TOL_US)
    if now is None:
        return None
    n = len(now)
    t_lab = now["t_us"].to_numpy(np.int64)
    centre, corners = rig_boxes(geom, now, t_ref_us)
    out = pd.DataFrame({"track": now["track"].to_numpy(), "cls": now["cls"].to_numpy(),
                        "x": centre[:, 0], "y": centre[:, 1], "z": centre[:, 2],
                        "t_label_us": t_lab})
    out["range"] = np.hypot(out["x"], out["y"])
    in_range = out["range"].to_numpy() <= out["cls"].map(cfg.MAX_RANGE_M).to_numpy()
    any_vis = np.zeros(n, dtype=bool)
    for cam in cameras:
        if n == 0:
            out[f"vis_{cam}"] = np.zeros(0, dtype=bool)
            out[f"h_{cam}"] = np.zeros(0)
            continue
        _, depth_c, inside_c = geom.project(cam, centre)
        uv_k, depth_k, _ = geom.project(cam, corners.reshape(-1, 3))
        v = uv_k[:, 1].reshape(n, 8)
        v = np.where(depth_k.reshape(n, 8) > 0.1, v, np.nan)
        h_img = geom.models[cam].height
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all corners behind camera -> NaN row
            height = np.clip(np.nanmax(v, axis=1), 0, h_img) - np.clip(np.nanmin(v, axis=1), 0, h_img)
        height = np.nan_to_num(height, nan=0.0)
        vis = inside_c & (depth_c > cfg.MIN_DEPTH_M) & (height >= cfg.MIN_BOX_HEIGHT_PX) & in_range
        out[f"vis_{cam}"] = vis
        out[f"h_{cam}"] = np.where(vis, height, 0.0)
        any_vis |= vis
    out["visible"] = any_vis
    return out


def previous_positions(geom, obs, t0_us, cfg):
    """Per-track positions CUT_IN_LOOKBACK_US before t0, in rig@t0 (ego motion cancels out).
    -> DataFrame[track, x, y, t_us] (t_us = that row's label time) or None."""
    prev = objects_at(obs, t0_us - cfg.CUT_IN_LOOKBACK_US, cfg.LABEL_TIME_TOL_US)
    if prev is None:
        return None
    centre, _ = rig_boxes(geom, prev, t0_us)
    return pd.DataFrame({"track": prev["track"].to_numpy(), "x": centre[:, 0], "y": centre[:, 1],
                         "t_us": prev["t_us"].to_numpy(np.int64)})


def check_rig_axes(geom, t_lo, t_hi, min_speed_mps=3.0):
    """While moving, 1 s of ego displacement in rig@t must point along +x. Returns #checked windows."""
    checked = 0
    for t in np.arange(t_lo, t_hi - 1_000_000, 1_000_000):
        p0 = geom.ego(np.asarray([t], dtype=np.int64)).pose[0]
        p1 = geom.ego(np.asarray([t + 1_000_000], dtype=np.int64)).pose[0]
        d = p0.inv().apply(p1.translation)
        if np.linalg.norm(d[:2]) < min_speed_mps:
            continue
        assert d[0] > 0 and abs(d[1]) < abs(d[0]), f"rig axes unexpected at t={t}: displacement {d}"
        checked += 1
    return checked
''')

md(r'''
### Visual check

- This cell draws the boxes on a real clip.
- If the boxes sit on the cars and people, set `VISUAL_CHECK_PASSED = True` and run the cell again.
- If they do not, change `OBSTACLE_FRAME` in Step 2 and run Steps 2, 8 and 9 again.
- Step 12 will not run until this check is passed. Your answer is remembered in `run_meta.json`.
''')

code(r'''
# ==== Cell 9b: visual gate on the probe clip ====
VISUAL_CHECK_PASSED = False      # <-- set to True after checking the images, then run again

# Approval is remembered in run_meta.json (same obstacle frame + same label fingerprint) -> no re-edit after restart
_gate = RUN_META.get("visual_check", {})
if (not VISUAL_CHECK_PASSED and _gate.get("passed") and _gate.get("frame") == OBSTACLE_FRAME
        and _gate.get("label_fingerprint") == RUN_META["label_fingerprint"]):
    VISUAL_CHECK_PASSED = True
    print(f"Visual gate already approved on {_gate.get('utc', '?')[:19]} (clip {str(_gate.get('clip'))[:8]}, "
          f"by {_gate.get('approved_by', 'user')}) -> images neeche phir bhi dikhenge")

def load_clip_geometry(av, clip_id, ego, frame=None):
    extr = av.get_clip_feature(clip_id, EXTR_FEATURE, maybe_stream=True)
    intr = av.get_clip_feature(clip_id, INTR_FEATURE, maybe_stream=True)
    return ClipGeometry(ego, extr, intr, frame or OBSTACLE_FRAME, CAM_FEATURE)

def load_obstacles(av, clip_id):
    """Normalised obstacles + the clip's coordinate frame (from the file when it says so)."""
    df = expand_vector_columns(raw_obstacle_to_df(av.get_clip_feature(clip_id, OBS_FEATURE, maybe_stream=True)))
    obs, unknown = normalize_obstacles(df, OBS_COLMAP, OBS_TIME_SCALE, CFG.CLASS_MAP,
                                       CFG.OBSTACLE_CLASS_ID_MAP, source=OBS_SOURCE)
    if unknown:
        raise ValueError(f"unknown classes {unknown}")
    frames = set(obs["frame"].dropna()) if CFG.OBSTACLE_FRAME == "auto" else set()
    if len(frames) > 1:
        raise ValueError(f"mixed reference frames in clip: {frames}")
    return obs, (frames.pop() if frames else OBSTACLE_FRAME)

vis_clip = PROBE_CLIPS[0]
obs0, frame0 = load_obstacles(avdi, vis_clip)
geom0 = load_clip_geometry(avdi, vis_clip, ego0, frame0)
n_axes = check_rig_axes(geom0, *ego0.time_range)
print(f"Rig-axis check passed on {n_axes} moving 1-s windows" + ("" if n_axes else " (ego static; check skipped)"))
t_vis = CFG.WINDOW_T0_US[1]
seen = observe(geom0, obs0, t_vis, CFG, CAMERA_NAMES)
assert seen is not None, "No labels at the probe moment. Check WINDOW_T0_US."
print(f"t0={t_vis/1e6:.1f}s: {len(seen)} labelled objects, {int(seen['visible'].sum())} visible")

fig, axes = plt.subplots(2, 3, figsize=(18, 7))
_, corners = rig_boxes(geom0, objects_at(obs0, t_vis, CFG.LABEL_TIME_TOL_US), t_vis)
for ax, cam in zip(axes.ravel(), CAMERA_NAMES):
    reader = avdi.get_clip_feature(vis_clip, CAM_FEATURE[cam], maybe_stream=True)
    img, _ = reader.decode_images_from_timestamps(np.array([t_vis], dtype=np.int64))
    ax.imshow(img[0]); ax.set_title(cam); ax.axis("off")
    for i in np.where(seen[f"vis_{cam}"])[0]:
        uv, depth, _ = geom0.project(cam, corners[i])
        uv = uv[depth > 0.1]
        if len(uv) < 2:
            continue
        (u0, v0), (u1, v1) = np.nanmin(uv, axis=0), np.nanmax(uv, axis=0)
        ax.add_patch(plt.Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, lw=1.5,
                                   color="lime" if seen.at[i, "cls"] in VEHICLE_CLASSES else "red"))
        ax.text(u0, v0 - 4, seen.at[i, "cls"], color="yellow", fontsize=7)
    if hasattr(reader, "close"):
        reader.close()
plt.suptitle(f"Visual check: boxes should sit on the objects (frame={OBSTACLE_FRAME}). Green = vehicle, red = pedestrian/rider/animal")
plt.tight_layout(); plt.show()

if not VISUAL_CHECK_PASSED:
    print("\n🚦 Check the images. If the boxes are right, set VISUAL_CHECK_PASSED = True and run this cell again.\n"
          "   If they are wrong, change OBSTACLE_FRAME in Step 2 ('anchor' / 'rig') and run Steps 2, 8 and 9 again.")
else:
    if not (_gate.get("passed") and _gate.get("label_fingerprint") == RUN_META["label_fingerprint"]):
        RUN_META["visual_check"] = {"passed": True, "clip": vis_clip, "frame": OBSTACLE_FRAME,
                                    "label_fingerprint": RUN_META["label_fingerprint"], "approved_by": "user",
                                    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        save_run_meta()
    print("\n✅ Step 9 OK (visual gate passed)")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 10 — Label rules

This step turns the visible boxes into one label:

```
lighting: daylight
road_users: {"car": "4-7", "pedestrian": "1"}
cameras: {"front_wide": ["car", "pedestrian"], "cross_left": ["car"]}
events: [{"type": "pedestrian_in_path", "camera": "front_wide", "distance": "10-20m"}]
```

### Event rules

| Event | Rule |
|---|---|
| `pedestrian_in_path` | A pedestrian, rider or animal inside our corridor, up to 25 m ahead |
| `vehicle_cut_in` | A vehicle up to 30 m ahead that moves sideways into our corridor at 0.6 m/s or more |
| `lead_vehicle_close` | A vehicle in our corridor, closer than 10 m |
| `vru_near_ego` | A pedestrian, rider or animal within 6 m of our car |

### Details

- Motion is measured only over the last 0.3 seconds, because that is all the model sees.
- The corridor bends with the car's current steering, so turns are handled.
- For each event type, only the nearest object is used.
- The event's camera is the camera where the object looks biggest.
- `validate_target()` checks that every label has the right format.

The tests for these rules are in `dev/test_notebook.py`.
''')

code(r'''
# ==== Cell 10: Label engine ====
# [DEFS]
def bucket(value, buckets):
    for upper, label in buckets:
        if value <= upper:
            return label
    return buckets[-1][1]


def sun_elevation_deg(lat, lon, tz_meridian, month, hour_local):
    """Approximate solar elevation (deg) for mid-month, middle of the given local clock hour.
    DST assumed Apr-Oct. Error is roughly +-1 h of sun motion, absorbed by the twilight band."""
    doy = int(30.44 * (month - 1) + 15)
    decl = math.radians(-23.44 * math.cos(2 * math.pi * (doy + 10) / 365.0))
    dst = 1.0 if 4 <= month <= 10 else 0.0
    solar_hour = hour_local + 0.5 - dst + (lon - tz_meridian) / 15.0
    ha = math.radians(15.0 * (solar_hour - 12.0))
    phi = math.radians(lat)
    sin_el = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))


def clip_sun_elevation(country, month, hour, cfg):
    geo = cfg.COUNTRY_GEO.get(country) if isinstance(country, str) else None
    if geo is None or month is None or hour is None or pd.isna(month) or pd.isna(hour):
        return None
    return sun_elevation_deg(geo[0], geo[1], geo[2], int(month), int(hour))


def scene_lighting(sun_elev, luma, cfg):
    """Sun elevation when known; image brightness only as fallback."""
    if sun_elev is None:
        if luma >= cfg.LUMA_DAYLIGHT_MIN:
            return "daylight"
        return "night" if luma <= cfg.LUMA_NIGHT_MAX else "twilight"
    if luma <= cfg.LUMA_NIGHT_MAX and sun_elev < cfg.SUN_DARK_OVERRIDE_MAX_DEG:
        return "night"            # sun estimate is +-1 h; a pitch-dark frame near sunrise/sunset is night
    if sun_elev >= cfg.SUN_DAYLIGHT_MIN_DEG:
        return "daylight"
    return "night" if sun_elev <= cfg.SUN_NIGHT_MAX_DEG else "twilight"


def image_luma(img):
    return float(np.asarray(img.convert("L"), dtype=np.float32).mean() / 255.0)


def path_offset(x, y, curvature):
    return np.asarray(y, dtype=np.float64) - 0.5 * curvature * np.asarray(x, dtype=np.float64) ** 2


def _best_camera(row, cameras):
    best, best_h = None, -1.0
    for cam in cameras:
        if row[f"vis_{cam}"] and row[f"h_{cam}"] > best_h:
            best, best_h = cam, row[f"h_{cam}"]
    return best


def compute_events(now, prev, curvature, cfg, cameras):
    """now: observe() output; prev: previous_positions() output (both rig@t0). prev may be None.
    Lateral speed uses each track's own time gap (labels are asynchronous per track)."""
    events = []
    vis = now[now["visible"]]
    if vis.empty:
        return events

    def add(etype, rows):
        if len(rows):
            r = rows.loc[rows["range"].idxmin()]
            events.append({"type": etype, "camera": _best_camera(r, cameras),
                           "distance": bucket(float(r["range"]), cfg.DIST_BUCKETS)})

    off = np.abs(path_offset(vis["x"], vis["y"], curvature))
    in_path = (vis["x"] > 0) & (vis["x"] <= cfg.PATH_LOOKAHEAD_M) & (off <= cfg.PATH_HALF_WIDTH_M)
    is_vru, is_veh = vis["cls"].isin(VRU_CLASSES), vis["cls"].isin(VEHICLE_CLASSES)

    add("pedestrian_in_path", vis[in_path & is_vru])

    if prev is not None and len(prev):
        p_ = prev[["track", "x", "y", "t_us"]].rename(columns={"x": "x_prev", "y": "y_prev", "t_us": "t_prev_us"})
        m = vis[is_veh].drop(columns=["x_prev", "y_prev", "t_prev_us"], errors="ignore").merge(p_, on="track")
        if len(m):
            dt = (m["t_label_us"].to_numpy(np.int64) - m["t_prev_us"].to_numpy(np.int64)) / 1e6
            off_now = np.abs(path_offset(m["x"], m["y"], curvature))
            off_prev = np.abs(path_offset(m["x_prev"], m["y_prev"], curvature))
            lat_speed = np.where(dt > 0.1, (off_prev - off_now) / np.maximum(dt, 1e-6), 0.0)
            cond = ((m["x"] > 0) & (m["x"] <= cfg.CUT_IN_MAX_X_M)
                    & (off_prev >= cfg.PATH_HALF_WIDTH_M + 0.3)
                    & (off_now <= cfg.PATH_HALF_WIDTH_M + 1.0)
                    & (lat_speed >= cfg.CUT_IN_MIN_LAT_SPEED_MPS))
            add("vehicle_cut_in", m[cond.to_numpy()])

    add("lead_vehicle_close", vis[in_path & is_veh & (vis["x"] <= cfg.LEAD_CLOSE_M)])
    add("vru_near_ego", vis[is_vru & (vis["range"] <= cfg.VRU_NEAR_RADIUS_M)])
    return events


def build_target(now, prev, curvature, lighting, cfg, cameras):
    vis = now[now["visible"]] if now is not None else pd.DataFrame(columns=["cls", "track"])
    counts = vis.groupby("cls")["track"].nunique().to_dict() if len(vis) else {}
    road_users = {c: bucket(counts[c], cfg.COUNT_BUCKETS) for c in SCHEMA_CLASSES if counts.get(c, 0) > 0}
    cams = {}
    for cam in cameras:
        if len(vis):
            present = set(vis.loc[vis[f"vis_{cam}"], "cls"])
            classes = [c for c in SCHEMA_CLASSES if c in present]
            if classes:
                cams[cam] = classes
    events = compute_events(now, prev, curvature, cfg, cameras) if now is not None else []
    return {"lighting": lighting, "road_users": road_users, "cameras": cams, "events": events}


def target_to_text(target):
    return json.dumps(target, separators=(",", ":"), ensure_ascii=False)


def validate_target(obj):
    """Returns list of schema errors (empty = valid). Also used later on model outputs."""
    errs = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    if list(obj.keys()) != TARGET_KEYS:
        errs.append(f"keys {list(obj.keys())} != {TARGET_KEYS}")
    if obj.get("lighting") not in LIGHTING_VALUES:
        errs.append(f"lighting {obj.get('lighting')!r}")
    ru = obj.get("road_users")
    if not isinstance(ru, dict):
        errs.append("road_users not object"); ru = {}
    for k, v in ru.items():
        if k not in SCHEMA_CLASSES: errs.append(f"road_users class {k!r}")
        if v not in COUNT_LABELS: errs.append(f"road_users bucket {v!r}")
    cams = obj.get("cameras")
    if not isinstance(cams, dict):
        errs.append("cameras not object"); cams = {}
    for cam, classes in cams.items():
        if cam not in CAMERA_NAMES: errs.append(f"camera {cam!r}")
        if not isinstance(classes, list) or not classes: errs.append(f"cameras[{cam}] empty/not list"); continue
        if len(set(classes)) != len(classes): errs.append(f"cameras[{cam}] duplicates")
        for c in classes:
            if c not in ru: errs.append(f"cameras[{cam}] has {c!r} missing from road_users")
    ev = obj.get("events")
    if not isinstance(ev, list):
        errs.append("events not list"); ev = []
    seen_types = set()
    for e in ev:
        if not isinstance(e, dict) or list(e.keys()) != ["type", "camera", "distance"]:
            errs.append(f"event shape {e!r}"); continue
        if e["type"] not in EVENT_TYPES: errs.append(f"event type {e['type']!r}")
        if e["type"] in seen_types: errs.append(f"duplicate event {e['type']}")
        seen_types.add(e["type"])
        if e["camera"] not in CAMERA_NAMES: errs.append(f"event camera {e['camera']!r}")
        if e["distance"] not in DIST_LABELS: errs.append(f"event distance {e['distance']!r}")
        need = VRU_CLASSES if e["type"] in ("pedestrian_in_path", "vru_near_ego") else VEHICLE_CLASSES
        if not need & set(ru): errs.append(f"event {e['type']} without matching road_user")
    return errs


def parse_and_validate(text):
    try:
        obj = json.loads(text)
    except Exception as exc:
        return None, [f"invalid JSON: {exc}"]
    return obj, validate_target(obj)
''')

pass  # (label-engine tests moved to dev/dataset_label_tests.py)

# ---------------------------------------------------------------------------
md(r'''
## Step 11 — Try one clip first

`process_clip()` does the whole job for one clip:

1. Streams the motion data, the calibration and the 3D boxes.
2. Picks the moments that have 0.3 s of history and labels.
3. Streams each camera video once and decodes all needed frames in one pass.
4. Resizes each frame and saves it as JPEG right away, so memory stays low.
5. Builds the labels.
6. Writes the zip file first and the clip's `.json` file last. The `.json` file means "this clip is done".

The second cell runs it on one clip and shows the 24 frames, the label, the time and the file size.

From that time, it estimates how long the full build will take.
''')

code(r'''
# ==== Cell 11: process_clip ====
_tls = threading.local()

def thread_avdi():
    if not hasattr(_tls, "avdi"):
        _tls.avdi = avdi if threading.current_thread() is threading.main_thread() else make_avdi()
    return _tls.avdi


def encode_jpeg(frame, max_pixels, quality):
    img = Image.fromarray(frame)
    h, w = frame.shape[:2]
    if h * w > max_pixels:
        s = math.sqrt(max_pixels / (h * w))
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), img


LOCK_STALE_S = 1800      # a lock older than 30 min = crashed run -> take over


def process_clip(row):
    """Resume + cross-process safe: done clips are skipped; a clip another process is building
    (fresh clips/<id>.lock) returns status 'busy' and is retried on the next run."""
    clip_id = row["clip_id"]
    meta_path = DIRS["clips"] / f"{clip_id}.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    lock = DIRS["clips"] / f"{clip_id}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()} utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}".encode())
        os.close(fd)
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
        except FileNotFoundError:
            return process_clip(row)                     # lock just released -> retry once
        if age < LOCK_STALE_S:
            return {"clip_id": clip_id, "status": "busy", "samples": []}
        lock.unlink(missing_ok=True)
        return process_clip(row)
    try:
        if meta_path.exists():                           # finished by another process meanwhile
            return json.loads(meta_path.read_text())
        return _process_clip_locked(row)
    finally:
        lock.unlink(missing_ok=True)


def _process_clip_locked(row):
    clip_id, split = row["clip_id"], row["split"]
    meta_path = DIRS["clips"] / f"{clip_id}.json"
    av = thread_avdi()
    t_start = time.time()
    base = {"clip_id": clip_id, "split": split, "chunk": int(row["chunk"]),
            "country": None if pd.isna(row.get("country")) else row.get("country"),
            "dataset_revision": DATASET_REVISION, "label_fingerprint": RUN_META["label_fingerprint"]}

    def finish(status, reason=None, samples=(), extra=None):
        meta = base | {"status": status, "reason": reason, "samples": list(samples),
                       "elapsed_s": round(time.time() - t_start, 1)} | (extra or {})
        atomic_write_text(meta_path, json.dumps(meta))
        return meta

    ego = av.get_clip_feature(clip_id, EGO_FEATURE, maybe_stream=True)
    try:
        obs, clip_frame = load_obstacles(av, clip_id)
    except (KeyError, FileNotFoundError) as exc:
        return finish("skipped", f"no obstacle labels: {exc}")
    geom = load_clip_geometry(av, clip_id, ego, clip_frame)
    if obs.empty:
        return finish("skipped", "obstacle table empty after class mapping")

    offs = np.asarray(CFG.VQA_OFFSETS_US, dtype=np.int64)
    lo, hi = ego.time_range
    windows = []
    for t0 in CFG.WINDOW_T0_US:
        if t0 + offs[0] < lo or t0 > hi:
            continue
        if objects_at(obs, t0, CFG.LABEL_TIME_TOL_US) is None:
            continue
        if objects_at(obs, t0 - CFG.CUT_IN_LOOKBACK_US, CFG.LABEL_TIME_TOL_US) is None:
            continue
        windows.append(t0)
    if not windows:
        return finish("skipped", "no window with ego + labels coverage")

    jpegs, frame_meta, luma, native = {}, defaultdict(list), {}, {}
    for cam in CAMERA_NAMES:
        reader = av.get_clip_feature(clip_id, CAM_FEATURE[cam], maybe_stream=True)
        try:
            avail = np.asarray(reader.timestamps)
            windows = [t0 for t0 in windows if t0 + offs[0] >= avail[0] and t0 <= avail[-1]]
            if not windows:
                return finish("skipped", f"{cam}: no window inside video time range")
            req = np.concatenate([t0 + offs for t0 in windows])
            frames, ts = reader.decode_images_from_timestamps(req)
        finally:
            if hasattr(reader, "close"):
                reader.close()
        frames, ts = np.asarray(frames), np.asarray(ts, dtype=np.int64)
        assert frames.ndim == 4 and frames.shape[0] == len(req) and frames.shape[-1] == 3, f"{cam} shape {frames.shape}"
        assert frames.dtype == np.uint8, f"{cam} dtype {frames.dtype}"
        assert ts.shape == req.shape and np.all(ts <= req) and np.all(np.diff(ts) >= 0), f"{cam} bad timestamps"
        model = geom.models[cam]
        assert (frames.shape[2], frames.shape[1]) == (model.width, model.height), \
            f"{cam}: decoded {frames.shape[2]}x{frames.shape[1]} != calibration {model.width}x{model.height}"
        native[cam] = [int(model.width), int(model.height)]
        for i, (frame, t_act) in enumerate(zip(frames, ts)):
            w_i, k = divmod(i, len(offs))
            t0 = windows[w_i] if w_i < len(windows) else None
            data, img = encode_jpeg(frame, CFG.STORE_MAX_PIXELS, CFG.JPEG_QUALITY)
            path = f"{t0}/{cam}_{k}.jpg"
            jpegs[path] = data
            frame_meta[t0].append({"camera_id": CAMERA_ID[cam], "camera": cam, "timestamp_us": int(t_act),
                                   "path": path})
            if cam == "front_wide" and k == len(offs) - 1:
                luma[t0] = image_luma(img)
                stored_size = list(img.size)
        del frames
    # a later camera may have dropped windows; keep only windows every camera covered
    windows = [t0 for t0 in windows if len(frame_meta[t0]) == 6 * len(offs)]

    dc = DATA_COLLECTION.loc[clip_id] if clip_id in DATA_COLLECTION.index else {}
    sun_elev = clip_sun_elevation(dc.get("country"), dc.get("month"), dc.get("hour_of_day"), CFG)
    samples = []
    for t0 in windows:
        now = observe(geom, obs, t0, CFG, CAMERA_NAMES)
        prev = previous_positions(geom, obs, t0, CFG)
        curvature = float(np.asarray(ego(np.asarray([t0], dtype=np.int64)).curvature).ravel()[0])
        target = build_target(now, prev, curvature, scene_lighting(sun_elev, luma[t0], CFG), CFG, CAMERA_NAMES)
        errs = validate_target(target)
        assert not errs, f"label engine produced invalid target: {errs}"
        ego_xy = ego(np.asarray([t0], dtype=np.int64)).pose[0].translation[:2].tolist()
        samples.append({
            "sample_id": f"{clip_id}__{t0}", "clip_id": clip_id, "t0_us": int(t0), "split": split,
            "frames_zip": f"frames/{clip_id}.zip",
            "frames": sorted(frame_meta[t0], key=lambda f: (f["camera_id"], f["timestamp_us"])),
            "target": target, "target_text": target_to_text(target),
            "label_meta": {"luma": round(luma[t0], 4), "n_labelled": int(len(now)),
                           "n_visible": int(now["visible"].sum()), "curvature": round(curvature, 5),
                           "ego_xy": ego_xy, "obstacle_frame": clip_frame},
        })

    keep = {p: d for p, d in jpegs.items() if int(p.split("/")[0]) in set(windows)}
    tag = f"{os.getpid()}_{threading.get_ident()}"      # unique per writer -> no clashes
    tmp_zip = Path(CFG.LOCAL_WORK) / f"{clip_id}.{tag}.zip"
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for p, d in keep.items():
            zf.writestr(p, d)
    dst = DIRS["frames"] / f"{clip_id}.zip"
    dst_tmp = dst.with_name(f"{dst.name}.{tag}.tmp")
    try:
        shutil.copyfile(tmp_zip, dst_tmp)
        os.replace(dst_tmp, dst)
    finally:
        dst_tmp.unlink(missing_ok=True)
        tmp_zip.unlink(missing_ok=True)
    return finish("ok", samples=samples, extra={"native_sizes": native, "stored_size_front_wide": stored_size,
                                               "zip_bytes": dst.stat().st_size})


def read_frame(sample, frame):
    with zipfile.ZipFile(ROOT / sample["frames_zip"]) as zf:
        return Image.open(io.BytesIO(zf.read(frame["path"]))).convert("RGB")
''')

md(r'''
### Run one clip
''')

code(r'''
# ==== Cell 11b: run on ONE clip ====
assert VISUAL_CHECK_PASSED, "Pass the visual check in Step 9 first"
meta = None
for test_row in SELECTION.head(10).to_dict("records"):
    t = time.time()
    meta = process_clip(test_row)
    elapsed = time.time() - t
    print(f"{test_row['clip_id'][:8]} status={meta['status']} reason={meta.get('reason')} "
          f"samples={len(meta['samples'])} time={elapsed:.1f}s zip={meta.get('zip_bytes', 0) / 1e6:.1f} MB")
    if meta["status"] == "ok":
        break
assert meta and meta["status"] == "ok", "The first 10 clips were all skipped. Read the skip reasons and check the settings."

s = meta["samples"][0]
assert len(s["frames"]) == 24
print(json.dumps(s["target"], indent=1))
fig, axes = plt.subplots(6, 4, figsize=(14, 13))
for ax, fr in zip(axes.ravel(), s["frames"]):
    ax.imshow(read_frame(s, fr)); ax.axis("off")
    ax.set_title(f"{fr['camera']} t={(fr['timestamp_us'] - s['t0_us']) / 1e3:+.0f}ms", fontsize=8)
plt.suptitle(f"{s['sample_id']}  (rows=cameras, cols=-0.3s..0s)"); plt.tight_layout(); plt.show()

per_clip = elapsed if elapsed > 1 else 10.0
n_left = len(SELECTION) - len(list(DIRS["clips"].glob("*.json")))
eta_h = n_left * per_clip / max(1, CFG.NUM_WORKERS) / 3600
print(f"\nEstimate for remaining {n_left} clips with {CFG.NUM_WORKERS} workers: ~{eta_h:.1f} h "
      f"| frames disk ~{meta.get('zip_bytes', 0) * len(SELECTION) / 1e9:.1f} GB")
print("\n✅ Step 11 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 12 — Build all clips

- All selected clips are processed, `NUM_WORKERS` at a time.
- **Resume:** finished clips are skipped. After a disconnect, run Steps 1–9 and the first cell of Step 11, then this cell.
- Network errors are retried. If Hugging Face answers `429 Too Many Requests`, set `NUM_WORKERS = 1`.
- Failed clips are written to `logs/failures.jsonl` and retried the next time you run this cell.
- A lock file makes sure two notebooks never build the same clip at the same time.

Tip: try a small `N_CLIPS` first, check Steps 13 and 14, then increase it.
''')

code(r'''
# ==== Cell 12: Full build ====
assert VISUAL_CHECK_PASSED, "The visual check in Step 9 has not been passed"
FAIL_LOG = DIRS["logs"] / "failures.jsonl"

def process_with_retry(row):
    for attempt in range(1, CFG.MAX_RETRIES + 1):
        try:
            return process_clip(row)
        except AssertionError:
            raise                                   # data/logic bug: retry won't help
        except Exception as exc:
            if attempt == CFG.MAX_RETRIES:
                raise
            wait = 60 if "429" in str(exc) else 5 * 2 ** attempt
            time.sleep(wait)

def _old(p, minutes=30):
    try:
        return time.time() - p.stat().st_mtime > minutes * 60
    except FileNotFoundError:
        return False

stale = [p for p in DIRS["frames"].glob("*.tmp") if _old(p)] + [p for p in Path(CFG.LOCAL_WORK).glob("*.zip") if _old(p)]
for p in stale:
    p.unlink(missing_ok=True)
if stale:
    print(f"Removed {len(stale)} stale temp files from an interrupted run")

todo = [r for r in SELECTION.to_dict("records") if not (DIRS["clips"] / f"{r['clip_id']}.json").exists()]
print(f"{len(SELECTION) - len(todo)} done, {len(todo)} to process")
status = Counter()
t_build = time.time()
with ThreadPoolExecutor(max_workers=CFG.NUM_WORKERS) as pool:
    futures = {pool.submit(process_with_retry, r): r for r in todo}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="clips"):
        r = futures[fut]
        try:
            status[fut.result()["status"]] += 1
        except Exception as exc:
            status["failed"] += 1
            with open(FAIL_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"clip_id": r["clip_id"], "error": repr(exc),
                                    "traceback": traceback.format_exc()[-3000:]}) + "\n")
print(f"\nThis run: {dict(status)} in {(time.time() - t_build) / 60:.1f} min")
done = len(list(DIRS["clips"].glob("*.json")))
print(f"Total clips done: {done}/{len(SELECTION)}")
if status["busy"]:
    print(f"ℹ️  {status['busy']} clips were being built by another process. Run this cell again when it finishes.")
if status["failed"]:
    print(f"⚠️  {status['failed']} clips failed. See {FAIL_LOG}. Running this cell again retries them.")
print("\n✅ Step 12 finished")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 13 — Write the train / val / test files

Each split gets one `.jsonl` file. Each line is one sample:

```
{"sample_id": "...", "clip_id": "...", "t0_us": 6000000, "split": "train",
 "frames_zip": "frames/<clip>.zip",
 "frames": [24 entries with camera, timestamp and file path],
 "target": {...}, "target_text": "...", "label_meta": {...}}
```

This step also:

- computes the lighting label (so lighting rules can change without rebuilding clips),
- removes duplicates: if the car moved less than 1 m and the label is the same, the second moment is dropped,
- checks every sample: 24 frames, all files exist, the label is valid, and no clip appears in two splits.
''')

code(r'''
# ==== Cell 13: Assemble manifests ====
metas = [json.loads(p.read_text()) for p in sorted(DIRS["clips"].glob("*.json"))]
assert all(m["label_fingerprint"] == RUN_META["label_fingerprint"] for m in metas), "mixed fingerprints!"
sel_split = dict(zip(SELECTION["clip_id"], SELECTION["split"]))
metas = [m for m in metas if m["clip_id"] in sel_split]
print(Counter(m["status"] for m in metas))
print("Skip reasons:", Counter((m.get("reason") or "")[:40] for m in metas if m["status"] == "skipped").most_common(5))

records, n_dupes, light_change = defaultdict(list), 0, Counter()
for m in metas:
    if m["status"] != "ok":
        continue
    dc = DATA_COLLECTION.loc[m["clip_id"]] if m["clip_id"] in DATA_COLLECTION.index else {}
    sun_elev = clip_sun_elevation(dc.get("country"), dc.get("month"), dc.get("hour_of_day"), CFG)
    for s in m["samples"]:                      # lighting is assembly-time: recompute with current rule
        new_light = scene_lighting(sun_elev, s["label_meta"]["luma"], CFG)
        light_change[(s["target"]["lighting"], new_light)] += 1
        s["target"] = {**s["target"], "lighting": new_light}
        s["target_text"] = target_to_text(s["target"])
        s["label_meta"] = {**s["label_meta"], "sun_elev_deg": None if sun_elev is None else round(sun_elev, 1),
                           "lighting_source": "sun" if sun_elev is not None else "luma",
                           "lighting_version": CFG.LIGHTING_VERSION}
    zpath = ROOT / f"frames/{m['clip_id']}.zip"
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
    last = None
    for s in sorted(m["samples"], key=lambda s: s["t0_us"]):
        if last is not None:
            moved = float(np.linalg.norm(np.subtract(s["label_meta"]["ego_xy"], last["label_meta"]["ego_xy"])))
            if moved < CFG.DEDUPE_MIN_EGO_MOVE_M and s["target_text"] == last["target_text"]:
                n_dupes += 1
                continue
        assert s["split"] == sel_split[s["clip_id"]]
        assert len(s["frames"]) == 24 and all(f["path"] in names for f in s["frames"]), s["sample_id"]
        assert Counter(f["camera"] for f in s["frames"]) == Counter({c: 4 for c in CAMERA_NAMES})
        assert validate_target(s["target"]) == [] and target_to_text(s["target"]) == s["target_text"]
        records[s["split"]].append(s | {"prompt_version": CFG.PROMPT_VERSION, "prompt": PROMPT})
        last = s

clip_sets = {sp: {r["clip_id"] for r in recs} for sp, recs in records.items()}
for a in clip_sets:
    for b in clip_sets:
        if a < b:
            assert not clip_sets[a] & clip_sets[b], f"clip leakage between {a} and {b}"

for split in ["train", "val", "test"]:
    path = DIRS["manifests"] / f"{split}.jsonl"
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records[split]))
    print(f"{split:5s}: {len(records[split]):5d} samples from {len(clip_sets.get(split, ())):4d} clips -> {path}")
print(f"Dropped {n_dupes} near-duplicate windows")
print("Lighting (stored -> assembled):", {f"{a}->{b}": n for (a, b), n in sorted(light_change.items())})
assert records["train"], "train empty"
print("\n✅ Step 13 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 14 — Check the numbers

Look at the output:

- **Events per split** — if an event appears fewer than 30 times in train, the model will barely learn it.
- **Lighting** — a good mix of daylight, twilight and night.
- **Class presence** — how often each class appears.
- **Empty scenes** — some are good (the model must learn to say "none"), but not more than half.

The numbers are saved to `stats/summary.json` and the charts to `stats/distributions.png`.
''')

code(r'''
# ==== Cell 14: Audit ====
summary = {}
for split, recs in records.items():
    n = len(recs)
    ev = Counter(e["type"] for r in recs for e in r["target"]["events"])
    cls = Counter(c for r in recs for c in r["target"]["road_users"])
    light = Counter(r["target"]["lighting"] for r in recs)
    summary[split] = {
        "samples": n,
        "clips": len({r["clip_id"] for r in recs}),
        "events": {e: ev.get(e, 0) for e in EVENT_TYPES},
        "class_presence_rate": {c: round(cls.get(c, 0) / n, 3) for c in SCHEMA_CLASSES},
        "lighting": dict(light),
        "empty_scene_rate": round(sum(not r["target"]["road_users"] for r in recs) / n, 3),
        "mean_target_chars": round(float(np.mean([len(r["target_text"]) for r in recs])), 1),
    }
atomic_write_text(DIRS["stats"] / "summary.json", json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))

for e in EVENT_TYPES:
    if summary["train"]["events"][e] < 30:
        print(f"⚠️  '{e}' appears only {summary['train']['events'][e]} times in train. Add more clips or oversample it during training.")
if summary["train"]["empty_scene_rate"] > 0.5:
    print("⚠️  More than half of the samples are empty scenes")

chunks = {sp: set(SELECTION.loc[SELECTION.split == sp, "chunk"]) for sp in ["train", "val", "test"]}
print(f"\nChunk overlap train∩test: {len(chunks['train'] & chunks['test'])} (info only, not leakage)")

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
pd.DataFrame({s: summary[s]["events"] for s in summary}).plot.bar(ax=axes[0], title="events per split")
pd.DataFrame({s: summary[s]["class_presence_rate"] for s in summary}).plot.bar(ax=axes[1], title="class presence rate")
pd.DataFrame({s: summary[s]["lighting"] for s in summary}).fillna(0).plot.bar(ax=axes[2], title="lighting")
plt.tight_layout(); plt.savefig(DIRS["stats"] / "distributions.png", dpi=110); plt.show()
print("\n✅ Step 14 OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 15 — Review test labels by hand (optional)

Automatic labels can contain mistakes.

The final score should use labels that a person has checked.

How it works:

- Up to `REVIEW_LIMIT` test samples are shown, one at a time.
- You see the 6 camera images and the label.
- **Approve** — the label is right.
- **Save edit** — fix the label and save it. It is checked before saving.
- **Reject** — the sample is bad (blurry, broken). It is left out of the final test.

Your answers are saved to `review/test_reviews.jsonl`. You can stop and continue later.

Tip: look at the images first, then read the label.
''')

code(r'''
# ==== Cell 15: Review widget ====
import ipywidgets as W
from IPython.display import display

REVIEW_PATH = DIRS["review"] / "test_reviews.jsonl"
rng = random.Random(CFG.SEED)
review_pool = sorted(records["test"], key=lambda r: r["sample_id"])
rng.shuffle(review_pool)
review_pool = review_pool[: CFG.REVIEW_LIMIT]

def load_reviews():
    out = {}
    if REVIEW_PATH.exists():
        for line in REVIEW_PATH.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); out[r["sample_id"]] = r
    return out

reviews = load_reviews()
state = {"i": next((i for i, r in enumerate(review_pool) if r["sample_id"] not in reviews), 0)}
img_out, info = W.Output(), W.HTML()
box = W.Textarea(layout=W.Layout(width="100%", height="160px"))
btn_ok, btn_edit, btn_rej = W.Button(description="Approve", button_style="success"), \
    W.Button(description="Save edit", button_style="warning"), W.Button(description="Reject", button_style="danger")
btn_prev, btn_next = W.Button(description="◀ Prev"), W.Button(description="Next ▶")

def show():
    r = review_pool[state["i"]]
    done = len(load_reviews())
    prev_dec = reviews.get(r["sample_id"], {}).get("decision", "—")
    info.value = (f"<b>{state['i'] + 1}/{len(review_pool)}</b> | reviewed {done} | "
                  f"{r['sample_id']} | previous decision: <b>{prev_dec}</b>")
    box.value = json.dumps(reviews.get(r["sample_id"], {}).get("label", r["target"]), indent=1)
    with img_out:
        img_out.clear_output(wait=True)
        fig, axes = plt.subplots(2, 3, figsize=(15, 6))
        last = {f["camera"]: f for f in r["frames"]}       # frames sorted by time -> last = t0
        for ax, cam in zip(axes.ravel(), CAMERA_NAMES):
            ax.imshow(read_frame(r, last[cam])); ax.set_title(cam); ax.axis("off")
        plt.tight_layout(); plt.show()

def save(decision):
    r = review_pool[state["i"]]
    label = r["target"]
    if decision == "edited":
        label, errs = parse_and_validate(box.value)
        if errs:
            info.value = f"<span style='color:red'>❌ Invalid: {'; '.join(errs)}</span>"; return
    rec = {"sample_id": r["sample_id"], "decision": decision, "label": label if decision != "rejected" else None,
           "auto_label": r["target"], "reviewer_id": CFG.REVIEWER_ID,
           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(REVIEW_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    reviews[r["sample_id"]] = rec
    step(1)

def step(d):
    state["i"] = min(max(state["i"] + d, 0), len(review_pool) - 1); show()

btn_ok.on_click(lambda _: save("approved")); btn_edit.on_click(lambda _: save("edited"))
btn_rej.on_click(lambda _: save("rejected"))
btn_prev.on_click(lambda _: step(-1)); btn_next.on_click(lambda _: step(1))
display(info, img_out, box, W.HBox([btn_prev, btn_ok, btn_edit, btn_rej, btn_next]))
show()
''')

md(r'''
## Step 16 — Save the reviewed labels

- Creates `manifests/test_gold.jsonl` from your reviews (approved and edited samples).
- Shows how often the automatic labels were right, per field.
- If lighting is right less than 85% of the time, change the `SUN_*_DEG` settings, change `LIGHTING_VERSION`, and run Step 13 again.
''')

code(r'''
# ==== Cell 16: Gold export + auto-label agreement ====
reviews = load_reviews()
by_id = {r["sample_id"]: r for r in records["test"]}
gold, agree = [], Counter()
for sid, rv in reviews.items():
    if sid not in by_id:
        continue
    agree[rv["decision"]] += 1
    if rv["decision"] == "rejected":
        continue
    assert validate_target(rv["label"]) == [], sid
    auto = by_id[sid]["target"]
    for k in TARGET_KEYS:
        agree[f"{k}_match"] += int(auto[k] == rv["label"][k])
    agree["exact_match"] += int(auto == rv["label"])
    gold.append(by_id[sid] | {"target": rv["label"], "target_text": target_to_text(rv["label"]),
                              "review": {"decision": rv["decision"], "reviewer_id": rv["reviewer_id"], "utc": rv["utc"]}})
n_kept = len(gold)
atomic_write_text(DIRS["manifests"] / "test_gold.jsonl", "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gold))
print(f"Reviewed: {sum(agree[d] for d in ('approved', 'edited', 'rejected'))} "
      f"(approved {agree['approved']}, edited {agree['edited']}, rejected {agree['rejected']})")
if n_kept:
    for k in ["exact_match"] + [f"{k}_match" for k in TARGET_KEYS]:
        print(f"  auto-label {k:18s} {agree[k] / n_kept:6.1%}")
print(f"Gold test set: {n_kept} samples -> {DIRS['manifests'] / 'test_gold.jsonl'}")
RUN_META["gold"] = {"n": n_kept, "agreement": {k: agree[k] for k in agree}}
save_run_meta()
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 17 — Write the dataset card

`DATASET_CARD.md` describes how the dataset was made: source version, settings, counts, rules and known limits.

**License:** the images come from NVIDIA's dataset. Do not upload or share them publicly. Keep them in your own Drive.
''')

code(r'''
# ==== Cell 17: Dataset card ====
summary = json.loads((DIRS["stats"] / "summary.json").read_text())
total_bytes = sum(p.stat().st_size for p in DIRS["frames"].glob("*.zip"))
card = f"""# DriveAnnotator dataset — {ROOT.name}

- Source: `{CFG.DATASET_REPO}` @ `{DATASET_REVISION}` (split `{CFG.SPLIT_NAME}`, session-isolated)
- Built: {datetime.datetime.now(datetime.timezone.utc).isoformat()} | fingerprint `{RUN_META['label_fingerprint']}`
- Prompt `{CFG.PROMPT_VERSION}` | rules `{CFG.RULE_VERSION}` | obstacle feature `{OBS_FEATURE}` (frame `{OBSTACLE_FRAME}`)
- Input profile: Alpamayo2-Edge VQA — 6 cameras x 4 frames at {list(CFG.VQA_OFFSETS_US)} us, stored <= {CFG.STORE_MAX_PIXELS} px JPEG q{CFG.JPEG_QUALITY}
- Frames on disk: {total_bytes / 1e9:.2f} GB

## Splits
| split | samples | clips | empty-scene rate |
|---|---|---|---|
""" + "".join(f"| {s} | {v['samples']} | {v['clips']} | {v['empty_scene_rate']} |\n" for s, v in summary.items()) + f"""
Gold (human-reviewed) test samples: {RUN_META.get('gold', {}).get('n', 0)}

## Events (train / val / test)
""" + "".join(f"- {e}: " + " / ".join(str(summary[s]['events'][e]) for s in ['train', 'val', 'test'] if s in summary) + "\n"
              for e in EVENT_TYPES) + f"""
## Label rules
Targets are generated from the dataset's 3D obstacle labels projected into each camera with NVIDIA calibration.
Visible = centre in image, depth > {CFG.MIN_DEPTH_M} m, projected height >= {CFG.MIN_BOX_HEIGHT_PX} px (native), within class range.
Motion-based events use only the {CFG.CUT_IN_LOOKBACK_US / 1e6:.1f} s the model sees. See notebook cell 10 for exact rules.

## Known limitations
- No occlusion reasoning (no LiDAR visibility test): hidden objects can be labelled visible.
- Lighting comes from approximate sun elevation (country centroid, mid-month, hourly local time, DST assumed Apr-Oct): +-1 h error near sunrise/sunset; tunnels/garages are not detected.
- Only 12 dynamic obstacle classes exist in the source; static objects, signs, lights are not labelled.
- Corridor is a curvature arc from current ego state, not the true future path.
- Distances are from the ego rear axle.

## License
Derived from the NVIDIA PhysicalAI-AV dataset under its license. Do not redistribute frames publicly.
"""
atomic_write_text(ROOT / "DATASET_CARD.md", card)
print(card)
print("Files:")
for p in sorted(DIRS["manifests"].glob("*.jsonl")):
    print(f"  {p}  ({sum(1 for _ in open(p, encoding='utf-8'))} lines)")
print("\n✅ Dataset stage complete. Next: training notebook (LoRA) will read manifests/ + frames/.")
''')

# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "cells": [],
}
for i, (kind, src) in enumerate(cells):
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
    if kind == "code":
        cell |= {"execution_count": None, "outputs": []}
    nb["cells"].append(cell)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
