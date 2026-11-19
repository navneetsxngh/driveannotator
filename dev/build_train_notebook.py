# Generates 02_train_lora.ipynb (English, beginner-friendly, no test cells).
# Tests for the code in this notebook live in dev/test_train_notebook.py.
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
# DriveAnnotator — Step 2: Fine-tune Alpamayo2-Edge with LoRA

## What this notebook does

- It loads the dataset you built in `01_build_dataset.ipynb`.
- It loads NVIDIA's `Alpamayo2-Edge` model (2.65 billion parameters).
- It teaches the model to describe a driving scene from 6 cameras.
- It only trains a small add-on called a **LoRA adapter** (about 16 million parameters).
- At the end it compares the model **before** and **after** training.

## What the model learns to write

For every scene, the model writes four short lines:

```
lighting: daylight
road users: car four to seven, pedestrian one
cameras: front camera car, pedestrian; front right camera pedestrian
hazards: none
```

## How to run it

1. In Colab, open **Runtime → Change runtime type** and pick **A100 GPU**.
2. Add your Hugging Face token in the 🔑 **Secrets** panel as `HF_TOKEN`.
3. Make sure the dataset is in Google Drive at `MyDrive/drive_annotator/dataset_v1`.
4. Run the cells from top to bottom.
5. When you are done, open **Runtime → Disconnect and delete runtime**. This stops using compute units.
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 1 — Install libraries

This cell installs the exact library versions we need.

- `transformers==5.17.0` contains the model class that Alpamayo2-Edge uses.
- `peft` gives us LoRA.
- `alpamayo2-edge` is NVIDIA's code for loading the model and preparing camera images.
- We remove `torchao`, because the old copy in Colab makes `peft` crash. We do not need it.
''')

code(r'''
import sys, os, subprocess, json, math, time, io, zipfile, random, shutil
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
A2E_COMMIT = "02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e"
PACKAGES = ["transformers==5.17.0", "peft==0.21.3", "accelerate>=1.12.0", "hydra-core>=1.3.2",
            "einops>=0.8.1", "safetensors>=0.8.0", "colorlog>=6.0", "huggingface_hub>=1.0,<2"]

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--ignore-requires-python",
                    f"git+https://github.com/NVlabs/alpamayo2-edge@{A2E_COMMIT}"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import transformers, peft
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from torchvision.transforms.functional import InterpolationMode, resize
from alpamayo2_edge.chat_template import build_vqa_conversation, CLOSED_THINKING
from alpamayo2_edge.sample import canonicalize_vqa_sample, VQA_CAMERA_PIXEL_BOUNDS
from alpamayo2_edge.helper import _normalize_patches, get_processor, to_device
from alpamayo2_edge.text_tasks import generate_vqa_answer

print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 2 — Check the GPU

Training needs a GPU that supports `bfloat16` and has at least 20 GB of memory.

An A100 (40 GB) is ideal.
''')

code(r'''
assert torch.cuda.is_available(), "No GPU found. Use Runtime -> Change runtime type -> A100 GPU."
GPU = torch.cuda.get_device_properties(0)
VRAM_GB = GPU.total_memory / 1e9
assert torch.cuda.is_bf16_supported() and VRAM_GB >= 20, f"{GPU.name} is not enough. Use an A100."

torch.backends.cuda.matmul.allow_tf32 = True   # faster matrix math on A100
DEVICE = "cuda"
print(f"GPU: {GPU.name} | {VRAM_GB:.0f} GB")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 3 — Settings

All settings live in one place, so you can change them easily.

| Setting | Value | Why |
|---|---|---|
| `LORA_R` | 16 | Size of the LoRA adapter. Big enough to learn the new answer style, small enough not to overfit 1,100 samples. |
| `LORA_ALPHA` | 32 | Scales the adapter. `alpha / r = 2` is a common default. |
| `LR` | 1e-4 | Learning rate. LoRA can use a larger rate than full fine-tuning. |
| `GRAD_ACCUM` | 8 | We process 1 sample at a time and update after 8. So each update sees 8 samples. |
| `EPOCHS` | 3 | Each training sample is seen 3 times. We keep the best checkpoint, not the last one. |
| `EVENT_OVERSAMPLE` | 3 | Scenes with hazards are rare, so we show them 3 times per epoch. |
| `UNITS_PER_HOUR` | 13 | How many Colab compute units an A100 uses per hour. Check the Resources panel. |
| `UNITS_BUDGET` | 60 | The run stops before training if the estimate is higher than this. |

We train LoRA only on the **language part** of the model.

The vision part (which reads the images) stays frozen.
''')

code(r'''
@dataclass
class Settings:
    # Paths
    DRIVE_DATA_ROOT: str = "/content/drive/MyDrive/drive_annotator/dataset_v1"
    LOCAL_DATA_ROOT: str = "/content/dataset_v1"          # fast local copy of the dataset
    OUT_DIR: str = "/content/drive/MyDrive/drive_annotator/runs/lora_v2"
    HF_CACHE_DIR: str = "/content/hf_cache"
    # Model
    MODEL_ID: str = "nvidia/Alpamayo2-Edge"
    MODEL_REVISION: str = "b92b6aa326bd3e2235db28f248ad5a30b5890e64"
    # LoRA
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    LORA_TARGETS: str = r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(fc1|fc2))"
    # Training
    LR: float = 1e-4
    WARMUP_RATIO: float = 0.03
    MIN_LR_RATIO: float = 0.1
    MAX_GRAD_NORM: float = 1.0
    GRAD_ACCUM: int = 8
    EPOCHS: int = 3
    EVENT_OVERSAMPLE: int = 3
    SEED: int = 42
    # Evaluation
    EVAL_EVERY_FRAC: float = 0.25    # check validation loss 4 times per epoch
    EARLY_STOP_PATIENCE: int = 4     # stop after 4 checks without improvement
    GEN_EVAL_N: int = 64             # validation samples used to score generated answers
    MAX_NEW_TOKENS: int = 200
    # Safety checks
    OVERFIT_N: int = 8
    OVERFIT_STEPS: int = 60
    OVERFIT_LR: float = 3e-4
    OVERFIT_TARGET_LOSS: float = 0.3
    PROBE_STEPS: int = 20
    # Budget
    UNITS_PER_HOUR: float = 13.0
    UNITS_BUDGET: float = 60.0
    # Data loading
    NUM_WORKERS: int = 4
    LOG_EVERY: int = 10


CFG = Settings()
random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
print(json.dumps(asdict(CFG), indent=1))
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 4 — Load the dataset

This cell:

1. Connects Google Drive (a popup asks for permission — click **Allow**).
2. Copies the dataset from Drive to the Colab disk. Reading from the local disk is much faster.
3. Loads the three lists of samples: `train`, `val` and `test`.

Each sample has 24 camera images (6 cameras × 4 frames) and a `target` answer.
''')

code(r'''
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

SRC, DATA = Path(CFG.DRIVE_DATA_ROOT), Path(CFG.LOCAL_DATA_ROOT)
assert (SRC / "manifests" / "train.jsonl").exists(), f"Dataset not found at {SRC}"

start, copied = time.time(), 0
for sub in ["manifests", "frames"]:
    (DATA / sub).mkdir(parents=True, exist_ok=True)
    for f in (SRC / sub).iterdir():
        local = DATA / sub / f.name
        if not local.exists() or local.stat().st_size != f.stat().st_size:   # skip files already copied
            shutil.copyfile(f, local)
            copied += 1
print(f"Copied {copied} files to the local disk in {time.time() - start:.0f} s")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


SPLITS = {name: read_jsonl(DATA / "manifests" / f"{name}.jsonl") for name in ["train", "val", "test"]}
gold = DATA / "manifests" / "test_gold.jsonl"
SPLITS["test_gold"] = read_jsonl(gold) if gold.exists() else []   # human-reviewed test samples (optional)

OUT = Path(CFG.OUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)
print({name: len(rows) for name, rows in SPLITS.items()})
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 5 — Log in to Hugging Face

The model weights are on Hugging Face.

The token comes from Colab Secrets (`HF_TOKEN`).
''')

code(r'''
from huggingface_hub import HfApi

HF_TOKEN = None
if IN_COLAB:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Add HF_TOKEN in Colab Secrets."

os.environ["HF_TOKEN"] = HF_TOKEN
HfApi(token=HF_TOKEN).auth_check(CFG.MODEL_ID)
print("Hugging Face access OK")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 6 — Load the model

We use NVIDIA's own loader. It stops with an error if any weight is missing.

We **freeze** every weight. Nothing in the base model will change during training.
''')

code(r'''
from alpamayo2_edge.checkpoint import load_model

start = time.time()
model, _ = load_model(CFG.MODEL_ID, device=DEVICE, dtype="bfloat16", attn_implementation="sdpa",
                      revision=CFG.MODEL_REVISION, cache_dir=CFG.HF_CACHE_DIR)
model.requires_grad_(False)
model.eval()

TOK = model.tokenizer                        # turns text into token ids
PROC = get_processor(TOK, model.config)      # turns text + images into model inputs
MODEL_CFG = model.config
EOT = "<|im_end|>"                           # the token that ends an answer

n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded {n_params / 1e9:.2f}B parameters in {time.time() - start:.0f} s | GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 7 — The answer format

The dataset stores each answer as structured data (a Python dictionary).

The model writes the answer as **four plain English lines**.

### Why plain English and not JSON?

We tested both on the real model:

| Answer format | Tokens per answer | Tokens the model rarely saw | Loss after 40 steps on 8 samples |
|---|---|---|---|
| JSON | 70 | 23% | 2.26 |
| Plain English lines | 48 | 2% | 0.25 |

Alpamayo2-Edge was trained to write driving explanations in English.

JSON symbols like `"],` were almost never in its training data, so it learns them slowly.

Plain English is shorter and the model learns it about 10 times faster.

### Two helper functions

- `render_answer(target)` turns a dictionary into the four lines (used for training).
- `parse_answer(text)` turns the four lines back into a dictionary (used for scoring).
''')

code(r'''
CAMERA_NAMES = ["cross_left", "front_wide", "cross_right", "rear_left", "rear_right", "front_tele"]
CLASSES = ["car", "truck", "bus", "tram", "other_vehicle", "pedestrian", "rider", "animal"]
EVENT_TYPES = ["pedestrian_in_path", "vehicle_cut_in", "lead_vehicle_close", "vru_near_ego"]
LIGHTING = ["daylight", "twilight", "night"]

# Dataset value -> English words written by the model
CAMERA_TEXT = {"cross_left": "front left camera", "front_wide": "front camera", "cross_right": "front right camera",
               "rear_left": "rear left camera", "rear_right": "rear right camera", "front_tele": "front telephoto camera"}
CLASS_TEXT = {c: c.replace("_", " ") for c in CLASSES}
COUNT_TEXT = {"1": "one", "2-3": "two to three", "4-7": "four to seven", "8+": "eight or more"}
DISTANCE_TEXT = {"<5m": "under 5 meters", "5-10m": "5 to 10 meters", "10-20m": "10 to 20 meters",
                 "20-40m": "20 to 40 meters", "40m+": "over 40 meters"}
HAZARD_TEXT = {"pedestrian_in_path": "pedestrian in path", "vehicle_cut_in": "vehicle cutting in",
               "lead_vehicle_close": "lead vehicle close", "vru_near_ego": "vulnerable road user near ego"}


def reverse(mapping):
    return {text: value for value, text in mapping.items()}


TEXT_CAMERA, TEXT_CLASS, TEXT_COUNT = reverse(CAMERA_TEXT), reverse(CLASS_TEXT), reverse(COUNT_TEXT)
TEXT_DISTANCE, TEXT_HAZARD = reverse(DISTANCE_TEXT), reverse(HAZARD_TEXT)

PROMPT = (
    "You are a driving-scene annotator. Look at the six camera views (four frames each, covering the last 0.3 seconds). "
    "Answer in exactly four lines.\n"
    "lighting: daylight, twilight or night\n"
    "road users: each visible class with a count (one, two to three, four to seven, eight or more), or none\n"
    "cameras: each camera that sees road users, followed by the classes it sees; separate cameras with semicolons; or none\n"
    "hazards: each hazard as '<type> on <camera> at <distance>'; separate hazards with semicolons; or none\n"
    "Classes: car, truck, bus, tram, other vehicle, pedestrian, rider, animal. "
    "Hazard types: pedestrian in path, vehicle cutting in, lead vehicle close, vulnerable road user near ego. "
    "Distances: under 5 meters, 5 to 10 meters, 10 to 20 meters, 20 to 40 meters, over 40 meters."
)


def render_answer(target):
    """Dictionary -> four English lines."""
    road = ", ".join(f"{CLASS_TEXT[c]} {COUNT_TEXT[n]}" for c, n in target["road_users"].items())
    cams = "; ".join(CAMERA_TEXT[cam] + " " + ", ".join(CLASS_TEXT[c] for c in seen)
                     for cam, seen in target["cameras"].items())
    hazards = "; ".join(f"{HAZARD_TEXT[e['type']]} on {CAMERA_TEXT[e['camera']]} at {DISTANCE_TEXT[e['distance']]}"
                        for e in target["events"])
    return (f"lighting: {target['lighting']}\n"
            f"road users: {road or 'none'}\n"
            f"cameras: {cams or 'none'}\n"
            f"hazards: {hazards or 'none'}")


def split_known_prefix(text, options):
    """Return (option, rest) for the longest option that `text` starts with."""
    for option in sorted(options, key=len, reverse=True):
        if text == option or text.startswith(option + " "):
            return option, text[len(option):].strip()
    raise ValueError(f"unknown item {text!r}")


def parse_answer(text):
    """Four English lines -> (dictionary, None), or (None, error message) if the text is not valid."""
    text = text.strip().removeprefix(CLOSED_THINKING).strip()
    lines = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            lines[key.strip().lower()] = value.strip()
    try:
        assert lines.get("lighting") in LIGHTING, "bad lighting line"
        road = {}
        if lines["road users"] != "none":
            for item in lines["road users"].split(", "):
                cls, count = split_known_prefix(item, TEXT_CLASS)
                road[TEXT_CLASS[cls]] = TEXT_COUNT[count]
        cams = {}
        if lines["cameras"] != "none":
            for item in lines["cameras"].split("; "):
                cam, rest = split_known_prefix(item, TEXT_CAMERA)
                cams[TEXT_CAMERA[cam]] = [TEXT_CLASS[c] for c in rest.split(", ")]
        events = []
        if lines["hazards"] != "none":
            for item in lines["hazards"].split("; "):
                kind, rest = item.split(" on ", 1)
                cam, dist = rest.rsplit(" at ", 1)
                events.append({"type": TEXT_HAZARD[kind], "camera": TEXT_CAMERA[cam], "distance": TEXT_DISTANCE[dist]})
    except (KeyError, ValueError, AssertionError) as error:
        return None, f"{type(error).__name__}: {error}"
    # Put everything in the same order the dataset uses
    target = {
        "lighting": lines["lighting"],
        "road_users": {c: road[c] for c in CLASSES if c in road},
        "cameras": {cam: [c for c in CLASSES if c in cams[cam]] for cam in CAMERA_NAMES if cam in cams},
        "events": sorted(events, key=lambda e: EVENT_TYPES.index(e["type"])),
    }
    if any(c not in target["road_users"] for seen in target["cameras"].values() for c in seen):
        return None, "a camera lists a class that is missing from road users"
    return target, None


example = SPLITS["train"][0]["target"]
print(render_answer(example))
print("Round trip OK:", parse_answer(render_answer(example))[0] == example)
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 8 — Turn one sample into model input

For each sample we:

1. Read the 24 camera images from its zip file.
2. Build the prompt the same way NVIDIA's own code does.
3. Resize the images to the size the model expects.
4. Add the answer after the prompt: `<think></think>` + four lines + `<|im_end|>`.

`<think></think>` is how this model always starts an answer.

We also remember how many tokens the answer has (`n_answer`).

The loss is computed **only on those answer tokens**, not on the prompt or the images.
''')

code(r'''
MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")


def load_frames(record):
    """Read the 24 JPEG frames of one sample."""
    with zipfile.ZipFile(DATA / record["frames_zip"]) as z:
        images = [np.asarray(Image.open(io.BytesIO(z.read(f["path"]))).convert("RGB")) for f in record["frames"]]
    return {
        "image_frames": torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(1).contiguous(),
        "camera_indices": torch.tensor([f["camera_id"] for f in record["frames"]], dtype=torch.int64),
        "absolute_timestamps": torch.tensor([f["timestamp_us"] for f in record["frames"]], dtype=torch.int64),
        "camera_pixel_bounds": dict(VQA_CAMERA_PIXEL_BOUNDS),
    }


def encode(record, with_answer=True):
    """One sample -> tensors for the model."""
    data = canonicalize_vqa_sample(load_frames(record))
    messages = build_vqa_conversation(data, PROMPT, include_camera_ids=MODEL_CFG.include_camera_ids,
                                      system_prompt=getattr(MODEL_CFG, "system_prompt", None))
    prompt = PROC.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=False,
                                      continue_final_message=False, enable_thinking=False)
    prompt = prompt.removesuffix(CLOSED_THINKING)          # same prompt NVIDIA uses for question answering

    image_proc = PROC.image_processor
    images = []
    for message in messages:
        for part in message["content"]:
            if part["type"] == "image":
                frame = part["image"].squeeze(0)
                h, w = smart_resize(*frame.shape[-2:], factor=image_proc.patch_size * image_proc.merge_size,
                                    min_pixels=part["min_pixels"], max_pixels=part["max_pixels"])
                images.append(resize(frame, [h, w], interpolation=InterpolationMode.BICUBIC))

    answer = CLOSED_THINKING + render_answer(record["target"]) + EOT
    text = prompt + answer if with_answer else prompt
    out = dict(PROC(text=text, images=images, padding=False, return_tensors="pt", return_mm_token_type_ids=True,
                    do_resize=False, do_rescale=False, do_normalize=False))
    out["pixel_values"] = _normalize_patches(out["pixel_values"], image_proc)
    if with_answer:
        out["n_answer"] = len(TOK(answer, add_special_tokens=False)["input_ids"])
    return out


class AnnotationDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return encode(self.records[i])


def make_loader(records):
    """Prepare samples in background workers while the GPU trains."""
    return torch.utils.data.DataLoader(AnnotationDataset(records), batch_size=None, shuffle=False,
                                       num_workers=CFG.NUM_WORKERS, prefetch_factor=2 if CFG.NUM_WORKERS else None)


sample = encode(SPLITS["train"][0])
print(f"One sample = {sample['input_ids'].shape[1]} tokens, of which {sample['n_answer']} are the answer")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 9 — Scoring

To score the model, we let it **write** an answer and compare it with the label.

| Score | Meaning |
|---|---|
| `valid` | The answer has the four lines in the right format |
| `exact_match` | The whole answer is exactly right |
| `lighting_acc` | Lighting is right |
| `road_users_f1` | Which classes are present (car, pedestrian, ...) |
| `count_acc` | For classes it got right, is the count bucket right |
| `cameras_f1` | Which camera sees which class |
| `hazards_f1` | Which hazards are present |

`f1` is a score from 0 to 1 that balances "found everything" and "did not invent things".

An invalid answer counts as an empty answer, so it gets no credit.
''')

code(r'''
def f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(predictions, targets):
    """predictions: parsed dictionaries (or None if invalid). targets: label dictionaries."""
    empty = {"lighting": None, "road_users": {}, "cameras": {}, "events": []}
    c = Counter()
    for pred, true in zip(predictions, targets):
        c["valid"] += pred is not None
        pred = pred or empty
        c["exact"] += pred == true
        c["light"] += pred["lighting"] == true["lighting"]
        p, t = set(pred["road_users"]), set(true["road_users"])
        c["ru_tp"] += len(p & t); c["ru_fp"] += len(p - t); c["ru_fn"] += len(t - p)
        c["count_ok"] += sum(pred["road_users"][k] == true["road_users"][k] for k in p & t)
        p = {(cam, k) for cam, ks in pred["cameras"].items() for k in ks}
        t = {(cam, k) for cam, ks in true["cameras"].items() for k in ks}
        c["cam_tp"] += len(p & t); c["cam_fp"] += len(p - t); c["cam_fn"] += len(t - p)
        p, t = {e["type"] for e in pred["events"]}, {e["type"] for e in true["events"]}
        c["hz_tp"] += len(p & t); c["hz_fp"] += len(p - t); c["hz_fn"] += len(t - p)
    n = max(len(targets), 1)
    return {
        "n": len(targets),
        "valid": round(c["valid"] / n, 3),
        "exact_match": round(c["exact"] / n, 3),
        "lighting_acc": round(c["light"] / n, 3),
        "road_users_f1": round(f1(c["ru_tp"], c["ru_fp"], c["ru_fn"]), 3),
        "count_acc": round(c["count_ok"] / c["ru_tp"], 3) if c["ru_tp"] else 0.0,
        "cameras_f1": round(f1(c["cam_tp"], c["cam_fp"], c["cam_fn"]), 3),
        "hazards_f1": round(f1(c["hz_tp"], c["hz_fp"], c["hz_fn"]), 3),
        "true_hazards": c["hz_tp"] + c["hz_fn"],
    }


@torch.no_grad()
def predict(record):
    """Let the model write an answer for one sample."""
    enc = encode(record, with_answer=False)
    inputs = {"tokenized_data": to_device({k: enc[k] for k in MODEL_INPUT_KEYS}, DEVICE)}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return generate_vqa_answer(model, inputs, max_new_tokens=CFG.MAX_NEW_TOKENS)


def evaluate(records, name, save_to=None):
    """Generate answers for `records` and score them."""
    was_training = model.vlm.training
    model.eval()
    model.vlm.config.use_cache = True          # generation is much faster with the cache on
    start, texts, parsed = time.time(), [], []
    for record in records:
        text = predict(record)
        texts.append(text)
        parsed.append(parse_answer(text)[0])
    metrics = score(parsed, [r["target"] for r in records])
    metrics["sec_per_sample"] = round((time.time() - start) / max(len(records), 1), 2)
    if save_to:
        Path(save_to).write_text(json.dumps({"metrics": metrics, "answers": [
            {"sample_id": r["sample_id"], "answer": t} for r, t in zip(records, texts)]}, indent=1))
    model.vlm.config.use_cache = False
    if was_training:
        model.vlm.train()
    print(f"[{name}] " + " | ".join(f"{k} {v}" for k, v in metrics.items()))
    return metrics, texts
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 10 — Score the model before training

We ask the untrained model the same question.

This gives us a starting point to compare against later.

The untrained model does not know our answer format, so a low score here is normal.

The result is saved, so re-running this cell does not repeat the work.
''')

code(r'''
VAL_SUBSET = random.Random(CFG.SEED).sample(SPLITS["val"], min(CFG.GEN_EVAL_N, len(SPLITS["val"])))
baseline_file = OUT / "baseline.json"

if baseline_file.exists():
    BASELINE = json.loads(baseline_file.read_text())
    print("Loaded saved baseline")
else:
    BASELINE = {}
    BASELINE["val"], texts = evaluate(VAL_SUBSET, "before training, val", OUT / "baseline_val_answers.json")
    BASELINE["test"], _ = evaluate(SPLITS["test"], "before training, test", OUT / "baseline_test_answers.json")
    baseline_file.write_text(json.dumps(BASELINE, indent=1))
    print("Example answer before training:", texts[0][:300])
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 11 — Add the LoRA adapter

LoRA adds two small matrices next to some layers of the model.

Only these small matrices are trained. The original weights stay frozen.

We add LoRA to 6 layer types in each of the 28 language layers = **168 places**.

The vision part has layers with the same names, so we use a pattern that only matches the language part.

`attach_lora()` can also be called again later to start over with a fresh adapter.
''')

code(r'''
PEFT = None


def attach_lora():
    """Add a fresh LoRA adapter (removes the previous one if there is one)."""
    global PEFT
    if PEFT is not None:
        PEFT.unload()
    config = LoraConfig(r=CFG.LORA_R, lora_alpha=CFG.LORA_ALPHA, lora_dropout=CFG.LORA_DROPOUT,
                        target_modules=CFG.LORA_TARGETS, bias="none")
    PEFT = get_peft_model(model.vlm, config)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    for _, p in trainable:
        p.data = p.data.float()                 # keep the adapter in full precision
    modules = {name.rsplit(".lora_", 1)[0] for name, _ in trainable}
    assert len(modules) == 168 and all(".language_model." in m for m in modules), "LoRA was added in the wrong place"
    model.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # saves GPU memory
    model.vlm.config.use_cache = False
    return trainable


TRAINABLE = attach_lora()
n_trainable = sum(p.numel() for _, p in TRAINABLE)
print(f"Trainable parameters: {n_trainable / 1e6:.1f}M ({n_trainable / n_params:.2%} of the model)")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 12 — The loss function

The loss measures how surprised the model is by the correct answer. Lower is better.

Each sample has about 5,000 tokens, but only about 50 belong to the answer.

So we only compute the loss on the answer tokens.

This is also faster, because the model only predicts ~50 words instead of 5,000.
''')

code(r'''
def answer_loss(enc):
    """Average cross-entropy over the answer tokens of one sample."""
    n = int(enc["n_answer"])
    inputs = {k: enc[k].to(DEVICE) for k in MODEL_INPUT_KEYS}
    ids = inputs["input_ids"]
    length = ids.shape[1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model.vlm.model(**inputs, use_cache=False).last_hidden_state
        logits = model.vlm.lm_head(hidden[:, length - n - 1:length - 1])   # position t predicts token t + 1
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, length - n:].reshape(-1))
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 13 — Safety check: can the model memorize 8 samples?

Before spending an hour on training, we make sure everything is wired correctly.

We train on just 8 samples for 60 steps.

If everything works, the model should almost memorize them (loss below 0.3).

If it does not, something is broken, and the cell stops here.

Afterwards the adapter is reset, so this check does not affect the real training.

If you are coming back to a run that already started (the folder `last/` exists on Drive), you can skip this step.
''')

code(r'''
model.vlm.train()
few = [encode(r) for r in SPLITS["train"][:CFG.OVERFIT_N]]
optimizer = torch.optim.AdamW([p for _, p in TRAINABLE], lr=CFG.OVERFIT_LR)
losses = []
for step in range(CFG.OVERFIT_STEPS):
    loss = answer_loss(few[step % len(few)])
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for _, p in TRAINABLE], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    losses.append(loss.item())

final = float(np.mean(losses[-len(few):]))
plt.figure(figsize=(6, 2.5)); plt.plot(losses); plt.yscale("log"); plt.title("memorize 8 samples"); plt.show()
print(f"Loss went from {losses[0]:.2f} to {final:.2f}")
assert final < CFG.OVERFIT_TARGET_LOSS, "The model could not memorize 8 samples. Do not start training."

del optimizer, few
TRAINABLE = attach_lora()      # fresh adapter for the real training
torch.cuda.empty_cache()
print("Check passed. Adapter reset.")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 14 — Safety check: how long and how many units?

We time 20 real training steps and estimate the whole run.

If the estimate is above `UNITS_BUDGET`, the cell stops.

Then you can lower `EPOCHS` or `GEN_EVAL_N`, or raise the budget.
''')

code(r'''
def epoch_order(epoch):
    """Training order for one epoch. Samples with hazards appear EVENT_OVERSAMPLE times."""
    order = []
    for i, record in enumerate(SPLITS["train"]):
        order += [i] * (CFG.EVENT_OVERSAMPLE if record["target"]["events"] else 1)
    random.Random(CFG.SEED + epoch).shuffle(order)
    return order


SAMPLES_PER_EPOCH = len(epoch_order(0))
UPDATES_PER_EPOCH = math.ceil(SAMPLES_PER_EPOCH / CFG.GRAD_ACCUM)
TOTAL_UPDATES = UPDATES_PER_EPOCH * CFG.EPOCHS
EVAL_EVERY = max(1, round(UPDATES_PER_EPOCH * CFG.EVAL_EVERY_FRAC))

model.vlm.train()
times = []
for enc in make_loader([SPLITS["train"][i] for i in epoch_order(0)[:CFG.PROBE_STEPS + 2]]):
    torch.cuda.synchronize(); start = time.time()
    (answer_loss(enc) / CFG.GRAD_ACCUM).backward()
    torch.cuda.synchronize(); times.append(time.time() - start)
for _, p in TRAINABLE:
    p.grad = None                              # throw away the test gradients
sec_per_sample = float(np.median(times[2:]))   # skip the first two (warm-up)

train_hours = SAMPLES_PER_EPOCH * CFG.EPOCHS * sec_per_sample / 3600
check_hours = (TOTAL_UPDATES // EVAL_EVERY) * len(SPLITS["val"]) * sec_per_sample / 3 / 3600
generate_hours = (CFG.EPOCHS * len(VAL_SUBSET) + len(SPLITS["test"])) * 2.5 / 3600
total_hours = train_hours + check_hours + generate_hours
units = total_hours * CFG.UNITS_PER_HOUR

print(f"{SAMPLES_PER_EPOCH} samples per epoch | {TOTAL_UPDATES} updates in total | {sec_per_sample:.2f} s per sample")
print(f"Estimated time: {total_hours:.1f} h | estimated cost: {units:.0f} units (budget {CFG.UNITS_BUDGET:.0f})")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of {VRAM_GB:.0f} GB")
assert units <= CFG.UNITS_BUDGET, "Over budget. Lower EPOCHS or GEN_EVAL_N, or raise UNITS_BUDGET."
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 15 — Train

How it works:

1. The model reads one sample and computes the loss on the answer.
2. After 8 samples, the adapter weights are updated (`GRAD_ACCUM = 8`).
3. The learning rate warms up first, then slowly goes down (cosine schedule).
4. Four times per epoch we measure the loss on the validation set.
5. If the validation loss is the best so far, we save the adapter to `best/`.
6. At the end of each epoch we let the model write answers for 64 validation samples and score them.

### If Colab disconnects

Progress is saved to Google Drive (`last/`).

Run Steps 1–12 and Step 14 again (skip Step 13), then run this cell. It continues where it stopped.

If training had already finished, the cell does not train again. It loads the saved progress and prints the full training log.

### How to read the log

- `loss` (training) should go down.
- `val loss` should also go down.
- If `loss` keeps going down but `val loss` goes up, the model is memorizing. The `best/` checkpoint protects you.
''')

code(r'''
def lr_factor(step):
    """Warm up linearly, then follow a cosine curve down to MIN_LR_RATIO."""
    warmup = max(1, int(TOTAL_UPDATES * CFG.WARMUP_RATIO))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, TOTAL_UPDATES - warmup))
    return CFG.MIN_LR_RATIO + (1 - CFG.MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validation_loss():
    model.eval()
    total, tokens = 0.0, 0
    for enc in make_loader(SPLITS["val"]):
        total += answer_loss(enc).item() * enc["n_answer"]
        tokens += enc["n_answer"]
    model.vlm.train()
    return total / tokens


def save_checkpoint(folder, optimizer=None, scheduler=None, state=None):
    """Save the adapter (and the training progress) to Google Drive."""
    tmp = OUT / (folder + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    PEFT.save_pretrained(str(tmp))
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": state},
                   tmp / "trainer_state.pt")
    shutil.rmtree(OUT / folder, ignore_errors=True)
    os.replace(tmp, OUT / folder)


def load_adapter(folder):
    from safetensors.torch import load_file
    set_peft_model_state_dict(PEFT, load_file(str(OUT / folder / "adapter_model.safetensors")))
    for _, p in TRAINABLE:
        p.data = p.data.float()


optimizer = torch.optim.AdamW([p for _, p in TRAINABLE], lr=CFG.LR, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
state = {"step": 0, "epoch": 0, "position": 0, "best_val": float("inf"), "best_step": 0,
         "bad_checks": 0, "history": [], "val_scores": [], "stopped_early": False}

if (OUT / "last" / "trainer_state.pt").exists():
    load_adapter("last")
    saved = torch.load(OUT / "last" / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"]); state = saved["state"]
    print(f"Resuming from update {state['step']} (epoch {state['epoch']})")
else:
    state["history"].append({"step": 0, "val_loss": validation_loss()})
    print(f"Validation loss before training: {state['history'][0]['val_loss']:.3f}")

model.vlm.train()
run_start, recent = time.time(), []
for epoch in range(state["epoch"], CFG.EPOCHS):
    if state["stopped_early"]:
        break
    order = epoch_order(epoch)
    start_at = state["position"] if epoch == state["epoch"] else 0
    for position, enc in enumerate(make_loader([SPLITS["train"][i] for i in order[start_at:]]), start=start_at):
        loss = answer_loss(enc)
        (loss / CFG.GRAD_ACCUM).backward()
        recent.append(loss.item())
        if (position + 1) % CFG.GRAD_ACCUM and position != len(order) - 1:
            continue                                            # keep collecting gradients

        torch.nn.utils.clip_grad_norm_([p for _, p in TRAINABLE], CFG.MAX_GRAD_NORM)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        state.update(step=state["step"] + 1, epoch=epoch, position=position + 1)

        if state["step"] % CFG.LOG_EVERY == 0:
            print(f"epoch {epoch} | update {state['step']}/{TOTAL_UPDATES} | loss {np.mean(recent[-80:]):.3f} | "
                  f"lr {scheduler.get_last_lr()[0]:.1e} | {(time.time() - run_start) / 3600:.2f} h")

        if state["step"] % EVAL_EVERY == 0 or state["step"] == TOTAL_UPDATES:
            val = validation_loss()
            state["history"].append({"step": state["step"], "train_loss": float(np.mean(recent[-80:])), "val_loss": val})
            if val < state["best_val"] - 1e-4:
                state.update(best_val=val, best_step=state["step"], bad_checks=0)
                save_checkpoint("best")
                print(f"   val loss {val:.3f} (new best, saved)")
            else:
                state["bad_checks"] += 1
                print(f"   val loss {val:.3f} (no improvement {state['bad_checks']}x)")
            save_checkpoint("last", optimizer, scheduler, state)
            if state["bad_checks"] >= CFG.EARLY_STOP_PATIENCE:
                state["stopped_early"] = True
                save_checkpoint("last", optimizer, scheduler, state)
                print("   Stopping early: validation loss stopped improving.")
                break
    if state["stopped_early"]:
        break
    scores, _ = evaluate(VAL_SUBSET, f"after epoch {epoch}, val", OUT / f"val_answers_epoch{epoch}.json")
    state["val_scores"].append({"epoch": epoch, **scores})
    state.update(epoch=epoch + 1, position=0)
    save_checkpoint("last", optimizer, scheduler, state)

pd.DataFrame(state["history"]).to_csv(OUT / "history.csv", index=False)
print(f"Done: {state['step']} updates | best val loss {state['best_val']:.3f} at update {state['best_step']}")

# Full training log, read from the saved progress (also works after a disconnect)
print("\nValidation checks:")
for h in state["history"]:
    train = f"train loss {h['train_loss']:.3f} | " if "train_loss" in h else ""
    print(f"  update {h['step']:3d} | {train}val loss {h['val_loss']:.3f}")
print("\nValidation scores after each epoch:")
for s in state["val_scores"]:
    print("  " + " | ".join(f"{k} {v}" for k, v in s.items()))
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 16 — Training curves

Left: training loss and validation loss. The dashed line marks the best checkpoint.

Right: the answer scores on the validation set after each epoch. Dotted lines are the scores before training.
''')

code(r'''
history = pd.DataFrame(state["history"])
fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
if "train_loss" in history:
    ax[0].plot(history["step"], history["train_loss"], "o-", label="train")
ax[0].plot(history["step"], history["val_loss"], "s-", label="validation")
ax[0].axvline(state["best_step"], ls="--", c="gray")
ax[0].set_yscale("log"); ax[0].set_xlabel("update"); ax[0].set_title("answer loss"); ax[0].legend()

if state["val_scores"]:
    scores = pd.DataFrame(state["val_scores"])
    for key in ["valid", "lighting_acc", "road_users_f1", "cameras_f1", "hazards_f1"]:
        line, = ax[1].plot(scores["epoch"], scores[key], "o-", label=key)
        ax[1].axhline(BASELINE["val"][key], ls=":", c=line.get_color())
    ax[1].set_xlabel("epoch"); ax[1].set_title("validation scores"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "curves.png", dpi=110); plt.show()
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 17 — Final test

We load the **best** adapter and score it on the test set.

The test set was never used for training or for choosing settings, so this is an honest result.

The table compares the model before and after training on the same test samples.
''')

code(r'''
load_adapter("best")
FINAL = {"best_step": state["best_step"], "best_val_loss": state["best_val"]}
FINAL["test"], _ = evaluate(SPLITS["test"], "after training, test", OUT / "final_test_answers.json")
if SPLITS["test_gold"]:
    FINAL["test_gold"], _ = evaluate(SPLITS["test_gold"], "after training, human-reviewed test", OUT / "final_gold_answers.json")
(OUT / "final_metrics.json").write_text(json.dumps(FINAL, indent=1))

keys = ["valid", "exact_match", "lighting_acc", "road_users_f1", "count_acc", "cameras_f1", "hazards_f1"]
print(pd.DataFrame({"before training": BASELINE["test"], "after training": FINAL["test"]}).loc[keys].to_string())
print(f"\nThe test set has only {FINAL['test']['true_hazards']} hazards, so hazards_f1 is a rough number.")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 18 — Try one example

We pick a test scene with a hazard, show the 6 cameras, and print what the model writes.
''')

code(r'''
record = next((r for r in SPLITS["test"] if r["target"]["events"]), SPLITS["test"][0])
start = time.time()
answer = predict(record)
seconds = time.time() - start

last_frame = {f["camera"]: f for f in record["frames"]}
fig, axes = plt.subplots(2, 3, figsize=(15, 5.5))
with zipfile.ZipFile(DATA / record["frames_zip"]) as z:
    for ax, cam in zip(axes.ravel(), CAMERA_NAMES):
        ax.imshow(Image.open(io.BytesIO(z.read(last_frame[cam]["path"]))))
        ax.set_title(CAMERA_TEXT[cam]); ax.axis("off")
plt.tight_layout(); plt.show()

print(f"Model answer ({seconds:.1f} s):\n{answer.removeprefix(CLOSED_THINKING).strip()}\n")
print(f"Correct answer:\n{render_answer(record['target'])}")
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 19 — Save the final adapter

The folder `final_adapter/` contains everything needed to use the trained model:

- the LoRA weights (about 65 MB),
- the prompt,
- the settings and the scores,
- a short README with loading instructions.

The base model is downloaded separately from Hugging Face. The adapter only stores the changes.

The adapter was trained on NVIDIA's dataset. Read the dataset and model licenses before sharing it.
''')

code(r'''
export = OUT / "final_adapter"
shutil.rmtree(export, ignore_errors=True)
shutil.copytree(OUT / "best", export)
(export / "prompt.txt").write_text(PROMPT, encoding="utf-8")
(export / "settings.json").write_text(json.dumps(asdict(CFG), indent=1))
(export / "metrics.json").write_text(json.dumps({"before_training": BASELINE, "after_training": FINAL}, indent=1))
(export / "README.md").write_text(f"""# DriveAnnotator LoRA adapter

Base model: {CFG.MODEL_ID} (revision {CFG.MODEL_REVISION})

```python
from alpamayo2_edge.checkpoint import load_model
from peft import PeftModel
model, _ = load_model("{CFG.MODEL_ID}", revision="{CFG.MODEL_REVISION}")
PeftModel.from_pretrained(model.vlm, "final_adapter")   # adds the adapter to the model
# Ask questions with the prompt in prompt.txt (see the training notebook, Steps 8 and 9).
```
""", encoding="utf-8")
print("Saved:", sorted(p.name for p in export.iterdir()))
''')

# ---------------------------------------------------------------------------
md(r'''
## Step 20 — Upload the adapter to Hugging Face (private)

This uploads `final_adapter/` to a **private** model repository on your Hugging Face account.

Only you can see a private repository.

### Why private?

- The adapter was trained on NVIDIA's dataset.
- That dataset's license limits sharing and "derivative works".
- A private repository keeps a safe backup without sharing anything.

### Before you run this cell

1. Create a token with **Write** access at https://huggingface.co/settings/tokens.
2. Add it in Colab Secrets as `HF_WRITE_TOKEN`.

### What gets uploaded

- the LoRA weights and their config,
- `prompt.txt`, `settings.json`, `metrics.json`,
- `README.md` (the model card),
- `LICENSE` (the base model's OpenMDW-1.1 license, which must travel with it) and `NOTICE.md`.
''')

code(r'''
from huggingface_hub import HfApi, hf_hub_download

REPO_NAME = "alpamayo2-edge-driveannotator-lora"   # change if you like
PRIVATE = True

write_token = userdata.get("HF_WRITE_TOKEN") if IN_COLAB else os.environ.get("HF_WRITE_TOKEN")
assert write_token, "Add a Write token as HF_WRITE_TOKEN in Colab Secrets."
hub = HfApi(token=write_token)
repo_id = f"{hub.whoami()['name']}/{REPO_NAME}"

# The base model's license must be included when its materials are redistributed
shutil.copyfile(hf_hub_download(CFG.MODEL_ID, "LICENSE", revision=CFG.MODEL_REVISION), export / "LICENSE")
(export / "NOTICE.md").write_text(
    "This adapter modifies nvidia/Alpamayo2-Edge, which is licensed under OpenMDW-1.1 (see LICENSE).\n"
    "It was trained on data from nvidia/PhysicalAI-Autonomous-Vehicles under the NVIDIA Autonomous Vehicle "
    "Dataset License Agreement. That dataset is not included here and may not be redistributed.\n",
    encoding="utf-8")

test = FINAL["test"]
(export / "README.md").write_text(f"""---
base_model: {CFG.MODEL_ID}
library_name: peft
license: other
tags: [lora, autonomous-driving, scene-annotation, alpamayo]
---

# DriveAnnotator LoRA for Alpamayo2-Edge

A LoRA adapter that makes `{CFG.MODEL_ID}` describe a driving scene from its 6 cameras in four lines:

```
lighting: daylight
road users: car four to seven, pedestrian one
cameras: front camera car, pedestrian; front right camera pedestrian
hazards: none
```

## Training

- Base model: `{CFG.MODEL_ID}` (revision `{CFG.MODEL_REVISION[:8]}`)
- Method: LoRA r={CFG.LORA_R}, alpha={CFG.LORA_ALPHA}, on the language model only (168 layers)
- Trainable parameters: {n_trainable / 1e6:.1f}M
- Epochs: {CFG.EPOCHS}, learning rate {CFG.LR}, effective batch {CFG.GRAD_ACCUM}
- Best validation loss: {state['best_val']:.3f} at update {state['best_step']}

## Results on the held-out test split ({test['n']} scenes)

| Score | Before training | After training |
|---|---|---|
""" + "".join(f"| {k} | {BASELINE['test'][k]} | {test[k]} |\n" for k in keys) + f"""
## How to use

```python
from alpamayo2_edge.checkpoint import load_model
from peft import PeftModel
model, _ = load_model("{CFG.MODEL_ID}", revision="{CFG.MODEL_REVISION}")
PeftModel.from_pretrained(model.vlm, "{repo_id}")
# Build the input with the prompt in prompt.txt (see 02_train_lora.ipynb, Steps 8 and 9).
```

## License

See `LICENSE` and `NOTICE.md`.
""", encoding="utf-8")

hub.create_repo(repo_id, repo_type="model", private=PRIVATE, exist_ok=True)
hub.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(export),
                  commit_message="Upload DriveAnnotator LoRA adapter")
print(f"Uploaded to https://huggingface.co/{repo_id} (private={PRIVATE})")
print("Done. In Colab: Runtime -> Disconnect and delete runtime.")
''')

# ---------------------------------------------------------------------------
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "A100"}},
      "cells": []}
for i, (kind, src) in enumerate(cells):
    cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"t{i:02d}"}
    if kind == "code":
        cell |= {"execution_count": None, "outputs": []}
    nb["cells"].append(cell)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({len(cells)} cells)")
