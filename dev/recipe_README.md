---
license: apache-2.0
tags: [autonomous-driving, alpamayo, lora, recipe]
---

# DriveAnnotator — recipe (code only, no data)

This repository contains **only code**.

It does **not** contain any images, labels, clip IDs or other data from NVIDIA's dataset.

With your own access to NVIDIA's dataset, these notebooks rebuild the dataset and train the adapter yourself.

## What the project does

- It takes driving scenes from 6 car cameras.
- It fine-tunes NVIDIA's `Alpamayo2-Edge` (2.65B parameters) with a small **LoRA** adapter.
- The trained model describes each scene in four lines:

```
lighting: daylight
road users: car four to seven, pedestrian one
cameras: front camera car, pedestrian; front right camera pedestrian
hazards: none
```

## Why it is useful

- Labelling driving data by hand is slow and expensive.
- The model writes a first draft of the label. A person only checks and corrects it.
- It needs only cameras, no LiDAR.

## What you need

1. A Hugging Face account with access to:
   - `nvidia/PhysicalAI-Autonomous-Vehicles` (accept the license on the dataset page),
   - `nvidia/Alpamayo2-Edge`.
2. Google Colab with an **A100** GPU for training (about 1.5 hours).
3. A computer or a Colab CPU runtime for building the dataset (about 1.5 hours, about 27 GB streamed).

## How to use it

### Step 1 — Build the dataset (`01_build_dataset.ipynb`)

- Run it on a CPU runtime (or locally after `pip install -r requirements.txt`).
- It picks 300 clips, takes 5 moments from each, saves 24 camera images per moment, and makes the labels automatically from NVIDIA's 3D boxes.
- Keep the result private (for example in your own Google Drive). NVIDIA's license does not allow sharing it.

### Step 2 — Train (`02_train_lora.ipynb`)

- Upload the dataset folder to Google Drive.
- Open the notebook in Colab with an A100 GPU and run it from top to bottom.
- It scores the model before and after training on a held-out test split.

## Files

| File | What it is |
|---|---|
| `01_build_dataset.ipynb` | Builds the dataset from NVIDIA's data |
| `02_train_lora.ipynb` | Trains and evaluates the LoRA adapter |
| `requirements.txt` | Libraries for building the dataset |
| `.env.example` | Template for your Hugging Face token (never commit the real `.env`) |
| `dev/` | Notebook generators and tests (they run on a laptop without a GPU) |

## License

- This code: Apache-2.0.
- NVIDIA's dataset and model have their own licenses. Read them before you use the data or share anything trained on it.
