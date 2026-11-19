# DriveAnnotator

**Teach NVIDIA's self-driving model `Alpamayo2-Edge` to label driving scenes from its cameras.**

[![NVIDIA Alpamayo2-Edge](https://img.shields.io/badge/NVIDIA-Alpamayo2--Edge-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/nvidia/Alpamayo2-Edge)
[![NVIDIA PhysicalAI-AV](https://img.shields.io/badge/NVIDIA-PhysicalAI--AV%20dataset-76B900?logo=nvidia&logoColor=white)](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-5.17-FFD21E?logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-FFD21E?logo=huggingface&logoColor=black)](https://github.com/huggingface/peft)
[![Google Colab](https://img.shields.io/badge/Colab-A100-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Claude Code](https://img.shields.io/badge/Claude%20Code-Colab%20MCP-D97757?logo=anthropic&logoColor=white)](https://claude.com/claude-code)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

This README explains the whole project from start to finish.
You do not need to know anything about machine learning to follow it.

---

## 1. The idea in one minute

- Self-driving teams record a lot of video from cars.
- Before that video can be used, people must **label** it: "is it night?", "is there a pedestrian?", "is a car cutting in?".
- Labelling by hand is slow and expensive.
- This project trains a model that writes a **first draft** of the label by itself.
- A person then only checks and corrects it.

The model looks at 6 cameras around the car and writes four short lines:

```
lighting: daylight
road users: car four to seven, pedestrian one
cameras: front camera car, pedestrian; front right camera pedestrian
hazards: none
```

---

## 2. Words used in this README

| Word | Meaning |
|---|---|
| **Model** | A program that learned a skill from examples. Here: `nvidia/Alpamayo2-Edge` (2.65 billion parameters). |
| **Fine-tuning** | Teaching an existing model a new skill with a few extra examples. |
| **LoRA** | A cheap way to fine-tune. The original model stays frozen. We only train a small add-on (the **adapter**), here 16.5 million parameters (0.6% of the model). |
| **Dataset** | The examples the model learns from. Here: 1,412 driving scenes. |
| **Train / val / test** | Three separate parts of the dataset. The model learns from **train**, we pick the best version with **val**, and we report the final score on **test**, which it never saw. |
| **Loss** | A number that says how wrong the model is. Lower is better. |
| **Epoch** | One full pass over the training data. |
| **F1 score** | A score from 0 to 1. It is high only when the model finds the right things **and** does not invent wrong ones. |

---

## 3. The workflow

```
NVIDIA dataset on Hugging Face (videos + 3D boxes)
        │
        ▼
01_build_dataset.ipynb   (CPU, ~1.5 h)
  • pick 300 clips
  • take 5 moments per clip
  • save 24 camera images per moment
  • make the label automatically from the 3D boxes
        │
        ▼
Google Drive: dataset_v1/ (2.3 GB)
        │
        ▼
02_train_lora.ipynb      (Colab A100 GPU, ~1.5 h)
  • score the model before training
  • train the LoRA adapter
  • score the model after training on the test split
        │
        ▼
LoRA adapter (~65 MB) → private Hugging Face repository
```

---

## 4. Step 1 — The dataset

### Where the data comes from

- **Source:** `nvidia/PhysicalAI-Autonomous-Vehicles` on Hugging Face (version `14ed7a77`).
- It has real driving clips from 25 countries, recorded with 7 cameras, LiDAR and radar.
- It also has **3D boxes** around every car, truck, person and so on.

### How one sample is made

1. Pick a clip (20 seconds long).
2. Pick a moment in it (at 2, 6, 10, 14 and 18 seconds).
3. Take 4 frames from each of 6 cameras: 0.3, 0.2 and 0.1 seconds before the moment, and the moment itself.
   This is exactly the input `Alpamayo2-Edge` was built for.
4. Project the 3D boxes into each camera image to see which objects are really visible.
5. Turn that into a label with simple rules (below).

### What the label contains

| Field | Values | How it is decided |
|---|---|---|
| `lighting` | daylight, twilight, night | From the height of the sun (country, month, local hour). Sun above +6° = daylight, below −6° = night. |
| `road users` | car, truck, bus, tram, other vehicle, pedestrian, rider, animal, each with a count group (one, two to three, four to seven, eight or more) | Objects visible in at least one camera, at least 18 pixels tall, and not too far away. |
| `cameras` | for each camera, the classes it sees | Same visibility test, per camera. |
| `hazards` | pedestrian in path, vehicle cutting in, lead vehicle close, vulnerable road user near ego | Distance and movement rules (for example: a pedestrian inside our lane up to 25 m ahead). |

### Dataset size

| Split | Scenes | Clips | Daylight / twilight / night | Scenes with nobody visible |
|---|---|---|---|---|
| train | 1,137 | 239 | 607 / 87 / 443 | 8% |
| val | 137 | 30 | 86 / 26 / 25 | 15% |
| test | 138 | 29 | 59 / 6 / 73 | 18% |

- Clips from the same drive are always in the same split, so the test scenes are really new.
- How often each class appears in train: car 88%, pedestrian 30%, truck 27%, rider 8%, bus 5%.
- Hazards are rare: in train, "pedestrian in path" appears 13 times, "vehicle cutting in" 8 times.

### Problems we found and fixed while building it

| Problem | What we saw | Fix |
|---|---|---|
| Boxes only for one object | Each object in the box file has its own timestamps | Pick the nearest row **per object** |
| Wrong coordinate frame | Guessing the frame gave different answers per clip | Read the frame from the file's own `reference_frame` column |
| Wrong lighting labels | Overcast days were "low light", lit city nights were "day" | Use the sun's height instead of image brightness |
| Two processes building the same clip | Leftover temporary files | A lock file per clip |

---

## 5. Step 2 — Fine-tuning

### Why LoRA (and not other methods)

| Method | GPU memory needed | Decision |
|---|---|---|
| Full fine-tuning | about 42 GB before even counting images | Does not fit on a 40 GB A100, and would overfit 1,100 scenes |
| **LoRA (used)** | about 7.5 GB measured | Fits easily, fast, and the original model stays untouched |
| QLoRA (4-bit) | a bit less | Not needed here (memory is not a problem) and 20–40% slower |

LoRA is added only to the **language part** of the model (28 layers × 6 layer types = 168 places).
The vision part, which reads the images, stays frozen.

### The most important finding: answer in English, not JSON

Our first plan was to make the model answer in JSON (`{"lighting":"daylight",...}`).
A safety check (memorize 8 samples) failed: the loss got stuck at 2.26.

We looked at which words were hard for the model.
They were JSON symbols like `"],` and `":"`. The model had almost never seen them, because it was trained to write driving explanations in English.

| Answer format | Tokens per answer | Rare tokens | Loss after 40 steps on 8 samples |
|---|---|---|---|
| JSON | 70 | 23% | 2.26 |
| **Plain English lines (used)** | 48 | 2% | 0.25 |

So the model now writes four plain English lines.
The notebook turns them back into structured data for scoring.

### Training settings and why

| Setting | Value | Why |
|---|---|---|
| LoRA rank / alpha | 16 / 32 | Enough to learn a new answer style; small enough not to overfit |
| Learning rate | 1e-4, warm-up then cosine decay | Standard for LoRA; the warm-up avoids an unstable start |
| Batch | 1 sample, update every 8 | Each sample is ~4,900 tokens (24 images), so one at a time |
| Epochs | 3, keep the best by validation loss | The best checkpoint, not the last one, is used |
| Hazard scenes | shown 3× per epoch | They are rare |
| Loss | cross-entropy on the ~50 answer tokens only | The prompt and images are not something to predict |

---

## 6. Results

The model was trained for 3 epochs (483 updates, about 1.3 hours on one A100, about 17 compute units).

How the run went, in words:

- **Before training** the model scored 0 on every score: it did not know the four-line format at all.
- **After the first epoch** almost every answer already had the right format, and most road users and cameras were right.
- **Epochs 2 and 3** kept improving every score a little. The validation loss went down the whole time, so the model did not overfit.
- **Hazards** stayed the weakest part, because they are rare in the data (see "Known limits").

### Where are the numbers?

The exact scores are in **`RESULTS.md`**, which is **not** in the public repository.

Why: the NVIDIA dataset license (§3) treats benchmark and performance results on the dataset as confidential.

If you rebuild the dataset and train with your own access, the notebook prints the same tables for you (Steps 15–17).

---

## 7. How to run it yourself

### What you need

- A Hugging Face account with access to `nvidia/PhysicalAI-Autonomous-Vehicles` and `nvidia/Alpamayo2-Edge`.
- Google Colab with compute units (an A100 GPU for about 1.5 hours ≈ 16–20 units).

### Build the dataset (local computer or Colab CPU)

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
copy .env.example .env          then put your Hugging Face Read token in .env
jupyter lab                     then run 01_build_dataset.ipynb from top to bottom
```

### Train (Colab, A100)

1. Upload `data/dataset_v1/` to Google Drive at `MyDrive/drive_annotator/dataset_v1`.
2. Open `02_train_lora.ipynb` in Colab and choose **Runtime → Change runtime type → A100 GPU**.
3. In the Secrets panel (key icon on the left) add `HF_TOKEN` (Read token). For the upload step also add `HF_WRITE_TOKEN` (Write token).
4. Run the cells from top to bottom.
5. When finished: **Runtime → Disconnect and delete runtime**.

### Optional: let Claude Code run the Colab notebook for you

This project includes `.mcp.json`, which connects Claude Code to a Colab tab through Google's Colab MCP server.
Claude can then add and run cells and read their output.

---

## 8. Files

| File | What it is |
|---|---|
| `01_build_dataset.ipynb` | Builds the dataset (17 steps) |
| `02_train_lora.ipynb` | Trains, scores and uploads the adapter (20 steps) |
| `requirements.txt` | Libraries for building the dataset |
| `.env.example` | Template for your tokens (copy to `.env`; never share `.env`) |
| `.mcp.json` | Colab MCP connection for Claude Code |
| `data/dataset_v1/` | The built dataset (private, not in any upload) |
| `RESULTS.md` | Exact scores of the trained model (private, not in the public repository) |
| `dev/build_notebook.py`, `dev/build_train_notebook.py` | Generate the two notebooks |
| `dev/test_notebook.py`, `dev/test_train_notebook.py`, `dev/dataset_label_tests.py` | Tests that run on a laptop without a GPU |
| `dev/run_notebook.py` | Runs a notebook from the command line |
| `dev/upload_recipe.py`, `dev/recipe_README.md` | Upload the code-only recipe to Hugging Face |

To change a notebook, edit its generator in `dev/`, then run:

```
python dev/build_notebook.py 01_build_dataset.ipynb
python dev/test_notebook.py 01_build_dataset.ipynb
python dev/build_train_notebook.py 02_train_lora.ipynb
python dev/test_train_notebook.py 02_train_lora.ipynb
```

---

## 9. What is uploaded to Hugging Face

| What | Where | Visibility | Why |
|---|---|---|---|
| Code, notebooks (no outputs), README | GitHub: `navneetsxngh/driveannotator` | Public | Contains no NVIDIA data and no scores |
| LoRA adapter + model card | `<your-username>/alpamayo2-edge-driveannotator-lora` | **Private** | Trained on NVIDIA's dataset, whose license limits sharing |
| Code only (notebooks + `dev/`) | `<your-username>/driveannotator-recipe` | Public | Contains no NVIDIA data, so others can rebuild with their own access |
| The dataset itself | — | **Not uploaded** | The NVIDIA license (§4.6) does not allow hosting or sharing it |

---

## 10. License notes (please read)

- **Dataset** (NVIDIA Autonomous Vehicle Dataset License):
  - use only for internal development of driving systems,
  - do not share or host the data or anything derived from it,
  - results and performance data about the dataset are treated as confidential,
  - the license ends 12 months after download; then the copies must be deleted.
- **Model** (`Alpamayo2-Edge`, OpenMDW-1.1): free to use and modify; keep the license and notices when sharing.
- **This project's code**: Apache-2.0.

---

## 11. Known limits and next steps

- Hazards are rare in the data, so the model learns them least. Next: pick more clips with hazards.
- Visibility does not check if an object is hidden behind another one.
- Lighting uses an approximate sun position; it can be wrong near sunrise and sunset.
- Test labels are automatic. Reviewing them by hand (dataset notebook, Steps 15–16) gives a more trustworthy score.
