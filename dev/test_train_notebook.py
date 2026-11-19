# Local CPU tests for 02_train_lora.ipynb (no model weights needed).
# The notebook itself has no test cells; all checks live here.
# usage (from project root): python dev/test_train_notebook.py 02_train_lora.ipynb
import json, os, re, sys, tempfile, types
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
NB = Path(sys.argv[1]).resolve()
CODE = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}


def cell_with(marker):
    hits = [s for s in CODE if marker in s]
    assert len(hits) == 1, f"{len(hits)} cells contain {marker!r}"
    return hits[0]


def run(marker, patch=()):
    src = cell_with(marker)
    for a, b in patch:
        assert a in src, (marker, a)
        src = src.replace(a, b)
    print(f"\n######## {marker} ########", flush=True)
    exec(compile(src, marker, "exec"), G)


run("IN_COLAB = ")                                   # install + imports (local: no installs)
torch, np = G["torch"], G["np"]
run("class Settings:")
CFG = G["CFG"]
CFG.DRIVE_DATA_ROOT = CFG.LOCAL_DATA_ROOT = str(Path.cwd() / "data" / "dataset_v1")
CFG.OUT_DIR = tempfile.mkdtemp(prefix="traintest_")
CFG.NUM_WORKERS = 0
run("def read_jsonl(")                               # load manifests

# ---- replaces the HF-login and model-loading cells: real tokenizer/processor/config, no weights ----
from huggingface_hub import snapshot_download
from alpamayo2_edge.config import Alpamayo2EdgeConfig, normalize_checkpoint_config, build_alpamayo2_edge_tokenizer
from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge
path = Path(snapshot_download(CFG.MODEL_ID, revision=CFG.MODEL_REVISION, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"]))
mcfg = Alpamayo2EdgeConfig(**normalize_checkpoint_config(json.loads((path / "config.json").read_text())))
mcfg._name_or_path = str(path); mcfg.set_dtype("bfloat16")
tok = build_alpamayo2_edge_tokenizer(str(path), mcfg.history_vocab_size, mcfg.future_vocab_size)
with torch.device("meta"):
    meta_model = Alpamayo2Edge(mcfg)
meta_model.requires_grad_(False)
G.update(MODEL_CFG=mcfg, TOK=tok, PROC=G["get_processor"](tok, mcfg), EOT="<|im_end|>", DEVICE="cpu",
         model=meta_model, n_params=sum(p.numel() for p in meta_model.parameters()))

# ---- answer format ----
run("def render_answer(")
render, parse = G["render_answer"], G["parse_answer"]
n = 0
for split in ["train", "val", "test"]:
    for r in G["SPLITS"][split]:
        text = render(r["target"])
        got, err = parse(text)
        assert err is None and got == r["target"], (r["sample_id"], text, err, got)
        assert got == parse(G["CLOSED_THINKING"] + text + "\n")[0], "think prefix / trailing newline must be ignored"
        n += 1
print(f"render -> parse round trip OK on {n} records")
for bad in ["", "sorry", "lighting: sunny\nroad users: none\ncameras: none\nhazards: none",
            "lighting: night\nroad users: car seven\ncameras: none\nhazards: none",
            "lighting: night\nroad users: none\ncameras: front camera car\nhazards: none",
            "lighting: night\nroad users: car one\ncameras: roof camera car\nhazards: none",
            "lighting: night\nroad users: car one\ncameras: none\nhazards: pedestrian in path on front camera"]:
    assert parse(bad)[0] is None, bad
# "front camera" must not swallow "front telephoto camera" / "front left camera"
t = {"lighting": "day" "light", "road_users": {"car": "1"},
     "cameras": {"cross_left": ["car"], "front_wide": ["car"], "front_tele": ["car"]}, "events": []}
assert parse(render(t))[0] == t
print("parser rejects bad answers OK")

# ---- encoding: answer tokens are exactly the tail of the sequence ----
run("def encode(")
for r in G["SPLITS"]["train"][:3]:
    enc = G["encode"](r)
    ans = tok(G["CLOSED_THINKING"] + render(r["target"]) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert enc["input_ids"][0, -len(ans):].tolist() == ans and enc["n_answer"] == len(ans)
    assert G["encode"](r, with_answer=False)["input_ids"].shape[1] == enc["input_ids"].shape[1] - len(ans)
print("answer tokens sit at the end of the sequence OK")

# ---- scoring ----
run("def score(")
T = G["SPLITS"]["train"][1]["target"]
s = G["score"]([T], [T]); assert s["exact_match"] == 1 and s["road_users_f1"] == 1
s = G["score"]([None], [T]); assert s["valid"] == 0 and s["road_users_f1"] == 0 and s["lighting_acc"] == 0
P = {**T, "road_users": {"truck": "1"}, "cameras": {"front_wide": ["truck"]}}
assert G["score"]([P], [T])["road_users_f1"] == 0
print("scoring OK")

# ---- LoRA placement (real architecture on the meta device) ----
run("def attach_lora(")
assert all(p.dtype == torch.float32 for _, p in G["TRAINABLE"])
G["attach_lora"]()                                   # re-attach must work too
print("LoRA placement OK")

# ---- loss alignment against a reference masked cross-entropy ----
run("def answer_loss(")
V, H, L, k = 50, 16, 12, 4
emb, head = torch.nn.Embedding(V, H), torch.nn.Linear(H, V)
class Inner(torch.nn.Module):
    def forward(self, input_ids=None, **kw):
        return types.SimpleNamespace(last_hidden_state=emb(input_ids))
real_model = G["model"]
G["model"] = types.SimpleNamespace(vlm=types.SimpleNamespace(model=Inner(), lm_head=head))
ids = torch.randint(0, V, (1, L))
enc = {key: ids for key in G["MODEL_INPUT_KEYS"]} | {"n_answer": k}
got = G["answer_loss"](enc)
labels = ids.clone(); labels[:, :L - k] = -100
ref = torch.nn.functional.cross_entropy(head(emb(ids))[:, :-1].float().reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
assert torch.allclose(got, ref, atol=2e-2), (got, ref)
G["model"] = real_model
print(f"loss alignment OK ({got.item():.4f} vs {ref.item():.4f})")

# ---- epoch order and learning-rate schedule (functions only) ----
for fn in ["epoch_order", "lr_factor"]:
    src = cell_with(f"def {fn}(")
    exec(re.search(rf"(def {fn}\(.*?)(?=\n\n\n|\nSAMPLES_PER_EPOCH)", src, re.S).group(1), G)
order = G["epoch_order"](0)
n_ev = sum(bool(r["target"]["events"]) for r in G["SPLITS"]["train"])
assert len(order) == len(G["SPLITS"]["train"]) + n_ev * (CFG.EVENT_OVERSAMPLE - 1) and order == G["epoch_order"](0)
G["TOTAL_UPDATES"] = 480
lrs = [G["lr_factor"](s) for s in range(480)]
warm = int(480 * CFG.WARMUP_RATIO)
assert abs(lrs[warm - 1] - 1) < 1e-9 and abs(lrs[-1] - CFG.MIN_LR_RATIO) < 0.01
assert all(a >= b - 1e-12 for a, b in zip(lrs[warm:], lrs[warm + 1:]))
print(f"epoch order + LR schedule OK ({len(order)} samples/epoch)")

# ---- notebook text is English-only (ASCII letters) and has a markdown before every code cell ----
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        assert i > 0 and cells[i - 1]["cell_type"] == "markdown", f"code cell {i} has no markdown before it"
hinglish = re.compile(r"\b(hai|karo|nahi|kyu|chahiye|wala|matlab|toh|agar|hoga)\b", re.I)
for c in cells:
    hit = hinglish.search("".join(c["source"]))
    assert not hit, f"non-English word {hit.group(0)!r} in: {''.join(c['source'])[:80]}"
print("English-only text + markdown before every code cell OK")
print("\nALL TRAIN-NOTEBOOK TESTS PASSED")
