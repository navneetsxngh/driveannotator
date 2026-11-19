# Headless runner: executes the notebook's code cells in order in one process (cwd = project root,
# so .env and ./data are used). Skips the interactive review cells. Logs go to stdout.
#   python dev/run_notebook.py 01_build_dataset.ipynb [--skip 15,16] [--stop-after 11b]
import argparse, json, os, re, sys, time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ap = argparse.ArgumentParser()
ap.add_argument("notebook")
ap.add_argument("--skip", default="15,16")
ap.add_argument("--stop-after", default=None)
args = ap.parse_args()

skip = set(filter(None, args.skip.split(",")))
cells = []
for c in json.loads(Path(args.notebook).read_text(encoding="utf-8"))["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    m = re.match(r"# ==== Cell (\w+):", src)
    cells.append((m.group(1) if m else "?", src))

G = {"__name__": "__main__"}
for cid, src in cells:
    if cid in skip:
        print(f"\n######## skip cell {cid} ########", flush=True)
        continue
    print(f"\n######## cell {cid} ########", flush=True)
    t = time.time()
    exec(compile(src, f"<cell {cid}>", "exec"), G)
    print(f"######## cell {cid} done in {time.time() - t:.1f}s ########", flush=True)
    if args.stop_after and cid == args.stop_after:
        break
print("\nRUNNER FINISHED", flush=True)
