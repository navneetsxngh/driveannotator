# Upload the code-only recipe (no NVIDIA data) to Hugging Face.
# usage (from the project root):
#   python dev/upload_recipe.py            -> shows what would be uploaded (dry run)
#   python dev/upload_recipe.py --upload   -> uploads
# Needs HF_WRITE_TOKEN in .env (a token with Write access).
import os, sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_NAME = "driveannotator-recipe"
REPO_TYPE = "dataset"          # holds the dataset-building recipe; contains no data
PRIVATE = False

ROOT = Path(__file__).resolve().parent.parent
FILES = {                       # local path -> path in the repository
    "dev/recipe_README.md": "README.md",
    "01_build_dataset.ipynb": "01_build_dataset.ipynb",
    "02_train_lora.ipynb": "02_train_lora.ipynb",
    "requirements.txt": "requirements.txt",
    ".env.example": ".env.example",
    "dev/build_notebook.py": "dev/build_notebook.py",
    "dev/build_train_notebook.py": "dev/build_train_notebook.py",
    "dev/test_notebook.py": "dev/test_notebook.py",
    "dev/test_train_notebook.py": "dev/test_train_notebook.py",
    "dev/dataset_label_tests.py": "dev/dataset_label_tests.py",
    "dev/run_notebook.py": "dev/run_notebook.py",
}


def check_no_data(path):
    """Refuse to upload notebooks that still contain outputs (they could show NVIDIA images or labels)."""
    if path.suffix == ".ipynb":
        import json
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        assert not any(c.get("outputs") for c in cells if c["cell_type"] == "code"), f"{path} has outputs; clear them first"
    text = path.read_text(encoding="utf-8", errors="ignore")
    assert "hf_" not in text or "hf_xxx" in text or path.name.endswith(".py"), f"{path} may contain a token"


load_dotenv(ROOT / ".env")
for local in FILES:
    check_no_data(ROOT / local)

token = os.environ.get("HF_WRITE_TOKEN")
assert token, "Add HF_WRITE_TOKEN (a Write token) to .env"
api = HfApi(token=token)
repo_id = f"{api.whoami()['name']}/{REPO_NAME}"

print(f"Repository: https://huggingface.co/datasets/{repo_id} ({'private' if PRIVATE else 'public'})")
for local, remote in FILES.items():
    print(f"  {local:32s} -> {remote}")

if "--upload" not in sys.argv:
    print("\nDry run only. Run again with --upload to upload.")
    sys.exit(0)

api.create_repo(repo_id, repo_type=REPO_TYPE, private=PRIVATE, exist_ok=True)
for local, remote in FILES.items():
    api.upload_file(path_or_fileobj=str(ROOT / local), path_in_repo=remote, repo_id=repo_id, repo_type=REPO_TYPE,
                    commit_message=f"Add {remote}")
print(f"\nUploaded to https://huggingface.co/datasets/{repo_id}")
