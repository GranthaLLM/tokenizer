import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "MuseMesh/sansar-sanskrit-corpus"
CONFIGS = ["cc0-1.0", "cc-by-4.0", "odc-by", "mit", "apache-2.0"]
OUT = Path("data/raw/sansar")

api = HfApi()
info = api.dataset_info(REPO)
files = [s.rfilename for s in info.siblings]
print("revision", info.sha)
print("\n".join(files))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "REVISION").write_text(info.sha)
patterns = [f"data/{c}/*" for c in CONFIGS] + ["README.md", "*.json", "*.md", "licenses/*"]
snapshot_download(REPO, repo_type="dataset", local_dir=OUT, allow_patterns=patterns)
