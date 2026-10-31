from huggingface_hub import hf_hub_download

for repo, f in [
    ("wikimedia/wikipedia", "20231101.en/train-00000-of-00041.parquet"),
    ("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet"),
]:
    print(hf_hub_download(repo, f, repo_type="dataset", local_dir="data/raw/english"))
