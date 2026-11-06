import gzip
import json
import pickle
import shutil
from pathlib import Path

import pyarrow.parquet as pq

OUT = Path("benchmark")
gold = pickle.load(open("data/dcs_gold_v2_test.pkl", "rb"))
unsplit = pickle.load(open("data/dcs_unsplit_test.pkl", "rb"))
with gzip.open(OUT / "compounds_test.jsonl.gz", "wt", encoding="utf-8") as f:
    for i, (w, _targets, cands) in enumerate(gold):
        f.write(json.dumps({"id": i, "word": w, "boundaries": cands}, ensure_ascii=False) + "\n")
with gzip.open(OUT / "unsplit_controls.txt.gz", "wt", encoding="utf-8") as f:
    f.write("\n".join(unsplit) + "\n")
t = pq.read_table("data/pool/classical_test.parquet")
pq.write_table(t, OUT / "heldout_text.parquet", compression="zstd")
works = sorted(set(zip(t["source"].to_pylist(), t["work"].to_pylist())))
json.dump([{"source": s, "work": w} for s, w in works], open(OUT / "heldout_works.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
shutil.copy("data/vedanta_to_label.csv", OUT / "vedanta_to_label.csv")
print(len(gold), "compounds,", len(unsplit), "controls,", len(works), "held-out works")
