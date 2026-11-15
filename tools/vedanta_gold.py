"""Hand labels (data/vedanta_labels.json) -> benchmark/compounds_vedanta.jsonl.gz.

Boundaries come from the typed real words via uoh_parse.gold, so a vowel fused into the
letter before a cut is accepted on either side, as in the DCS benchmark. Words where that
alignment disagrees with the cuts the labeler confirmed keep the confirmed cuts only.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from uoh_parse import gold

labels = json.load(open("data/vedanta_labels.json", encoding="utf-8"))
work = {r["word"]: r["work"] for r in csv.DictReader(open("data/vedanta_to_label.csv", encoding="utf-8"))}
items, fallback = [], 0
for w, v in labels.items():
    if v["status"] != "checked" or not v["cuts"]:
        continue
    targets, cands = gold(w, v["parts"].split("+"))
    if targets == v["cuts"]:
        boundaries = cands
    else:
        boundaries = [[c] for c in v["cuts"]]
        fallback += 1
    items.append({"id": len(items), "word": w, "parts": v["parts"], "work": work[w], "boundaries": boundaries})
for path in ("data/vedanta_gold.jsonl.gz", "benchmark/compounds_vedanta.jsonl.gz"):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
open("data/vedanta_gold.jsonl", "w", encoding="utf-8").write("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
print(len(items), "words;", fallback, "kept confirmed cuts only")
