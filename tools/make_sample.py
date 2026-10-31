import random

import pyarrow.parquet as pq

CAP = {"sangraha": 700_000, "indiccorp_v2": 400_000}
random.seed(0)
t = pq.read_table("data/pool/train.parquet").to_pydict()
by = {}
for x, s in zip(t["text"], t["source"]):
    by.setdefault(s, []).append(x)
out = []
for s, xs in by.items():
    random.shuffle(xs)
    out += xs[: CAP.get(s, len(xs))]
random.shuffle(out)
open("data/pool/sample.txt", "w").write("\n".join(out) + "\n")
print(len(out), sum(len(x) for x in out) / 1e6, "Mchars")
