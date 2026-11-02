import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import compound_eval as ce
from cbpe import CBPE

name = sys.argv[1]
tk = CBPE(f"data/tok_{name}.model", f"data/{name}_map.json")
p, r, f1 = ce.score(tk.cuts, 0)
f1b = ce.score(tk.cuts, 1)[2]
F = Path("data/flores/flores200_dataset/devtest")
fl = {l: (F / f"{l}.devtest").read_text().splitlines() for l in ("san_Deva", "eng_Latn")}


def tpw(lines):
    return sum(tk.count(x) for x in lines) / sum(len(x.split()) for x in lines)


random.seed(7)
t = pq.read_table("data/pool/classical_test.parquet").to_pydict()
by = defaultdict(list)
for x, s in zip(t["text"], t["source"]):
    by[s].append(x)
cl = {s: random.sample(v, min(50000, len(v))) for s, v in by.items()}
print(f"| {name} | {p:.3f} | {r:.3f} | {f1:.3f} | {f1b:.3f} | {tpw(fl['san_Deva']):.2f} | {tpw(fl['eng_Latn']):.2f} | {tpw(cl['dcs']):.2f} | {tpw(cl['gretil']):.2f} |")
