import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import compound_eval2 as ce
from tagged import TaggedTok
from tagger import Tagger

name = sys.argv[1]
mapping = f"data/{name}_map.json" if Path(f"data/{name}_map.json").exists() else None
import os
import pickle

TG = os.environ.get("TAGGER", "1")
tg = Tagger() if TG == "1" else Tagger(f"data/tagger{TG[0]}.pt", f"data/tagger{TG[0]}_vocab.json", hid=192)
if TG.endswith("h"):
    tg.override = {w: list(c) for w, c in pickle.load(open("data/teacher_cuts.pkl", "rb")).items()}
tk = TaggedTok(f"data/tok_{name}.model", tg, mapping)
p, r, f1, over = ce.score(tk.cuts)
import pickle
from eval_taggers import nyaya_test, score as tscore

random.seed(2)
uoh_g = random.sample(pickle.load(open("data/uoh_gold_test.pkl", "rb")), 8000)
uoh_f1 = tscore(type("T", (), {"cuts": staticmethod(tk.cuts)}), uoh_g)[2]
ny_f1 = tscore(type("T", (), {"cuts": staticmethod(tk.cuts)}), nyaya_test())[2]
F = Path("data/flores/flores200_dataset/devtest")
fl = {l: (F / f"{l}.devtest").read_text().splitlines() for l in ("san_Deva", "eng_Latn")}


def tpw(lines):
    return sum(tk.count(x) for x in lines) / sum(len(x.split()) for x in lines)


random.seed(7)
t = pq.read_table("data/pool/classical_test.parquet").to_pydict()
by = defaultdict(list)
for x, s in zip(t["text"], t["source"]):
    by[s].append(x)
cl = {s: random.sample(v, min(20000, len(v))) for s, v in by.items()}
print(f"| {name} | {p:.3f} | {r:.3f} | {f1:.3f} | {uoh_f1:.3f} | {ny_f1:.3f} | {tpw(fl['san_Deva']):.2f} | {tpw(fl['eng_Latn']):.2f} | {tpw(cl['dcs']):.2f} | {tpw(cl['gretil']):.2f} |")
