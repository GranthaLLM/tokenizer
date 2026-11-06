import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import compound_eval2 as ce
from cbpe import apply_map
from tagged import B, RUN, TaggedTok
from tagger import Tagger

tg = Tagger("data/tagger6.pt", "data/tagger6_vocab.json", hid=256)
tg.override = {w: list(c) for w, c in pickle.load(open("data/teacher_cuts.pkl", "rb")).items()}
tk = TaggedTok("data/tok_v10_tagak_128.model", tg, "data/v10_tagak_128_map.json")
mp, inv = tk.mp, tk.inv


def plain_bounds(run):
    pos, out = 0, set()
    for p in tk.sp.encode(apply_map(run, mp), out_type=str):
        pos += len("".join(inv.get(c, c) for c in p).replace("▁", ""))
        out.add(pos)
    return out


def prep(text, mode):
    runs = [m.group() for m in RUN.finditer(text)]
    cuts = dict(zip(runs, tg.cuts_many(runs)))
    marked = 0

    def sub(m):
        nonlocal marked
        w, cs = m.group(), cuts[m.group()]
        if not cs:
            return w
        if mode == "selective" and set(cs) <= plain_bounds(w):
            return w
        marked += 1
        out, last = [], 0
        for c in cs:
            out.append(w[last:c])
            last = c
        out.append(w[last:])
        return B.join(out)

    t = RUN.sub(sub, text)
    return (apply_map(t, mp) if mp else t), marked, len(runs)


def count(text, mode):
    t, marked, n = prep(text, mode)
    return sum(1 for i in tk.sp.encode(t) if i != tk.bid), marked, n


def cuts_fn(mode):
    def f(w):
        t, _, _ = prep(w, mode)
        out, pos = set(), 0
        for p in tk.sp.encode(t, out_type=str):
            if p.replace("▁", "") == B:
                continue
            pos += len("".join(inv.get(c, c) for c in p).replace(B, "").replace("▁", ""))
            if 0 < pos < len(w):
                out.add(pos)
        return out

    return f


F = Path("data/flores/flores200_dataset/devtest")
fl = {l: (F / f"{l}.devtest").read_text().splitlines() for l in ("san_Deva", "eng_Latn")}
random.seed(7)
t = pq.read_table("data/pool/classical_test.parquet").to_pydict()
by = defaultdict(list)
for x, s in zip(t["text"], t["source"]):
    by[s].append(x)
cl = {s: random.sample(v, min(10000, len(v))) for s, v in by.items()}
print("| policy | cuts right | boundaries respected | F1 | FLORES san | DCS | GRETIL | words marked |")
print("|---|---|---|---|---|---|---|---|")
for mode in ("always", "selective"):
    p, r, f1, _ = ce.score(cuts_fn(mode))

    def tpw(lines):
        tot = marked = nw = words = 0
        for x in lines:
            c, m, n = count(x, mode)
            tot += c
            marked += m
            nw += n
            words += len(x.split())
        return tot / words, marked, nw

    a = tpw(fl["san_Deva"])
    b = tpw(cl["dcs"])
    c = tpw(cl["gretil"])
    mk = (a[1] + b[1] + c[1]) / (a[2] + b[2] + c[2])
    print(f"| {mode} | {p:.3f} | {r:.3f} | {f1:.3f} | {a[0]:.2f} | {b[0]:.2f} | {c[0]:.2f} | {mk:.1%} |", flush=True)
