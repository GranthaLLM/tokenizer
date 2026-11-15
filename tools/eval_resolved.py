import pickle
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from cbpe import apply_map
from resolved import J, RUN, RUN2
from sandhi import Joiner, resolve

labels = {}
for p in ("data/teacher_labels.pkl", "data/resolve_labels.pkl"):
    labels.update(pickle.load(open(p, "rb")))
joiner = Joiner.load("data/sandhi_join.json")
sp = spm.SentencePieceProcessor(model_file="data/tok_res_128.model")
plain = spm.SentencePieceProcessor(model_file="release/grantha-128k.model")
cache = {}


def resolved(run):
    if run not in cache:
        o = labels.get(run)
        cache[run] = resolve(joiner, run, o) if o and len(run) > 3 else None
    return cache[run]


def enc(text):
    hit = tot = 0

    def sub(m):
        nonlocal hit, tot
        tot += 1
        w = resolved(m.group())
        if w:
            hit += 1
            return J.join(w)
        return m.group()

    t = RUN.sub(sub, text)
    return sp.encode(t, out_type=str), hit, tot


def dec(pieces):
    s = "".join(pieces).replace("▁", " ")
    return RUN2.sub(lambda m: joiner.join_deva(m.group().split(J)), s)


F = Path("data/flores/flores200_dataset/devtest/san_Deva.devtest")
random.seed(7)
t = pq.read_table("data/pool/classical_test.parquet").to_pydict()
by = {}
for x, s in zip(t["text"], t["source"]):
    by.setdefault(s, []).append(x)
sets = {"FLORES san": F.read_text().splitlines(), "DCS held-out": random.sample(by["dcs"], 4000), "GRETIL held-out": random.sample(by["gretil"], 4000)}
print("| set | tokens per word, plain | tokens per word, resolved | word occurrences resolved | exact round trip |")
print("|---|---|---|---|---|")
for name, lines in sets.items():
    words = pt = rt = hit = tot = ok = 0
    for x in lines:
        words += len(x.split())
        pt += len(plain.encode(x))
        ps, h, n = enc(x)
        rt += len(ps)
        hit += h
        tot += n
        ok += dec(ps).strip() == x.strip()
    print(f"| {name} | {pt/words:.2f} | {rt/words:.2f} | {hit/tot:.1%} | {ok/len(lines):.1%} |", flush=True)
