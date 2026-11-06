import itertools
import json
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from cbpe import apply_map, build_map, clusters
from tagged import B, RUN, mark
from tagger import Tagger

name, vocab, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
TOTAL = 800_000_000
SAN = TOTAL // 2
random.seed(1)
t = pq.read_table("data/pool/classical_train.parquet").to_pydict()
classical = list(t["text"])
cchars = sum(map(len, classical))


def take(paths, budget):
    its = [open(p) for p in paths]
    out, c = [], 0
    for ln in itertools.chain.from_iterable(zip(*its)):
        if c >= budget:
            break
        ln = ln.rstrip("\n")
        if ln:
            out.append(ln)
            c += len(ln)
    for f in its:
        f.close()
    return out


san = classical + take(["data/pool/san_g.txt"], max(SAN - cchars, 0))
import os

TG = os.environ.get("TAGGER", "1")
tg = Tagger() if TG == "1" else Tagger(f"data/tagger{TG[0]}.pt", f"data/tagger{TG[0]}_vocab.json", hid=(256 if TG[0] == "6" else 192))
if TG.endswith("h"):
    tg.override = {w: list(c) for w, c in pickle.load(open("data/teacher_cuts.pkl", "rb")).items()}
cache = Path("data/pool_cuts.pkl" if TG == "1" else f"data/pool_cuts{os.environ.get('CACHE', TG)}.pkl")
if cache.exists():
    tg.cache = pickle.load(open(cache, "rb"))
else:
    runs = Counter()
    for ln in san:
        runs.update(m.group() for m in RUN.finditer(ln))
    print("unique runs", len(runs), flush=True)
    tg.cuts_many(list(runs))
    pickle.dump(tg.cache, open(cache, "wb"))
print("tagged", len(tg.cache), flush=True)
san = [mark(x, tg) for x in san]
mp = {}
if k:
    cnt = Counter()
    for ln in san:
        cnt.update(clusters(ln))
    mp = build_map(cnt, k)
    json.dump(mp, open(f"data/{name}_map.json", "w", encoding="utf-8"), ensure_ascii=False)
    san = [apply_map(x, mp) for x in san]
lines = san + take(["data/pool/en_wiki.txt", "data/pool/en_fineweb.txt"], TOTAL - SAN)
random.shuffle(lines)
mix = Path(f"data/mix_{name}.txt")
mix.write_text("\n".join(lines) + "\n")
del lines
spm.SentencePieceTrainer.train(
    input=str(mix),
    model_prefix=f"data/tok_{name}",
    model_type="bpe",
    vocab_size=vocab,
    character_coverage=1.0,
    byte_fallback=True,
    normalization_rule_name="identity",
    split_by_unicode_script=False,
    split_digits=True,
    num_threads=8,
    input_sentence_size=6_000_000,
    shuffle_input_sentence=True,
    max_sentence_length=8000,
    user_defined_symbols=[B],
)
mix.unlink()
