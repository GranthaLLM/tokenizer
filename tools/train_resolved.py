import itertools
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from resolved import J, RUN
from sandhi import Joiner, resolve

name, vocab = sys.argv[1], int(sys.argv[2])
label_files = sys.argv[3:] or ["data/teacher_labels.pkl", "data/resolve_labels.pkl"]
TOTAL = 800_000_000
SAN = TOTAL // 2
random.seed(1)


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


classical = list(pq.read_table("data/pool/classical_train.parquet").to_pydict()["text"])
cchars = sum(map(len, classical))
san = classical + take(["data/pool/san_g.txt"], max(SAN - cchars, 0))

labels = {}
for p in label_files:
    labels.update(pickle.load(open(p, "rb")))
joiner = Joiner.load("data/sandhi_join.json")

runs = Counter()
for ln in san:
    runs.update(m.group() for m in RUN.finditer(ln))
print("unique runs", len(runs), "labelled", sum(1 for r in runs if r in labels), flush=True)

resolved = {}
for r, o in ((r, labels[r]) for r in runs if r in labels):
    words = resolve(joiner, r, o)
    if words:
        resolved[r] = J.join(words)
print("resolved types", len(resolved), "mass", sum(runs[r] for r in resolved) / sum(runs.values()), flush=True)

san = [RUN.sub(lambda m: resolved.get(m.group(), m.group()), x) for x in san]
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
    user_defined_symbols=[J],
)
mix.unlink()
