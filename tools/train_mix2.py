import itertools
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

name, variant, vocab = sys.argv[1], sys.argv[2], int(sys.argv[3])
TOTAL = 800_000_000
SAN = TOTAL // 2
random.seed(1)

classical = []
if variant != "none":
    t = pq.read_table("data/pool/classical_train.parquet").to_pydict()
    keep = {"dcs"} if variant == "dcs" else {"dcs", "gretil"}
    classical = [x for x, s in zip(t["text"], t["source"]) if s in keep]
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


lines = classical + take(["data/pool/san_g.txt"], max(SAN - cchars, 0))
lines += take(["data/pool/en_wiki.txt", "data/pool/en_fineweb.txt"], TOTAL - SAN)
random.shuffle(lines)
mix = Path(f"data/mix_{name}.txt")
mix.write_text("\n".join(lines) + "\n")
print(name, variant, "classical chars", cchars, "lines", len(lines), flush=True)
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
)
mix.unlink()
