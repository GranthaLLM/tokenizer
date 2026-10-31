import itertools
import random
import sys
from pathlib import Path

import sentencepiece as spm

name, share, vocab = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
extra = {"user_defined_symbols": ["।", "॥"]} if "dd" in sys.argv[4:] else {}
TOTAL = 800_000_000
random.seed(1)


def take(paths, budget):
    its = [open(p) for p in paths]
    out, c = [], 0
    for ln in itertools.chain.from_iterable(zip(*its)):
        if c >= budget:
            break
        ln = ln.rstrip("\n")
        out.append(ln)
        c += len(ln)
    for f in its:
        f.close()
    return out


lines = []
if share > 0:
    lines += take(["data/pool/san_f.txt"], TOTAL * share)
if share < 1:
    lines += take(["data/pool/en_wiki.txt", "data/pool/en_fineweb.txt"], TOTAL * (1 - share))
random.shuffle(lines)
mix = Path(f"data/mix_{name}.txt")
mix.write_text("\n".join(lines) + "\n")
print(name, len(lines), "lines", flush=True)
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
    **extra,
)
mix.unlink()
