import itertools
import os
import pickle
import re
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from common import stripped

name, split_digits = sys.argv[1], sys.argv[2] == "digits"
TOTAL = 800_000_000
SAN = TOTAL // 2
SPECIAL = ["<|im_start|>", "<|im_end|>"] + [f"<|reserved_{i}|>" for i in range(14)]
random.seed(1)

idx = pickle.load(open("data/holdout_index.pkl", "rb"))
K, grams, exact = idx["K"], idx["grams"], idx["lines"]


def clean(ln):
    t = stripped(ln)
    return t not in exact and not any(t[i : i + K] in grams for i in range(0, len(t) - K + 1, 8))


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


NOSPACE = float(os.environ.get("NOSPACE", "0"))
DEVA_GAP = re.compile("(?<=[ऀ-ॿ]) +(?=[ऀ-ॿ])")


def unspaced(ln):
    return DEVA_GAP.sub("", ln) if random.random() < NOSPACE and sum("ऀ" <= c <= "ॿ" for c in ln) > len(ln) / 2 else ln


mix = Path(f"data/mix_space{round(NOSPACE * 100)}.txt" if NOSPACE else "data/mix_final.txt")
if not mix.exists():
  classical = [x for x in pq.read_table("data/pool/classical_train.parquet").to_pydict()["text"] if clean(x)]
  cchars = sum(map(len, classical))
  lines = classical + take(["data/pool/san_h.txt"], max(SAN - cchars, 0)) + take(["data/pool/en_wiki.txt", "data/pool/en_fineweb.txt"], TOTAL - SAN)
  lines = [unspaced(x) for x in lines]
  random.shuffle(lines)
  mix.write_text("\n".join(lines) + "\n")
  print(name, "lines", len(lines), "classical lines kept", len(classical), flush=True)
  del lines
spm.SentencePieceTrainer.train(
    input=str(mix),
    model_prefix=f"data/tok_{name}",
    model_type="bpe",
    vocab_size=128000,
    character_coverage=1.0,
    byte_fallback=True,
    normalization_rule_name="identity",
    split_by_unicode_script=False,
    split_digits=split_digits,
    remove_extra_whitespaces=False,
    add_dummy_prefix=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    pad_piece="<pad>",
    user_defined_symbols=SPECIAL,
    num_threads=8,
    input_sentence_size=6_000_000,
    shuffle_input_sentence=True,
    max_sentence_length=8000,
)
if name.startswith("final_nodigits"):
    mix.unlink()
