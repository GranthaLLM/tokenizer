import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from measure import HF

N = 50000
random.seed(7)
t = pq.read_table("data/pool/classical_test.parquet").to_pydict()
by = defaultdict(list)
for x, s, w in zip(t["text"], t["source"], t["work"]):
    by[s].append((x, w))
sample = {s: random.sample(v, min(N, len(v))) for s, v in by.items()}


def spm_enc(p):
    sp = spm.SentencePieceProcessor(model_file=p)
    return lambda xs: [len(i) for i in sp.encode(xs)]


def hf_enc(n):
    tk = AutoTokenizer.from_pretrained(n, trust_remote_code=False)
    return lambda xs: [len(i) for i in tk(xs, add_special_tokens=False)["input_ids"]]


def main():
    names = sys.argv[1:]
    rows = ["| tokenizer | DCS tok/word | GRETIL tok/word | worst-work GRETIL | best-work GRETIL |", "|---|---|---|---|---|"]
    for n in names:
        try:
            enc = spm_enc(f"data/{n}.model") if Path(f"data/{n}.model").exists() else hf_enc(HF[n])
        except Exception as e:
            print("skip", n, str(e)[:60], file=sys.stderr)
            continue
        res, perwork = {}, defaultdict(lambda: [0, 0])
        for s, items in sample.items():
            toks = enc([x for x, _ in items])
            words = [len(x.split()) for x, _ in items]
            res[s] = sum(toks) / sum(words)
            if s == "gretil":
                for (x, w), tk, wd in zip(items, toks, words):
                    perwork[w][0] += tk
                    perwork[w][1] += wd
        pw = [a / b for a, b in perwork.values() if b >= 500]
        rows.append(f"| {n} | {res['dcs']:.2f} | {res['gretil']:.2f} | {max(pw):.2f} | {min(pw):.2f} |")
        print(rows[-1], flush=True)
    Path("results/classical_eval.md" if len(names) > 8 else "results/classical_eval_partial.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
