import os
import pickle
import random
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
import sentencepiece as spm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from measure import HF

N = 30000
random.seed(11)
gold = pickle.load(open("data/dcs_compound_gold.pkl", "rb"))
gold = random.sample(gold, min(N, len(gold)))


def spm_cuts(path):
    sp = spm.SentencePieceProcessor(model_file=path)

    def f(w):
        pr = sp.encode(w, return_type="proto").pieces
        b2c, acc = {}, 0
        for i, ch in enumerate(w):
            b2c[acc] = i
            acc += len(ch.encode())
        b2c[acc] = len(w)
        return {b2c[p.end] for p in pr if p.end in b2c and 0 < b2c[p.end] < len(w)}

    return f


def hf_cuts(name):
    tk = AutoTokenizer.from_pretrained(name, trust_remote_code=False)

    def f(w):
        enc = tk(w, add_special_tokens=False, return_offsets_mapping=True)
        return {e for _, e in enc["offset_mapping"] if 0 < e < len(w)}

    return f


def score(fn, slack):
    tp = fp = fn_ = 0
    for w, gc in gold:
        cuts = fn(w)
        near = set()
        for g in gc:
            near.update(range(g - slack, g + slack + 1))
        ok = {g for g in gc if any(abs(g - c) <= slack for c in cuts)}
        tp += len(ok)
        fn_ += len(gc) - len(ok)
        fp += len([c for c in cuts if c not in near])
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn_, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    names = sys.argv[1:]
    rows = ["| tokenizer | exact P | exact R | exact F1 | within 1 char F1 |", "|---|---|---|---|---|"]
    for n in names:
        try:
            fn = spm_cuts(f"data/{n}.model") if Path(f"data/{n}.model").exists() else hf_cuts(HF[n])
        except Exception as e:
            print("skip", n, str(e)[:50], file=sys.stderr)
            continue
        p, r, f1 = score(fn, 0)
        f1b = score(fn, 1)[2]
        rows.append(f"| {n} | {p:.3f} | {r:.3f} | {f1:.3f} | {f1b:.3f} |")
        print(rows[-1], flush=True)
    Path("results/compound_eval.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
