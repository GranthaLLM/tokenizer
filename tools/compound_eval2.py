import os
import pickle
import random
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

random.seed(11)
GOLD = pickle.load(open("data/dcs_gold_v2_test.pkl", "rb"))
GOLD = random.sample(GOLD, min(30000, len(GOLD)))
UNSPLIT = pickle.load(open("data/dcs_unsplit_test.pkl", "rb"))
UNSPLIT = random.sample(UNSPLIT, min(30000, len(UNSPLIT)))


def score(cutfn):
    tp = fp = fn = 0
    for w, targets, cands in GOLD:
        cuts = cutfn(w)
        allc = set().union(*[set(c) for c in cands])
        for c in cands:
            if any(x in cuts for x in c):
                tp += 1
            else:
                fn += 1
        fp += len([x for x in cuts if x not in allc])
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    over = sum(len(cutfn(w)) > 0 for w in UNSPLIT) / len(UNSPLIT)
    return p, r, 2 * p * r / max(p + r, 1e-9), over


def spm_cuts(path):
    import sentencepiece as spm

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
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(name, trust_remote_code=False)

    def f(w):
        enc = tk(w, add_special_tokens=False, return_offsets_mapping=True)
        return {e for _, e in enc["offset_mapping"] if 0 < e < len(w)}

    return f


if __name__ == "__main__":
    from measure import HF

    rows = ["| tokenizer | cuts that are right | boundaries found | F1 | unsplit words wrongly cut |", "|---|---|---|---|---|"]
    for n in sys.argv[1:]:
        fn = spm_cuts(f"data/{n}.model") if Path(f"data/{n}.model").exists() else hf_cuts(HF[n])
        p, r, f1, over = score(fn)
        rows.append(f"| {n} | {p:.3f} | {r:.3f} | {f1:.3f} | {over:.1%} |")
        print(rows[-1], flush=True)
    Path("results/compound_eval2.md").write_text("\n".join(rows) + "\n")
