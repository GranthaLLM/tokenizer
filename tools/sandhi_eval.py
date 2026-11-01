import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import sentencepiece as spm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
D = Path("data/SandhiKosh")
SHEETS = [
    ("Astaadhyaayii Corpus.xls", None, "ashtadhyayi"),
    ("Bhagvad_Gita Corpus.xls", None, "gita"),
    ("Rule-based Corpus and Literature Corpus.xls", "Internal", "internal"),
    ("Rule-based Corpus and Literature Corpus.xls", "External", "external"),
    ("Rule-based Corpus and Literature Corpus.xls", "Literature", "literature"),
    ("UoH_Corpus.xls", None, "uoh"),
]


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load():
    items = defaultdict(list)
    for f, sh, tag in SHEETS:
        df = pd.read_excel(D / f, sheet_name=sh or 0, header=None, engine="xlrd")
        for _, r in df.iterrows():
            w, sp = r.get(1), r.get(2)
            if not isinstance(w, str) or not isinstance(sp, str) or "+" not in sp:
                continue
            w, comps = w.strip(), [c.strip() for c in sp.split("+")]
            if " " in w or len(w) < 3 or len(comps) < 2 or any(not c for c in comps):
                continue
            pos, gold = 0, set()
            for c in comps[:-1]:
                pos += lcp(w[pos:], c)
                if 0 < pos < len(w):
                    gold.add(pos)
            if gold:
                items[tag].append((w, gold))
    return items


def spm_bounds(path):
    sp = spm.SentencePieceProcessor(model_file=path)

    def f(w):
        pr = sp.encode(w, return_type="proto").pieces
        cuts, b2c = set(), {}
        bs = w.encode()
        acc = 0
        for i, ch in enumerate(w):
            b2c[acc] = i
            acc += len(ch.encode())
        b2c[acc] = len(w)
        for p in pr:
            e = p.end
            if e in b2c and 0 < b2c[e] < len(w):
                cuts.add(b2c[e])
        return cuts, len(pr)

    return f


def hf_bounds(name):
    tk = AutoTokenizer.from_pretrained(name)

    def f(w):
        enc = tk(w, add_special_tokens=False, return_offsets_mapping=True)
        cuts = {e for _, e in enc["offset_mapping"] if 0 < e < len(w)}
        return cuts, len(enc["input_ids"])

    return f


def score(fn, items):
    tp = fp = fn_ = ntok = n = 0
    for w, gold in items:
        cuts, k = fn(w)
        ntok += k
        n += 1
        ok = {g for g in gold if g in cuts or g + 1 in cuts}
        tp += len(ok)
        fn_ += len(gold) - len(ok)
        near = set(gold) | {g + 1 for g in gold}
        fp += len([c for c in cuts if c not in near])
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn_, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), ntok / n


def main():
    items = load()
    allit = [x for v in items.values() for x in v]
    print({k: len(v) for k, v in items.items()}, "total", len(allit), flush=True)
    models = {
        "bpe_sa_128k": spm_bounds("data/tok_bpe_128k.model"),
        "mix50_128": spm_bounds("data/tok_mix50_128.model"),
        "v2_dcsgr_128": spm_bounds("data/tok_v2_dcsgr_128.model"),
        "mix50_64": spm_bounds("data/tok_mix50_64.model"),
    }
    for k, n in [("gemma3", "unsloth/gemma-3-4b-it"), ("sarvam1", "sarvamai/sarvam-1")]:
        models[k] = hf_bounds(n)
    rows = ["| tokenizer | precision | recall | F1 | tokens/word |", "|---|---|---|---|---|"]
    for k, fn in models.items():
        p, r, f1, t = score(fn, allit)
        rows.append(f"| {k} | {p:.3f} | {r:.3f} | {f1:.3f} | {t:.2f} |")
        print(rows[-1], flush=True)
    Path("results/sandhi_alignment.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()


def chance(items, fn, seed=0):
    import random

    rnd = random.Random(seed)
    tp = fp = fn_ = 0
    for w, gold in items:
        k = fn(w)[1]
        cuts = set(rnd.sample(range(1, len(w)), min(max(k - 1, 0), len(w) - 1)))
        ok = {g for g in gold if g in cuts or g + 1 in cuts}
        near = set(gold) | {g + 1 for g in gold}
        tp += len(ok)
        fn_ += len(gold) - len(ok)
        fp += len([c for c in cuts if c not in near])
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn_, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)
