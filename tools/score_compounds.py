"""Score a tokenizer on the Grantha compound-boundary benchmark.

usage: python tools/score_compounds.py sp:release/grantha-128k.model [--limit N]
       python tools/score_compounds.py hf:google/gemma-4-12B-it

Each benchmark item is a Devanagari word and a list of boundaries; a boundary is a list of
acceptable cut positions (character offsets), because sandhi can fuse a vowel into the letter
before the cut. A cut is correct if it hits any acceptable position of some boundary.
Prints precision, recall (share of true boundaries that are also token boundaries), F1 and
tokens per word.
"""
import argparse
import gzip
import json
from pathlib import Path


def sp_cutter(path):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=path)

    def cuts(w):
        pieces = sp.encode(w, return_type="proto").pieces
        b2c, acc = {}, 0
        for i, ch in enumerate(w):
            b2c[acc] = i
            acc += len(ch.encode())
        b2c[acc] = len(w)
        return {b2c[p.end] for p in pieces if p.end in b2c and 0 < b2c[p.end] < len(w)}, len(pieces)

    return cuts


def hf_cutter(name):
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(name, trust_remote_code=False)

    def cuts(w):
        enc = tk(w, add_special_tokens=False, return_offsets_mapping=True)
        return {e for _, e in enc["offset_mapping"] if 0 < e < len(w)}, len(enc["input_ids"])

    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("--data", default="benchmark/compounds_test.jsonl.gz")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    kind, spec = a.tokenizer.split(":", 1)
    cutter = sp_cutter(spec) if kind == "sp" else hf_cutter(spec)
    tp = fp = fn = toks = n = 0
    with gzip.open(a.data, "rt", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            w, cands = item["word"], item["boundaries"]
            cuts, k = cutter(w)
            allowed = set().union(*map(set, cands))
            for c in cands:
                if cuts & set(c):
                    tp += 1
                else:
                    fn += 1
            fp += len(cuts - allowed)
            toks += k
            n += 1
            if a.limit and n >= a.limit:
                break
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"{a.tokenizer}: words {n}  precision {p:.3f}  recall {r:.3f}  F1 {2*p*r/max(p+r,1e-9):.3f}  tokens per word {toks/n:.2f}")


if __name__ == "__main__":
    main()
