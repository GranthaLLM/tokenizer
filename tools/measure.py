import sys
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm
from transformers import AutoTokenizer

F = Path("data/flores/flores200_dataset/devtest")
flores = {l: (F / f"{l}.devtest").read_text().splitlines() for l in ("san_Deva", "eng_Latn", "hin_Deva")}
held = pq.read_table("data/pool/heldout.parquet").to_pydict()["text"]
held = held[:: max(1, len(held) // 5000)]

HF = {
    "llama3": "NousResearch/Meta-Llama-3-8B",
    "gemma3": "unsloth/gemma-3-4b-it",
    "qwen3": "Qwen/Qwen3-8B",
    "sarvam1": "sarvamai/sarvam-1",
    "sarvam-m": "sarvamai/sarvam-m",
}


def spm_enc(path):
    sp = spm.SentencePieceProcessor(model_file=path)
    return lambda s: len(sp.encode(s))


def hf_enc(name):
    tk = AutoTokenizer.from_pretrained(name)
    return lambda s: len(tk.encode(s, add_special_tokens=False))


def stats(enc, lines):
    toks = sum(enc(s) for s in lines)
    w = sum(len(s.split()) for s in lines)
    c = sum(len(s) for s in lines)
    b = sum(len(s.encode()) for s in lines)
    return toks / w, toks / c, toks / b


def main():
    names = sys.argv[1:]
    encs = {}
    for p in sorted(Path("data").glob("tok_*.model")):
        if not names or p.stem in names or "baselines" in names and p.stem.startswith("tok_bpe"):
            encs[p.stem] = spm_enc(str(p))
    if not names or "baselines" in names:
        for k, v in HF.items():
            try:
                encs[k] = hf_enc(v)
            except Exception as e:
                print("skip", k, type(e).__name__, str(e)[:80], file=sys.stderr)
    rows = [
        "| tokenizer | san tok/word | eng tok/word | san/eng | san tok/char | eng tok/char | san held-out tok/word |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, enc in encs.items():
        s = stats(enc, flores["san_Deva"])
        e = stats(enc, flores["eng_Latn"])
        h = stats(enc, held)
        ratio = sum(enc(x) for x in flores["san_Deva"]) / sum(enc(x) for x in flores["eng_Latn"])
        rows.append(f"| {k} | {s[0]:.2f} | {e[0]:.2f} | {ratio:.2f} | {s[1]:.3f} | {e[1]:.3f} | {h[0]:.2f} |")
        print(rows[-1], flush=True)
    out = Path("results/fertility_bilingual.md")
    out.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
