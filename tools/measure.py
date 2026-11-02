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
    "gemma4": "google/gemma-4-12B-it",
    "qwen3.5": "Qwen/Qwen3.5-9B",
    "qwen3.8": "Qwen/Qwen3.8-Flash-Next",
    "llama4": "unsloth/Llama-4-Scout-17B-16E-Instruct",
    "deepseek-v4.1": "deepseek-ai/DeepSeek-V4.1-Flash",
    "kimi-k3": "moonshotai/Kimi-K3",
    "glm-5.3": "zai-org/GLM-5.3",
    "gpt-oss": "openai/gpt-oss-20b",
    "mistral-large-4": "mistralai/Mistral-Large-4-1T-A52B",
    "command-a-plus": "CohereLabs/command-a-plus-05-2026-bf16",
    "tiny-aya": "CohereLabs/tiny-aya-global",
    "apertus-1.5": "swiss-ai/Apertus-v1.5-8B",
    "nemotron-3.5": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "granite-4.1": "ibm-granite/granite-4.1-3b",
    "qwen3.6": "Qwen/Qwen3.6-27B",
    "olmo-3": "allenai/Olmo-3-7B-Instruct",
    "param2": "bharatgenai/Param2-17B-A2.4B-Thinking",
    "krutrim-2": "krutrim-ai-labs/Krutrim-2-instruct",
    "brahmic-131k": "theschoolofai/BrahmicTokenizer-131K",
    "sarvam-30b": "sarvamai/sarvam-30b",
}


def spm_enc(path):
    sp = spm.SentencePieceProcessor(model_file=path)
    return lambda s: len(sp.encode(s))


def hf_enc(name):
    tk = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
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
    if not names or "baselines" in names or any(n in HF for n in names):
        for k, v in HF.items():
            if names and "baselines" not in names and k not in names:
                continue
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
    out = Path("results/fertility_bilingual.md" if not names else "results/fertility_partial.md")
    out.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
