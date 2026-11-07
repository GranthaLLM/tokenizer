"""Bits per byte of open models on held-out Sanskrit, a tokenizer-independent measure of what a model already knows.

usage: python tools/model_bpb.py Qwen/Qwen3-8B-Base [--per-work 4] [--chars 1000]
Writes results/model_bpb/<model>.json. The text is benchmark/heldout_text.parquet: DCS and GRETIL works
that no Grantha component trained on. Models may have seen these works in pretraining.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def documents(per_work, chars):
    t = pq.read_table("benchmark/heldout_text.parquet").to_pydict()
    by = defaultdict(list)
    for x, s, w in zip(t["text"], t["source"], t["work"]):
        by[(s, w)].append(x)
    docs = defaultdict(list)
    for (s, w), lines in sorted(by.items()):
        text = "\n".join(lines)
        if len(text) < chars:
            continue
        starts = [int(i * (len(text) - chars) / max(per_work - 1, 1)) for i in range(per_work)]
        docs[s] += [text[a : a + chars] for a in dict.fromkeys(starts)]
    return docs


@torch.no_grad()
def score(model, tok, texts, device, ctx):
    prefix = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    nll = nbytes = 0.0
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"][: ctx - 1]
        kept = tok.decode(ids)
        x = torch.tensor([[prefix] + ids], device=device)
        logits = model(x).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum").item()
        nbytes += len(kept.encode("utf-8"))
    return nll / (nbytes * math.log(2)), int(nbytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--per-work", type=int, default=4)
    ap.add_argument("--chars", type=int, default=1000)
    ap.add_argument("--ctx", type=int, default=2048)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(device).eval()
    out = {"model": a.model, "params_b": sum(p.numel() for p in model.parameters()) / 1e9, "bpb": {}}
    for source, texts in documents(a.per_work, a.chars).items():
        bpb, nb = score(model, tok, texts, device, a.ctx)
        out["bpb"][source] = {"bpb": round(bpb, 4), "docs": len(texts), "bytes": nb}
        print(a.model, source, f"{bpb:.4f} bits/byte on {len(texts)} docs", flush=True)
    path = Path("results/model_bpb") / (a.model.replace("/", "__") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
