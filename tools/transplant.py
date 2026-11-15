"""Does the Grantha tokenizer work inside a pretrained model? Swap it into Qwen3-0.6B-Base, train only the embeddings, and track bits per byte on held-out DCS.

usage: python tools/transplant.py grantha|native [--tokens 6000000] [--eval-every 1000000]
Both modes train just the (tied) embedding matrix on the same pool text, so the curves are comparable per training token.
bits per byte is tokenizer independent; the transplanted model also sees about 3x more text per token, reported as bytes_seen.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_bpb import documents, score

BASE = "Qwen/Qwen3-0.6B-Base"
SEQ, BATCH, ACC = 512, 2, 4


def transplant(model, old_tok, new_tok):
    emb = model.get_input_embeddings().weight.data
    mean = emb.mean(0)
    rows = []
    for tid in range(len(new_tok)):
        piece = new_tok.convert_ids_to_tokens(tid)
        if piece is None or (piece.startswith("<") and piece.endswith(">")):
            rows.append(mean)
            continue
        ids = old_tok(piece.replace("▁", " "), add_special_tokens=False)["input_ids"]
        rows.append(emb[ids].mean(0) if ids else mean)
    model.resize_token_embeddings(len(new_tok))
    model.get_input_embeddings().weight.data.copy_(torch.stack(rows))
    model.config.vocab_size = len(new_tok)


def batches(tok, seed):
    pf = pq.ParquetFile("data/pool/train.parquet")
    groups = random.Random(seed).sample(range(pf.num_row_groups), 3)
    lines = pf.read_row_groups(groups, columns=["text"]).column("text").to_pylist()
    random.Random(seed).shuffle(lines)
    print("pool lines", len(lines), flush=True)
    sep = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    buf = []
    for line in lines:
        buf += tok(line, add_special_tokens=False)["input_ids"] + [sep]
        while len(buf) >= SEQ * BATCH:
            chunk, buf = buf[: SEQ * BATCH], buf[SEQ * BATCH :]
            yield torch.tensor(chunk).view(BATCH, SEQ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grantha", "native"])
    ap.add_argument("--tokens", type=int, default=6_000_000)
    ap.add_argument("--eval-every", type=int, default=1_000_000)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    old_tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to("cpu")
    tok = old_tok
    if a.mode == "grantha":
        tok = AutoTokenizer.from_pretrained("release/hf")
        transplant(model, old_tok, tok)
    model.to(device).train()
    for p in model.parameters():
        p.requires_grad = False
    emb = model.get_input_embeddings().weight
    emb.requires_grad = True
    opt = torch.optim.AdamW([emb], lr=a.lr, weight_decay=0.0)
    docs = documents(4, 1000)["dcs"]
    curve = []

    def evaluate(seen, nbytes):
        model.eval()
        bpb, _ = score(model, tok, docs, device, 2048)
        model.train()
        curve.append({"tokens": seen, "bytes_seen": nbytes, "dcs_bpb": round(bpb, 4)})
        print(a.mode, curve[-1], flush=True)

    evaluate(0, 0)
    seen = nbytes = 0
    next_eval = a.eval_every
    steps = a.tokens // (SEQ * BATCH * ACC)
    t0 = time.time()
    micro = 0
    step = 0
    for x in batches(tok, 0):
        x = x.to(device)
        loss = model(x, labels=x).loss
        (loss / ACC).backward()
        micro += 1
        seen += x.numel()
        nbytes += len(tok.decode(x.flatten().tolist()).encode())
        if micro % ACC:
            continue
        step += 1
        warm = min(step / 50, 1.0)
        for g in opt.param_groups:
            g["lr"] = a.lr * warm * (0.5 * (1 + math.cos(math.pi * min(step / steps, 1.0))))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 5 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f} {seen / (time.time() - t0):.0f} tok/s", flush=True)
        if seen >= next_eval:
            evaluate(seen, nbytes)
            next_eval += a.eval_every
        if step >= steps:
            break
    if curve[-1]["tokens"] != seen:
        evaluate(seen, nbytes)
    out = Path("results/transplant")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.mode}.json").write_text(json.dumps({"base": BASE, "mode": a.mode, "lr": a.lr, "curve": curve}, indent=1))


if __name__ == "__main__":
    main()
