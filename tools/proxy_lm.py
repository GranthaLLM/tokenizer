import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.ln(h) @ self.tok.weight.T


def batches(arr, ctx, bs, rng):
    n = (len(arr) - 1) // ctx
    idx = rng.permutation(n)
    for i in range(0, n - bs + 1, bs):
        s = idx[i : i + bs] * ctx
        yield torch.from_numpy(np.stack([arr[j : j + ctx + 1].astype(np.int64) for j in s]))


@torch.no_grad()
def eval_bits(model, arr, ctx, dev):
    model.eval()
    tot, cnt = 0.0, 0
    n = (len(arr) - 1) // ctx
    for i in range(0, n, 32):
        x = torch.from_numpy(np.stack([arr[j * ctx : j * ctx + ctx + 1].astype(np.int64) for j in range(i, min(i + 32, n))])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        tot += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item()
        cnt += x[:, 1:].numel()
    model.train()
    return tot / math.log(2), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--data", default="data/proxy")
    ap.add_argument("--vocab", type=int, default=128000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    D = Path(a.data)
    train = np.load(D / f"{a.name}_train.npy", mmap_mode="r")
    meta = json.load(open(D / "meta.json"))
    model = GPT(a.vocab, a.d, a.layers, a.heads, a.ctx).to(dev)
    print(f"{a.name}: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, {len(train)/1e6:.1f}M train tokens", flush=True)
    steps_per_epoch = ((len(train) - 1) // a.ctx) // a.bs
    total = int(steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    step, t0, log = 0, time.time(), []
    while step < total:
        for batch in batches(train, a.ctx, a.bs, rng):
            lr = a.lr * min(1, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
            for g in opt.param_groups:
                g["lr"] = lr
            x = batch.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x[:, :-1]).float().reshape(-1, a.vocab), x[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                log.append((step, loss.item()))
                print(f"step {step}/{total} loss {loss.item():.3f} {a.bs*a.ctx*step/(time.time()-t0)/1e3:.0f}k tok/s", flush=True)
            if step >= total:
                break
    res = {"name": a.name, "params_m": sum(p.numel() for p in model.parameters()) / 1e6, "train_tokens": int(len(train)), "steps": total, "bits_per_byte": {}}
    for k, nbytes in meta["val_bytes"].items():
        arr = np.load(D / f"{a.name}_val_{k}.npy")
        bits, cnt = eval_bits(model, arr, a.ctx, dev)
        bpb = bits / cnt * len(arr) / nbytes
        res["bits_per_byte"][k] = {"bpb": bpb, "val_tokens": int(len(arr)), "bytes": nbytes}
        print(f"  {k}: {bpb:.4f} bits per byte", flush=True)
    Path("results").mkdir(exist_ok=True)
    json.dump(res, open(f"results/proxy_{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
