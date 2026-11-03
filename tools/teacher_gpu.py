import argparse
import pickle
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from deva2iast import to_iast

NAME = "chronbmm/sanskrit5-multitask"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--max-bytes", type=int, default=40000, help="padded bytes per batch")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tk = AutoTokenizer.from_pretrained(NAME)
    model = T5ForConditionalGeneration.from_pretrained(NAME, torch_dtype=dtype).to(dev).eval()
    words = [w for w in Path(a.words).read_text(encoding="utf-8").split("\n") if w]
    out = pickle.load(open(a.out, "rb")) if Path(a.out).exists() else {}
    todo = [w for w in words if w not in out]
    print(f"device {dev}, {len(words)} words, {len(out)} done, {len(todo)} to do", flush=True)
    t0, n0 = time.time(), len(out)
    for c in range(0, len(todo), a.chunk):
        chunk = sorted(todo[c : c + a.chunk], key=len)
        i = 0
        while i < len(chunk):
            n = max(1, a.max_bytes // (len(to_iast(chunk[min(i + 1, len(chunk) - 1)])) * 2 + 8))
            ws = chunk[i : i + n]
            while len(ws) > 1 and len(ws) * (len(to_iast(ws[-1])) + 4) > a.max_bytes:
                ws = ws[:-1]
            enc = tk(["S " + to_iast(w) for w in ws], return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_length=256, num_beams=1)
            for w, g in zip(ws, gen):
                out[w] = tk.decode(g, skip_special_tokens=True)
            i += len(ws)
        pickle.dump(out, open(a.out, "wb"))
        print(f"{len(out)}/{len(words)}  {(len(out)-n0)/(time.time()-t0):.0f} words/s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
