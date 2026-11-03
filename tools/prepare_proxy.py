import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from cbpe import CBPE, apply_map
from holdout_index import CORPUS, sanskrit_strings
from tagged import B, TaggedTok
from tagger import Tagger

OUT = Path("data/proxy")
OUT.mkdir(parents=True, exist_ok=True)
SAN_BYTES, EN_BYTES = 330_000_000, 70_000_000
random.seed(0)


def take(path, budget):
    out, n = [], 0
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                out.append(ln)
                n += len(ln.encode())
                if n >= budget:
                    break
    return out


def build_text():
    san = take("data/pool/san_g.txt", SAN_BYTES)
    en = take("data/pool/en_wiki.txt", EN_BYTES // 2) + take("data/pool/en_fineweb.txt", EN_BYTES // 2)
    lines = san + en
    random.shuffle(lines)
    t = pq.read_table("data/pool/classical_test.parquet").to_pydict()
    by = {}
    for x, s in zip(t["text"], t["source"]):
        by.setdefault(s, []).append(x)
    val = {s: random.sample(v, min(8000, len(v))) for s, v in by.items()}
    import json as _j

    corpus = []
    for p in sorted(CORPUS.glob("*/*.json")):
        d = _j.loads(p.read_text())
        if isinstance(d, dict) and "units" in d:
            for s in sanskrit_strings(d["units"]):
                corpus += [x for x in s.split("\n") if x.strip()]
    random.shuffle(corpus)
    val["corpus"] = corpus[:8000]
    val["english"] = (Path("data/flores/flores200_dataset/devtest/eng_Latn.devtest")).read_text().splitlines()
    return lines, val


def make_encoders():
    import sentencepiece as spm

    encs = {}
    sp = spm.SentencePieceProcessor(model_file="data/tok_v2_dcsgr_128.model")
    encs["current"] = lambda x: sp.encode(x)
    cb = CBPE("data/tok_v4_cbpe_128.model", "data/v4_cbpe_128_map.json")
    encs["akshara"] = lambda x: cb.sp.encode(apply_map(x, cb.mp))
    tg = Tagger("data/tagger5.pt", "data/tagger5_vocab.json", hid=192)
    tg.override = {w: list(c) for w, c in pickle.load(open("data/teacher_cuts.pkl", "rb")).items()}
    tg.cache = pickle.load(open("data/pool_cuts5h.pkl", "rb"))
    tt = TaggedTok("data/tok_v8_tagak_128.model", tg, "data/v8_tagak_128_map.json")
    encs["tagged"] = lambda x: [i for i in tt.sp.encode(tt._prep(x)) if i != tt.bid]
    return encs


def main():
    lines, val = build_text()
    vb = {k: sum(len(x.encode()) for x in v) for k, v in val.items()}
    json.dump({"val_bytes": vb, "train_bytes": sum(len(x.encode()) for x in lines)}, open(OUT / "meta.json", "w"))
    print("train bytes", sum(len(x.encode()) for x in lines), "val bytes", vb, flush=True)
    only = sys.argv[1:]
    for name, enc in make_encoders().items():
        if only and name not in only:
            continue
        ids = []
        for i in range(0, len(lines), 20000):
            for ln in lines[i : i + 20000]:
                ids.extend(enc(ln))
                ids.append(2)
            print(name, i + 20000, "/", len(lines), flush=True)
        np.save(OUT / f"{name}_train.npy", np.array(ids, dtype=np.uint32))
        for k, v in val.items():
            vi = []
            for ln in v:
                vi.extend(enc(ln))
                vi.append(2)
            np.save(OUT / f"{name}_val_{k}.npy", np.array(vi, dtype=np.uint32))
        print(name, "train tokens", len(ids), flush=True)


if __name__ == "__main__":
    main()
