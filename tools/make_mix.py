import glob
import random
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import normalize, stripped

FL = Path("data/flores/flores200_dataset")
K = 20
W = 8
EN_BUDGET = {"wiki": 600_000_000, "fineweb": 600_000_000}
SA_BUDGET = 1_000_000_000
EN_FILES = {
    "wiki": "data/raw/english/20231101.en/train-00000-of-00041.parquet",
    "fineweb": "data/raw/english/sample/10BT/000_00000.parquet",
}


def flores(lang):
    out = []
    for split in ("dev", "devtest"):
        out += (FL / split / f"{lang}.{split}").read_text().splitlines()
    return out


SAN_GRAMS = set()
for s in flores("san_Deva"):
    t = stripped(normalize(s))
    SAN_GRAMS.update(t[i : i + K] for i in range(len(t) - K + 1))
EN_GRAMS = set()
for s in flores("eng_Latn"):
    w = s.lower().split()
    EN_GRAMS.update(" ".join(w[i : i + W]) for i in range(len(w) - W + 1))


def san_leak(line):
    t = stripped(line)
    return any(t[i : i + K] in SAN_GRAMS for i in range(0, len(t) - K + 1, 8))


def en_leak(line):
    w = line.lower().split()
    return any(" ".join(w[i : i + W]) in EN_GRAMS for i in range(len(w) - W + 1))


def en_task(arg):
    path, rg = arg
    tab = pq.ParquetFile(path).read_row_group(rg, columns=["text"])
    out, leaked = [], 0
    for text in tab.column(0).to_pylist():
        for ln in text.split("\n"):
            ln = unicodedata.normalize("NFC", ln).strip()
            if len(ln) < 40 or len(ln) > 3000:
                continue
            if sum(c.isascii() for c in ln) < 0.9 * len(ln):
                continue
            if en_leak(ln):
                leaked += 1
                continue
            out.append(ln)
    return out, leaked


def san_task(chunk):
    out, leaked = [], 0
    for ln in chunk:
        if san_leak(ln):
            leaked += 1
        else:
            out.append(ln)
    return out, leaked


def main():
    random.seed(0)
    for name, path in EN_FILES.items():
        if Path(f"data/pool/en_{name}.txt").exists():
            continue
        with Pool(8) as pool:
            n = pq.ParquetFile(path).num_row_groups
            lines, leaked, chars = [], 0, 0
            for out, lk in pool.imap(en_task, [(path, i) for i in range(n)]):
                lines += out
                leaked += lk
                chars += sum(map(len, out))
                if chars > EN_BUDGET[name] * 1.1:
                    break
            random.shuffle(lines)
            keep, c = [], 0
            for ln in lines:
                if c >= EN_BUDGET[name]:
                    break
                keep.append(ln)
                c += len(ln)
            print(name, "lines", len(keep), "chars", c, "flores-leaked lines dropped", leaked, flush=True)
            Path(f"data/pool/en_{name}.txt").write_text("\n".join(keep) + "\n")
    def chunks():
        buf = []
        with open("data/pool/sample.txt") as f:
            for ln in f:
                ln = ln.rstrip("\n")
                if ln:
                    buf.append(ln)
                if len(buf) == 20000:
                    yield buf
                    buf = []
        if buf:
            yield buf

    keep, c, leaked = [], 0, 0
    with Pool(8) as pool:
        for out, lk in pool.imap(san_task, chunks()):
            leaked += lk
            for ln in out:
                if c >= SA_BUDGET:
                    break
                keep.append(ln)
                c += len(ln)
            if c >= SA_BUDGET:
                pool.terminate()
                break
    print("san lines", len(keep), "chars", c, "flores-leaked dropped (in scanned part)", leaked, flush=True)
    Path("data/pool/san_f.txt").write_text("\n".join(keep) + "\n")


if __name__ == "__main__":
    main()
