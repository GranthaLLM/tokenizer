import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import stripped

STRIDE = 8
IDX = None


def init():
    global IDX
    IDX = pickle.load(open("data/holdout_index.pkl", "rb"))


def keep(chunk):
    out, dropped = [], 0
    k, grams, lines = IDX["K"], IDX["grams"], IDX["lines"]
    for ln in chunk:
        t = stripped(ln)
        if t in lines or any(t[i : i + k] in grams for i in range(0, len(t) - k + 1, STRIDE)):
            dropped += 1
        else:
            out.append(ln)
    return out, dropped


def chunks(path):
    buf = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                buf.append(ln)
            if len(buf) == 20000:
                yield buf
                buf = []
    if buf:
        yield buf


def main():
    n = d = 0
    with Pool(8, initializer=init) as pool, open("data/pool/san_h.txt", "w") as out:
        for lines, dropped in pool.imap(keep, chunks("data/pool/san_g.txt")):
            out.write("\n".join(lines) + "\n" if lines else "")
            n += len(lines)
            d += dropped
    print("kept", n, "dropped for any shared window with corpus/", d, f"({d/(n+d):.2%})")


if __name__ == "__main__":
    main()
