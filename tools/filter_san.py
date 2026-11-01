import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import stripped

STRIDE = 8
G = K = None


def init():
    global G, K
    d = pickle.load(open("data/classical_test_index.pkl", "rb"))
    G, K = d["grams"], d["K"]


def keep(chunk):
    out, dropped = [], 0
    for ln in chunk:
        t = stripped(ln)
        if any(t[i : i + K] in G for i in range(0, len(t) - K + 1, STRIDE)):
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
    with Pool(8, initializer=init) as pool, open("data/pool/san_g.txt", "w") as out:
        for lines, dropped in pool.imap(keep, chunks("data/pool/san_f.txt")):
            out.write("\n".join(lines) + "\n" if lines else "")
            n += len(lines)
            d += dropped
    print("kept", n, "dropped as overlap with classical test works", d)


if __name__ == "__main__":
    main()
