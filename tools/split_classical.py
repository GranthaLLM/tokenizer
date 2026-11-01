import hashlib
import pickle
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import stripped

K = 20
STRIDE = 8


def is_test(source, work):
    d = hashlib.blake2b(f"{source}/{work}".encode(), digest_size=2).digest()
    return int.from_bytes(d, "big") % 20 == 0


def hits(t, grams):
    return sum(t[i : i + K] in grams for i in range(0, len(t) - K + 1, STRIDE))


def main():
    corpus = pickle.load(open("data/holdout_index.pkl", "rb"))
    t = pq.read_table("data/pool/classical_all.parquet").to_pydict()
    n = len(t["text"])
    test_rows = [i for i in range(n) if is_test(t["source"][i], t["work"][i])]
    test_set = set(test_rows)
    test_grams = set()
    for i in test_rows:
        s = stripped(t["text"][i])
        test_grams.update(s[j : j + K] for j in range(len(s) - K + 1))
    seen, train_rows, dropped_corpus, dropped_dup, dropped_test = set(), [], 0, 0, 0
    for i in range(n):
        if i in test_set:
            continue
        s = stripped(t["text"][i])
        if s in corpus["lines"] or hits(s, corpus["grams"]) >= 2:
            dropped_corpus += 1
            continue
        if hits(s, test_grams) >= 1:
            dropped_test += 1
            continue
        if s in seen:
            dropped_dup += 1
            continue
        seen.add(s)
        train_rows.append(i)
    schema = pa.schema([("text", pa.string()), ("source", pa.string()), ("work", pa.string()), ("licence", pa.string())])

    def table(rows):
        return pa.table([[t[c][i] for i in rows] for c in ("text", "source", "work", "licence")], schema=schema)

    pq.write_table(table(test_rows), "data/pool/classical_test.parquet", compression="zstd")
    pq.write_table(table(train_rows), "data/pool/classical_train.parquet", compression="zstd")
    pickle.dump({"grams": test_grams, "K": K}, open("data/classical_test_index.pkl", "wb"))
    for name, rows in (("test", test_rows), ("train", train_rows)):
        for src in ("dcs", "gretil"):
            r = [i for i in rows if t["source"][i] == src]
            print(name, src, "works", len({t["work"][i] for i in r}), "lines", len(r), "words", sum(len(t["text"][i].split()) for i in r))
    print("dropped: corpus overlap", dropped_corpus, "test-work overlap", dropped_test, "duplicates", dropped_dup)


if __name__ == "__main__":
    main()
