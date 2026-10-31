import glob
import hashlib
import pickle
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import good_line, normalize, stripped

PERMISSIVE = ["cc0-1.0", "cc-by-4.0", "odc-by", "mit", "apache-2.0"]
STRIDE = 8
IDX = None


def init():
    global IDX
    IDX = pickle.load(open("data/holdout_index.pkl", "rb"))


def leaked(t):
    if t in IDX["lines"]:
        return True
    k, grams, hits = IDX["K"], IDX["grams"], 0
    for i in range(0, len(t) - k + 1, STRIDE):
        if t[i : i + k] in grams:
            hits += 1
            if hits >= 2 or len(t) < 3 * k:
                return True
    return False


def task(arg):
    path, rg = arg
    pf = pq.ParquetFile(path)
    tab = pf.read_row_group(rg, columns=["text", "source", "licence"])
    out, st = [], Counter()
    for text, src, lic in zip(*(tab.column(i).to_pylist() for i in range(3))):
        for ln in text.split("\n"):
            st[src, "in"] += 1
            ln = normalize(ln)
            if not good_line(ln):
                st[src, "quality"] += 1
                continue
            if leaked(stripped(ln)):
                st[src, "holdout"] += 1
                continue
            out.append((ln, src, lic))
    return out, st


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def main():
    jobs = []
    for c in PERMISSIVE:
        for f in sorted(glob.glob(f"data/raw/sansar/data/{c}/*.parquet")):
            jobs += [(f, i) for i in range(pq.ParquetFile(f).num_row_groups)]
    print(len(jobs), "row groups", flush=True)
    seen, stats = set(), Counter()
    words = defaultdict(int)
    chars = defaultdict(int)
    kept = defaultdict(int)
    Path("data/pool").mkdir(parents=True, exist_ok=True)
    writer = None
    schema = pa.schema([("text", pa.string()), ("source", pa.string()), ("licence", pa.string())])
    with Pool(8, initializer=init) as pool:
        for n, (rows, st) in enumerate(pool.imap(task, jobs, chunksize=1)):
            stats.update(st)
            buf = [[], [], []]
            for ln, src, lic in rows:
                k = h(stripped(ln))
                if k in seen:
                    stats[src, "dup"] += 1
                    continue
                seen.add(k)
                buf[0].append(ln)
                buf[1].append(src)
                buf[2].append(lic)
                kept[src] += 1
                words[src] += len(ln.split())
                chars[src] += len(ln)
            if buf[0]:
                if writer is None:
                    writer = pq.ParquetWriter("data/pool/pool_all.parquet", schema, compression="zstd")
                writer.write_table(pa.table(buf, schema=schema))
            if n % 50 == 0:
                print(n, sum(kept.values()), flush=True)
    writer.close()
    srcs = sorted({s for s, _ in stats}, key=lambda s: -words[s])
    lines = ["| source | lines in | quality drop | holdout drop | dup drop | kept | Mwords | Mchars |", "|---|---|---|---|---|---|---|---|"]
    for s in srcs:
        lines.append(f"| {s} | {stats[s,'in']} | {stats[s,'quality']} | {stats[s,'holdout']} | {stats[s,'dup']} | {kept[s]} | {words[s]/1e6:.2f} | {chars[s]/1e6:.1f} |")
    lines.append(f"| total | {sum(stats[s,'in'] for s in srcs)} | {sum(stats[s,'quality'] for s in srcs)} | {sum(stats[s,'holdout'] for s in srcs)} | {sum(stats[s,'dup'] for s in srcs)} | {sum(kept.values())} | {sum(words.values())/1e6:.2f} | {sum(chars.values())/1e6:.1f} |")
    Path("results/pool_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
