import hashlib
import html
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from common import good_line, normalize
from iast import ALLOWED, iast_ratio, to_devanagari

DCS = Path("data/raw/dcs/dcs/data/conllu/files")
GRETIL = Path("data/raw/gretil/gretil.sub.uni-goettingen.de/gretil/corpustei")
TAG = re.compile(r"<[^>]+>")
BREAK = re.compile(r"<(?:/p|/l|lb\s*/|/lg|/head|/div|p|l|lg|head)[^>]*>")
BODY = re.compile(r"<text\b.*?<body>(.*)</body>", re.S)
JUNK = re.compile(r"[=*+~^_<>{}\\#@%&$]")
LIC = re.compile(r'<licence target="([^"]+)"')


def convert(line):
    line = JUNK.sub("", html.unescape(TAG.sub("", line))).strip()
    if len(line) < 6 or iast_ratio(line) < 0.9:
        return None
    if any(c.lower() not in ALLOWED and not c.isalpha() for c in line):
        return None
    if any(c.isalpha() and c.lower() not in ALLOWED and not ("ā" <= c.lower() <= "ṷ" or c.lower() in "ñṅṭḍṇśṣṛṝḷḹṃṁḥ") for c in line):
        return None
    out = normalize(to_devanagari(line))
    return out if good_line(out) else None


def dcs_rows():
    for d in sorted(DCS.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.conllu")):
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.startswith("# text = "):
                    t = convert(ln[9:])
                    if t:
                        yield t, "dcs", d.name, "CC-BY-4.0"


def gretil_rows():
    for f in sorted(GRETIL.glob("sa_*.xml")):
        raw = f.read_text(encoding="utf-8", errors="replace")
        m = BODY.search(raw)
        if not m:
            continue
        lic = LIC.search(raw)
        lic = "CC-BY-NC-SA-4.0" if lic and "by-nc-sa" in lic.group(1) else (lic.group(1) if lic else "unknown")
        body = BREAK.sub("\n", m.group(1))
        for ln in body.split("\n"):
            t = convert(ln)
            if t:
                yield t, "gretil", f.stem, lic


def main():
    schema = pa.schema([("text", pa.string()), ("source", pa.string()), ("work", pa.string()), ("licence", pa.string())])
    cols = [[], [], [], []]
    stats = {}
    for gen in (dcs_rows, gretil_rows):
        for t, s, w, l in gen():
            for c, v in zip(cols, (t, s, w, l)):
                c.append(v)
            a = stats.setdefault((s, w), [0, 0])
            a[0] += 1
            a[1] += len(t.split())
    Path("data/pool").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols, schema=schema), "data/pool/classical_all.parquet", compression="zstd")
    for s in ("dcs", "gretil"):
        works = [v for (a, _), v in stats.items() if a == s]
        print(s, "works", len(works), "lines", sum(v[0] for v in works), "words", sum(v[1] for v in works))


if __name__ == "__main__":
    main()
