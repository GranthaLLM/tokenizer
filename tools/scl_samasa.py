"""UoHyd samaasa-tagged corpus -> compound gold.

Input: data/scl/samas/*.html and unzipped *.txt, lines like `<धर्म-क्षेत्रे>T6`.
The tags give sandhi-resolved leaf components. The written compound is rebuilt
with the learned joiner and kept only if that word is attested in the corpus.
"""

import glob
import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sandhi import Joiner
from uoh_parse import gold

INNER = re.compile(r"<([^<>]*)>[A-Za-z][A-Za-z0-9]*")
NOISE = re.compile(r"\{[^}]*\}|[\s‌‍]")
DEVA = re.compile("^[ऀ-ॣॱ-ॿ]+$")


def leaves(inner):
    while True:
        out = INNER.sub(lambda m: m.group(1), inner)
        if out == inner:
            break
        inner = out
    return [c for c in (NOISE.sub("", p) for p in inner.split("-")) if c]


def compounds(text):
    pos = 0
    while True:
        start = text.find("<", pos)
        if start < 0:
            return
        depth, i = 0, start
        while i < len(text):
            if text[i] == "<":
                depth += 1
            elif text[i] == ">":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        else:
            return
        m = re.match(r"[A-Za-z][A-Za-z0-9]*", text[i + 1 :])
        if m and "\n" not in text[start:i]:
            yield text[start + 1 : i]
        pos = i + 1


def main():
    joiner = Joiner.load("data/sandhi_join.json")
    attested = pickle.load(open("data/teacher_cuts.pkl", "rb"))
    files = glob.glob("data/scl/samas/*.html") + glob.glob("data/scl/samas/x/**/*.txt", recursive=True)
    seen, items, total, rebuilt = set(), [], 0, 0
    for f in sorted(files):
        t = open(f, encoding="utf-8", errors="ignore").read()
        for inner in compounds(t):
            total += 1
            parts = leaves(inner)
            if len(parts) < 2 or not all(DEVA.match(p) for p in parts):
                continue
            surf = joiner.join_deva(parts)
            if not DEVA.match(surf) or surf in seen:
                continue
            rebuilt += 1
            if surf not in attested:
                continue
            seen.add(surf)
            targets, cands = gold(surf, parts)
            if targets:
                items.append({"word": surf, "parts": parts, "boundaries": cands})
    print(f"tags {total}  rebuilt-unique {rebuilt}  attested+gold {len(items)}")
    with open("data/scl_samasa_gold.jsonl", "w", encoding="utf-8") as o:
        for i, it in enumerate(items):
            o.write(json.dumps({"id": i, **it}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
