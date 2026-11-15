import hashlib
import html
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from akshara import bounds
import os

from align import junctions, pick

BOTH = os.environ.get("BOTH") == "1"

ROOT = Path("data/raw/uoh")
DEVW = re.compile("[ऀ-ॣॱ-ॿ]")


def read(p):
    for enc in ("utf-8", "utf-16"):
        try:
            txt = p.read_text(encoding=enc)
            break
        except (UnicodeError, OSError):
            continue
    else:
        return ""
    if p.suffix == ".html":
        txt = re.sub(r"<(script|style).*?</\1>", "", txt, flags=re.S)
        txt = html.unescape(re.sub(r"<[^>]+>", "\n", txt))
    return txt


def work_of(p):
    parts = p.relative_to(ROOT).parts
    return "/".join(parts[:4] if parts[0] == "SHMT" else parts[:2])


def is_test_work(w):
    if w == "CIIL/Tatvacintamani-prakasa":
        return True
    return int.from_bytes(hashlib.blake2b(w.encode(), digest_size=2).digest(), "big") % 6 == 0


def gold(surf, comps):
    bs = bounds(surf)
    bset = set(bs)
    found = {}
    for pos, changed in junctions(surf, comps):
        t, alts, extra = pick(pos, bs, bset, set(found), changed, len(surf))
        if 0 < t < len(surf):
            found.setdefault(t, set()).update(alts)
            if BOTH and extra is not None and 0 < extra < len(surf):
                found.setdefault(extra, set()).update(alts)
    targets = sorted(found)
    return targets, [sorted(found[t]) for t in targets]


def main():
    train, test, unsplit_test = Counter(), {}, set()
    per_work = defaultdict(lambda: [0, 0])
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix == ".zip":
            continue
        txt = read(p)
        if "=>" not in txt:
            continue
        w = work_of(p)
        t = is_test_work(w)
        for ln in txt.splitlines():
            if "=>" not in ln:
                continue
            l, r = ln.split("=>", 1)
            lt, rt = l.split(), r.split()
            if len(lt) != len(rt):
                continue
            for s, g in zip(lt, rt):
                s = unicodedata.normalize("NFC", s)
                comps = [unicodedata.normalize("NFC", c) for c in g.split("+") if c]
                if len(s) < 2 or not DEVW.search(s) or not comps or re.search(r"[A-Za-z0-9]", s):
                    continue
                if len(comps) == 1:
                    targets, cands = [], []
                else:
                    targets, cands = gold(s, comps)
                    if not targets:
                        continue
                per_work[w][0] += 1
                if t:
                    if targets:
                        test[(s, tuple(targets))] = cands
                    else:
                        unsplit_test.add(s)
                else:
                    train[(s, tuple(targets))] += 1
    pickle.dump(train, open("data/uoh_train.pkl", "wb"))
    pickle.dump([(s, list(t), c) for (s, t), c in test.items()], open("data/uoh_gold_test.pkl", "wb"))
    pickle.dump(sorted(unsplit_test), open("data/uoh_unsplit_test.pkl", "wb"))
    tw = [w for w in per_work if is_test_work(w)]
    print("works", len(per_work), "test works", len(tw), tw[:8])
    print("train unique", len(train), "split", sum(1 for (s, t) in train if t), "| test split words", len(test), "test unsplit", len(unsplit_test))
    for (s, t), n in train.most_common(3):
        print(s, t, n)


if __name__ == "__main__":
    main()
