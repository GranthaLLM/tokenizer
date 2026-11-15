import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deva2iast import to_iast
from iast import to_devanagari

END = "$"
PH = re.compile("[kgcjṭḍtdpb]h|a[iu]|[^\\W\\d_]|'", re.U)


def phonemes(s):
    return PH.findall(s)


def key_a(a, depth):
    return "".join(phonemes(a)[-depth:])


def key_b(b):
    ph = phonemes(b)
    return ph[0] if ph else ""


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def align_junction(surf, pos, a, b):
    """Find (drop, X, skip) so surf[pos:] == a[:len(a)-drop] + X + b[skip:]..."""
    k = lcp(surf[pos:], a)
    best = None
    for drop in range(0, 4):
        keep = len(a) - drop
        if keep < 0 or keep > k:
            continue
        for skip in range(0, 3):
            if skip >= len(b):
                continue
            tail = b[skip : skip + 3]
            for xl in range(0, 4):
                s0 = pos + keep
                if surf[s0 + xl : s0 + xl + len(tail)] != tail:
                    continue
                cost = drop + xl + skip
                if best is None or cost < best[0]:
                    best = (cost, drop, surf[s0 : s0 + xl], skip)
    return best


def learn(labels, depths=(1, 2, 3)):
    votes = {d: defaultdict(Counter) for d in depths}
    for w, out in labels.items():
        comps = [c for c in out.replace(" ", "_").split("_") if c]
        if len(comps) < 2:
            continue
        surf = to_iast(w)
        pos = 0
        cur = comps[0]
        for nxt in comps[1:]:
            if not cur:
                break
            r = align_junction(surf, pos, cur, nxt)
            if r is None:
                break
            _, drop, x, skip = r
            for d in depths:
                votes[d][(key_a(cur, d), key_b(nxt))][(drop, x, skip)] += 1
            pos += len(cur) - drop + len(x)
            cur = nxt[skip:]
        else:
            k = lcp(surf[pos:], cur)
            if k > 0 and len(cur) - k <= 3 and len(surf) - pos - k <= 3:
                for d in depths:
                    votes[d][(key_a(cur, d), END)][(len(cur) - k, surf[pos + k :], 0)] += 1
    return {d: {k: c.most_common(1)[0][0] for k, c in v.items()} for d, v in votes.items()}


class Joiner:
    def __init__(self, table, depths=(3, 2, 1)):
        self.table = {int(d): {tuple(k.split("\t")) if isinstance(k, str) else k: tuple(v) for k, v in t.items()} for d, t in table.items()}
        self.depths = depths

    def rule(self, acc, b):
        kb = key_b(b)
        for d in self.depths:
            r = self.table[d].get((key_a(acc, d), kb))
            if r:
                return r
        return (0, "", 0)

    def join(self, words):
        acc = words[0]
        for b in words[1:] + [END]:
            drop, x, skip = self.rule(acc, b)
            acc = (acc[: len(acc) - drop] if drop else acc) + x + (b[skip:] if b != END else "")
        return acc

    def join_deva(self, words):
        return to_devanagari(self.join([to_iast(w) for w in words]))

    def save(self, path):
        ser = {str(d): {"\t".join(k): list(v) for k, v in t.items()} for d, t in self.table.items()}
        json.dump(ser, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path, encoding="utf-8")))


def final_variants(last):
    yield last
    if last.endswith("m"):
        yield last[:-1] + "ṃ"
        yield last[:-1] + "ṁ"
    if last.endswith("aḥ"):
        yield last[:-2] + "o"
        yield last[:-2] + "o'"
    if last.endswith("ḥ"):
        for r in ("r", "s", "ś", "ṣ", ""):
            yield last[:-1] + r
    if last.endswith("t"):
        yield last[:-1] + "d"
    if last.endswith("n"):
        yield last[:-1] + "ṃ"


def resolve(joiner, word, out):
    """ByT5 output -> sandhi-resolved Devanagari words that join back to `word`, else None."""
    comps = [c for c in out.replace(" ", "_").split("_") if c]
    if len(comps) < 2:
        return None
    head = comps[:-1]
    for last in final_variants(comps[-1]):
        deva = [to_devanagari(c) for c in head + [last]]
        if joiner.join_deva(deva) == word:
            return deva
    return None


if __name__ == "__main__":
    import random

    clean = re.compile("^[ऀ-ॣॱ-ॿ]+$")
    labels = {}
    for p in sys.argv[1:] or ["data/teacher_labels.pkl"]:
        labels.update(pickle.load(open(p, "rb")))
    items = [(w, o) for w, o in labels.items() if clean.match(w) and len([c for c in o.replace(" ", "_").split("_") if c]) > 1]
    random.seed(0)
    random.shuffle(items)
    cut = len(items) * 9 // 10
    ser = lambda t: {d: {"\t".join(k): list(v) for k, v in tt.items()} for d, tt in t.items()}
    j = Joiner(ser(learn(dict(items[:cut]))))
    test = items[cut:]
    ok = [(w, o) for w, o in test if resolve(j, w, o)]
    print(f"multi-component words {len(items)}  held-out {len(test)}  resolvable {len(ok)/len(test):.4f}")
    for w, o in [x for x in test if x not in set(ok)][:20]:
        print(" ", w, "<-", o)
    Joiner(ser(learn(dict(items)))).save("data/sandhi_join.json")
