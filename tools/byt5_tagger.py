import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from akshara import bounds
from align import junctions
from teacher import comps_from


def cuts_from_output(w, out, both=True):
    comps = comps_from(out)
    if len(comps) < 2:
        return []
    bs = bounds(w)
    bset = set(bs)
    cuts = set()
    for pos, _changed in junctions(w, comps):
        if pos in bset:
            cuts.add(pos)
        else:
            lo = max(b for b in bs if b <= pos)
            hi = min(b for b in bs if b > pos)
            cuts.add(lo)
            if both:
                cuts.add(hi)
    return sorted(c for c in cuts if 0 < c < len(w))


class ByT5Tagger:
    def __init__(self, labels="data/teacher_labels.pkl", both=True):
        from byt5_seg import Seg

        self.seg = Seg()
        self.both = both
        self.labels = pickle.load(open(labels, "rb")) if Path(labels).exists() else {}
        self.cache = {}

    def cuts_many(self, words, batch=64):
        todo = [w for w in dict.fromkeys(words) if w not in self.cache and len(w) > 3]
        for w in [w for w in todo if w in self.labels]:
            self.cache[w] = cuts_from_output(w, self.labels[w], self.both)
        todo = [w for w in todo if w not in self.cache]
        for i in range(0, len(todo), batch):
            ws = todo[i : i + batch]
            for w, o in zip(ws, self.seg.split(ws)):
                self.cache[w] = cuts_from_output(w, o, self.both)
        return [self.cache.get(w, []) for w in words]

    def cuts(self, w):
        return set(self.cuts_many([w])[0])
