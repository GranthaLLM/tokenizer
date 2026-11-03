import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from byt5_seg import Seg
from iast import to_devanagari
from uoh_parse import gold


def comps_from(out):
    return [to_devanagari(x) for x in out.replace(" ", "_").split("_") if x]


def teacher_cuts(seg, words, batch=64):
    res = {}
    for i in range(0, len(words), batch):
        ws = words[i : i + batch]
        for w, o in zip(ws, seg.split(ws)):
            comps = comps_from(o)
            targets, _ = gold(w, comps) if len(comps) > 1 else ([], [])
            res[w] = set(targets)
    return res


if __name__ == "__main__":
    import pickle
    import random

    import compound_eval2 as ce
    from eval_taggers import nyaya_test

    random.seed(5)
    seg = Seg()
    sets = {
        "DCS": random.sample(ce.GOLD, 400),
        "UoH": random.sample(pickle.load(open("data/uoh_gold_test.pkl", "rb")), 400),
        "Nyaya": random.sample(nyaya_test(), 400),
    }
    for name, G in sets.items():
        t0 = time.time()
        cuts = teacher_cuts(seg, [g[0] for g in G])
        el = time.time() - t0
        tp = fp = fn = 0
        for w, t, c in G:
            cs = cuts[w]
            allc = set().union(*[set(x) for x in c])
            for x in c:
                if any(y in cs for y in x):
                    tp += 1
                else:
                    fn += 1
            fp += len([y for y in cs if y not in allc])
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"teacher on {name}: P {p:.3f} R {r:.3f} F1 {2*p*r/max(p+r,1e-9):.3f}  ({len(G)/el:.0f} words/s)", flush=True)
