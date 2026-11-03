import pickle
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from akshara import bounds
from align import junctions, pick
from dcs_morph import DCS, sentences
from iast import convert_map
from split_classical import is_test


def norm(x):
    return unicodedata.normalize("NFC", x).lower()


def word_gold(surf, comps):
    surf = norm(surf)
    dev, starts, surf = convert_map(surf)
    bs = bounds(dev)
    bset = set(bs)
    found = {}
    if comps and len(comps) > 1:
        for pos, changed in junctions(surf, [norm(c) for c in comps]):
            t, alts = pick(starts[pos], bs, bset, set(found), changed, len(dev))
            if 0 < t < len(dev):
                found.setdefault(t, set()).update(alts)
    targets = sorted(found)
    return dev, targets, [sorted(found[t]) for t in targets]


def main():
    train, test, unsplit = Counter(), [], set()
    for d in sorted(DCS.iterdir()):
        if not d.is_dir():
            continue
        t = is_test("dcs", d.name)
        for sent in sentences(d):
            for surf, comps in sent:
                dev, targets, cands = word_gold(surf, comps)
                if len(dev) < 2:
                    continue
                if t:
                    if targets:
                        test.append((dev, targets, cands))
                    else:
                        unsplit.add(dev)
                else:
                    train[(dev, tuple(targets))] += 1
    pickle.dump(test, open("data/dcs_gold_v2_test.pkl", "wb"))
    pickle.dump(train, open("data/dcs_tagger_train.pkl", "wb"))
    pickle.dump(sorted(unsplit), open("data/dcs_unsplit_test.pkl", "wb"))
    neg = sum(1 for (w, t) in train if not t)
    print("test compounds", len(test), "train unique (word, cuts)", len(train), "of which unsplit", neg)
    for dev, tg, cd in test[:4]:
        print(dev, tg, cd, "|".join(dev[a:b] for a, b in zip([0] + tg, tg + [len(dev)])))


if __name__ == "__main__":
    main()
