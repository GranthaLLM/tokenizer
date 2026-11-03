import pickle
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compound_eval2 as ce
import uoh_parse as u
from tagger import Tagger


def nyaya_test():
    out = []
    for p in (u.ROOT / "CIIL" / "Tatvacintamani-prakasa").rglob("*"):
        if not p.is_file() or p.suffix == ".zip":
            continue
        t = u.read(p)
        for ln in t.splitlines():
            if "=>" not in ln:
                continue
            l, r = ln.split("=>", 1)
            lt, rt = l.split(), r.split()
            if len(lt) != len(rt):
                continue
            for s, g in zip(lt, rt):
                s = unicodedata.normalize("NFC", s)
                comps = [unicodedata.normalize("NFC", c) for c in g.split("+") if c]
                if len(s) < 2 or not u.DEVW.search(s) or len(comps) < 2:
                    continue
                tg, cd = u.gold(s, comps)
                if tg:
                    out.append((s, tg, cd))
    return list(dict.fromkeys((a, tuple(b), tuple(map(tuple, c))) for a, b, c in out))


def score(tg, G):
    tp = fp = fn = 0
    for w, t, c in G:
        cuts = set(tg.cuts(w))
        allc = set().union(*[set(x) for x in c])
        for x in c:
            if any(y in cuts for y in x):
                tp += 1
            else:
                fn += 1
        fp += len([y for y in cuts if y not in allc])
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


if __name__ == "__main__":
    random.seed(2)
    dcs = ce.GOLD
    uoh = random.sample(pickle.load(open("data/uoh_gold_test.pkl", "rb")), 8000)
    ny = nyaya_test()
    print("nyaya test words", len(ny))
    specs = {"tagger1 (DCS)": ("data/tagger.pt", "data/tagger_vocab.json", 192), "tagger3 (DCS+UoH)": ("data/tagger3.pt", "data/tagger3_vocab.json", 192)}
    print("| tagger | DCS held-out | UoH held-out (stories, drama) | Nyaya held-out (Tattvacintamani) |")
    print("|---|---|---|---|")
    for name, (m, v, h) in specs.items():
        if not Path(m).exists():
            continue
        tg = Tagger(m, v, hid=h)
        row = [score(tg, g)[2] for g in (dcs, uoh, ny)]
        print(f"| {name} | " + " | ".join(f"{x:.3f}" for x in row) + " |", flush=True)
