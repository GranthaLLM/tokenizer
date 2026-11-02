import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from iast import to_devanagari
from split_classical import is_test

DCS = Path("data/raw/dcs/dcs/data/conllu/files")


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def sentences(work_dir):
    for f in sorted(work_dir.glob("*.conllu")):
        block = []
        for ln in f.read_text(encoding="utf-8").splitlines() + [""]:
            if ln.strip() == "":
                if block:
                    yield parse(block)
                    block = []
            elif not ln.startswith("#"):
                block.append(ln.split("\t"))


def parse(rows):
    sent, i, n = [], 0, len(rows)
    while i < n:
        r = rows[i]
        if len(r) < 3:
            i += 1
        elif "-" in r[0]:
            a, b = r[0].split("-")
            k = int(b) - int(a) + 1
            sent.append((r[1], [rows[i + 1 + j][1] for j in range(k) if i + 1 + j < n]))
            i += 1 + k
        else:
            sent.append((r[1], None))
            i += 1
    return sent


def gold_cuts(surface, comps):
    pos, cuts = 0, []
    for c in comps[:-1]:
        pos += lcp(surface[pos:], c.lower())
        if 0 < pos < len(surface):
            cuts.append(pos)
    return cuts


def main():
    morph, gold = [], []
    for d in sorted(DCS.iterdir()):
        if not d.is_dir():
            continue
        test = is_test("dcs", d.name)
        for sent in sentences(d):
            if not test:
                forms = []
                for surf, comps in sent:
                    forms += comps if comps else [surf]
                line = " ".join(to_devanagari(x) for x in forms)
                if len(line) > 8:
                    morph.append(line)
            else:
                for surf, comps in sent:
                    if comps and len(comps) > 1:
                        cuts = gold_cuts(surf.lower(), comps)
                        if cuts:
                            dev = to_devanagari(surf)
                            gc = sorted({len(to_devanagari(surf.lower()[:c])) for c in cuts})
                            gc = [g for g in gc if 0 < g < len(dev)]
                            if gc:
                                gold.append((dev, gc))
    Path("data/pool/dcs_morph_train.txt").write_text("\n".join(morph) + "\n", encoding="utf-8")
    pickle.dump(gold, open("data/dcs_compound_gold.pkl", "wb"))
    print("train morph lines", len(morph), "test compounds", len(gold))
    for dev, gc in gold[:5]:
        print(dev, gc, "|".join(dev[a:b] for a, b in zip([0] + gc, gc + [len(dev)])))


if __name__ == "__main__":
    main()
