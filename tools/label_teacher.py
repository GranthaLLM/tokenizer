import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from byt5_seg import Seg

RUN = re.compile("[ऀ-ॣॱ-ॿ]+")
OUT = Path("data/teacher_labels.pkl")
CAP = 120000
KEEP = ["btn", "anu_vyakhyana", "rg_bhashya", "nyaya_vivarana", "vishnu_tattva_nirnaya", "tantrasara_sangraha", "karma_nirnaya", "krshna_amrta_maharnava", "tattvodyota", "dvadasha_stotra", "tattva_viveka", "upadhi_khandana", "mayavada_khandana", "prapancha_mithyatva_anumana_khandana", "pramana_lakshana", "katha_lakshana", "tattva_sankhyana", "skandha_1", "skandha_2", "skandha_4", "skandha_5", "skandha_6", "skandha_7", "skandha_8", "skandha_9", "skandha_10", "skandha_12"]


def texts(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "text" and isinstance(v, str):
                yield v
            else:
                yield from texts(v)
    elif isinstance(o, list):
        for v in o:
            yield from texts(v)


def collect():
    random.seed(4)
    words = {}
    import os

    if os.environ.get("MODE") == "freq":
        cnt = pickle.load(open("data/run_counts.pkl", "rb"))
        top = [w for w, c in cnt.most_common() if 12 <= len(w) <= 70 and c >= 3][:150000]
        return {w: "freq" for w in top}, len(top)
    for n in KEEP:
        p = Path(f"data/raw/setutila/JSON/{n}.json")
        if p.exists():
            for t in texts(json.load(open(p, encoding="utf-8"))):
                for w in RUN.findall(t):
                    if 8 <= len(w) <= 70:
                        words.setdefault(w, "setutila")
    nset = len(words)
    uoh = pickle.load(open("data/uoh_train.pkl", "rb"))
    ut = [s for (s, _t) in uoh if 8 <= len(s) <= 70]
    random.shuffle(ut)
    for s in ut[: max(0, CAP - nset)]:
        words.setdefault(s, "uoh")
    return words, nset


def main():
    words, nset = collect()
    done = pickle.load(open(OUT, "rb")) if OUT.exists() else {}
    todo = sorted((w for w in words if w not in done), key=len)
    print("setutila words", nset, "total", len(words), "already labelled", len(done), "to do", len(todo), flush=True)
    seg = Seg()
    t0, n0 = time.time(), len(done)
    BS = 128
    for i in range(0, len(todo), BS):
        ws = todo[i : i + BS]
        for w, o in zip(ws, seg.split(ws)):
            done[w] = o
        if (i // BS) % 40 == 0:
            pickle.dump(done, open(OUT, "wb"))
            rate = (len(done) - n0) / (time.time() - t0)
            print(f"{len(done)}/{len(words)} {rate:.0f} words/s", flush=True)
    pickle.dump(done, open(OUT, "wb"))
    print("done", len(done))


if __name__ == "__main__":
    main()
