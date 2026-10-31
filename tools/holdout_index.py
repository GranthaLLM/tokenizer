import json
import pickle
from pathlib import Path

from common import normalize, stripped

K = 20
CORPUS = Path(__file__).resolve().parents[2] / "corpus"


def sanskrit_strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "sanskrit" and isinstance(v, str):
                yield v
            else:
                yield from sanskrit_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from sanskrit_strings(v)


def main():
    grams, lines, works = set(), set(), 0
    for p in sorted(CORPUS.glob("*/*.json")):
        d = json.loads(p.read_text())
        if not isinstance(d, dict) or "units" not in d:
            continue
        works += 1
        for s in sanskrit_strings(d["units"]):
            for ln in s.split("\n"):
                t = stripped(normalize(ln))
                if len(t) < 8:
                    continue
                lines.add(t)
                for i in range(len(t) - K + 1):
                    grams.add(t[i : i + K])
    Path("data").mkdir(exist_ok=True)
    with open("data/holdout_index.pkl", "wb") as f:
        pickle.dump({"grams": grams, "lines": lines, "K": K}, f)
    print(f"works {works} lines {len(lines)} grams {len(grams)}")


if __name__ == "__main__":
    main()
