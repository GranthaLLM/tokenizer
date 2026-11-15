import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from teacher import comps_from
from uoh_parse import gold

lab = pickle.load(open("data/teacher_labels.pkl", "rb"))
if Path("data/resolve_labels.pkl").exists():
    lab.update(pickle.load(open("data/resolve_labels.pkl", "rb")))
cuts, dropped = {}, 0
for w, o in lab.items():
    comps = comps_from(o)
    tot = sum(len(c) for c in comps)
    if not comps or not 0.7 * len(w) <= tot <= 1.4 * len(w):
        dropped += 1
        continue
    cuts[w] = tuple(gold(w, comps)[0]) if len(comps) > 1 else ()
pickle.dump(cuts, open("data/teacher_cuts.pkl", "wb"))
print("dictionary words", len(cuts), "dropped", dropped, "with cuts", sum(1 for c in cuts.values() if c))
w = "सृष्टिस्थित्यप्ययेहानियतिदृशितमोबन्धमोक्षाश्च"
print("user word in dictionary:", w in cuts)
