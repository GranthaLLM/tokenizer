import pickle
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from teacher import comps_from
from uoh_parse import gold

lab = pickle.load(open("data/teacher_labels.pkl", "rb"))
random.seed(6)
words = sorted(lab)
random.shuffle(words)
hold = set(words[: max(2000, len(words) // 20)])
train, test, dropped = Counter(), [], 0
for w in words:
    comps = comps_from(lab[w])
    tot = sum(len(c) for c in comps)
    if not comps or not 0.7 * len(w) <= tot <= 1.4 * len(w):
        dropped += 1
        continue
    if len(comps) == 1:
        targets, cands = [], []
    else:
        targets, cands = gold(w, comps)
    if w in hold:
        if targets:
            test.append((w, targets, cands))
    else:
        train[(w, tuple(targets))] += 1
pickle.dump(train, open("data/teacher_train.pkl", "wb"))
pickle.dump(test, open("data/teacher_test.pkl", "wb"))
print("labelled", len(lab), "dropped", dropped, "train", len(train), "split", sum(1 for (w, t) in train if t), "heldout split words", len(test))
