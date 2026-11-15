import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compound_eval2 as ce
from eval_taggers import score
from tagger import Tagger

SPECS = {
    "tagger5 (student)": ("data/tagger5.pt", "data/tagger5_vocab.json", 192),
    "tagger6 (bigger, both sides, long boost)": ("data/tagger6.pt", "data/tagger6_vocab.json", 256),
}
HARD = ["सृष्टिस्थित्यप्ययेहानियतिदृशितमोबन्धमोक्षाश्च", "श्रीब्रह्मरुद्रप्रभृतिसुरनरद्व्यीशशत्र्वात्मकस्य", "तद्व्याख्या", "आधिकारिकाधिकरणम्", "तत्तच्छब्दादिनामवान्", "हेत्वादिविकल्पनिराकरणम्", "सर्वज्ञमच्युतमपेतसमस्तदोषम्", "निखिलपूर्णगुणैकदेहं"]
tt = pickle.load(open("data/teacher_test.pkl", "rb"))
long_tt = [g for g in tt if len(g[0]) >= 24]
print(f"teacher held-out: {len(tt)} split words, {len(long_tt)} long (>=24 chars)")
print("| tagger | DCS held-out | ByT5 agreement, all | ByT5 agreement, long words |")
print("|---|---|---|---|")
taggers = {}
for n, (m, v, h) in SPECS.items():
    if not Path(m).exists():
        continue
    tg = Tagger(m, v, hid=h)
    taggers[n] = tg
    print(f"| {n} | {score(tg, ce.GOLD)[2]:.3f} | {score(tg, tt)[2]:.3f} | {score(tg, long_tt)[2]:.3f} |", flush=True)
for n, tg in taggers.items():
    print("\n" + n)
    for w in HARD:
        c = tg.cuts_many([w])[0]
        print("  " + " | ".join(w[a:b] for a, b in zip([0] + c, c + [len(w)])))
