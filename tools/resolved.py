import json
import re
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from cbpe import apply_map
from sandhi import Joiner, resolve

J = "⁤"
B = "⁣"
RUN = re.compile("[ऀ-ॣॱ-ॿ]+")


class LabelSeg:
    """Looks words up in precomputed ByT5 outputs, falling back to a live segmenter."""

    def __init__(self, labels, live=None):
        self.labels, self.live = labels, live

    def split(self, ws):
        miss = [w for w in ws if w not in self.labels]
        if miss:
            if self.live is None:
                raise KeyError(f"{len(miss)} words without labels")
            self.labels.update(zip(miss, self.live.split(miss)))
        return [self.labels[w] for w in ws]


class ResolvedTok:
    """Tokenizes ByT5-style sandhi-resolved words; decode re-joins them with learned sandhi rules.

    A run whose resolved words do not join back to the surface run exactly is left untouched.
    """

    def __init__(self, model, seg, joiner, mapping=None):
        self.sp = spm.SentencePieceProcessor(model_file=model)
        self.seg, self.joiner = seg, joiner
        self.mp = json.load(open(mapping, encoding="utf-8")) if mapping else {}
        self.inv = {v: k for k, v in self.mp.items()}
        self.cache = {}

    def words(self, runs):
        todo = [r for r in dict.fromkeys(runs) if r not in self.cache]
        for r in [r for r in todo if len(r) <= 3]:
            self.cache[r] = None
        todo = [r for r in todo if r not in self.cache]
        for i in range(0, len(todo), 64):
            ws = todo[i : i + 64]
            for w, o in zip(ws, self.seg.split(ws)):
                self.cache[w] = resolve(self.joiner, w, o)
        return [self.cache[r] for r in runs]

    def prep(self, text):
        runs = [m.group() for m in RUN.finditer(text)]
        res = dict(zip(runs, self.words(runs)))
        t = RUN.sub(lambda m: J.join(res[m.group()]) if res[m.group()] else m.group(), text)
        return apply_map(t, self.mp) if self.mp else t

    def pieces(self, text):
        return ["".join(self.inv.get(c, c) for c in p) for p in self.sp.encode(self.prep(text), out_type=str)]

    def decode(self, pieces):
        s = "".join(pieces).replace("▁", " ")
        return RUN2.sub(lambda m: self.joiner.join_deva(m.group().split(J)), s)


RUN2 = re.compile(f"[ऀ-ॣॱ-ॿ{J}]+")


if __name__ == "__main__":
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from byt5_seg import Seg

    tk = ResolvedTok("data/tok_v8_tagak_128.model", Seg(), Joiner.load("data/sandhi_join.json"), "data/v8_tagak_128_map.json")
    for t in sys.argv[1:]:
        ps = tk.pieces(t)
        print(" | ".join(p.replace(J, "·").replace("▁", "") for p in ps))
        print("roundtrip:", tk.decode(ps).strip() == t)
