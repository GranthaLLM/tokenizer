import json
import re
import sys
from pathlib import Path

import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from cbpe import apply_map

B = "⁣"
RUN = re.compile("[ऀ-ॣॱ-ॿ]+")


def mark(text, tagger):
    runs = [m.group() for m in RUN.finditer(text)]
    if not runs:
        return text
    cuts = dict(zip(runs, tagger.cuts_many(runs)))

    def sub(m):
        w, cs = m.group(), cuts[m.group()]
        if not cs:
            return w
        out, last = [], 0
        for c in cs:
            out.append(w[last:c])
            last = c
        out.append(w[last:])
        return B.join(out)

    return RUN.sub(sub, text)


class TaggedTok:
    def __init__(self, model, tagger, mapping=None):
        self.sp = spm.SentencePieceProcessor(model_file=model)
        self.tagger = tagger
        self.mp = json.load(open(mapping, encoding="utf-8")) if mapping else {}
        self.inv = {v: k for k, v in self.mp.items()}
        self.bid = self.sp.piece_to_id(B)

    def _prep(self, text):
        t = mark(text, self.tagger)
        return apply_map(t, self.mp) if self.mp else t

    def pieces(self, text):
        out = []
        for i, p in zip(self.sp.encode(self._prep(text)), self.sp.encode(self._prep(text), out_type=str)):
            if i == self.bid:
                continue
            out.append("".join(self.inv.get(c, c) for c in p.replace(B, "")))
        return out

    def count(self, text):
        return sum(1 for i in self.sp.encode(self._prep(text)) if i != self.bid)

    def cuts(self, text):
        out, pos = set(), 0
        for p in self.pieces(text):
            pos += len(p.replace("▁", ""))
            if 0 < pos < len(text):
                out.add(pos)
        return out
