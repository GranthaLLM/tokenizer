import json
import re
from pathlib import Path

import sentencepiece as spm

CONS = "क-हक़-य़"
AKSHARA = re.compile(
    rf"(?:[{CONS}]़?्)*[{CONS}]़?(?:्|[ऺऻा-ौॎॏॕ-ॗॢॣ]*[ऀ-ः॑-॔]*)"
    r"|[ऄ-औॠॡ][ऀ-ः॑-॔]*"
)
PUA0 = 0xE000
PUA_MAX = 0xF8FF


def clusters(text):
    return AKSHARA.findall(text)


def build_map(counter, k):
    top = [c for c, _ in counter.most_common() if len(c) > 1][: min(k, PUA_MAX - PUA0 + 1)]
    return {c: chr(PUA0 + i) for i, c in enumerate(top)}


def apply_map(text, mp):
    return AKSHARA.sub(lambda m: mp.get(m.group(), m.group()), text)


class CBPE:
    def __init__(self, model, mapping):
        self.sp = spm.SentencePieceProcessor(model_file=model)
        self.mp = json.load(open(mapping, encoding="utf-8"))
        self.inv = {v: k for k, v in self.mp.items()}

    def unmap(self, s):
        return "".join(self.inv.get(c, c) for c in s)

    def pieces(self, text):
        return [self.unmap(p) for p in self.sp.encode(apply_map(text, self.mp), out_type=str)]

    def count(self, text):
        return len(self.sp.encode(apply_map(text, self.mp)))

    def cuts(self, text):
        out, pos = set(), 0
        for p in self.pieces(text):
            pos += len(p.replace("▁", ""))
            if 0 < pos < len(text):
                out.add(pos)
        return out
