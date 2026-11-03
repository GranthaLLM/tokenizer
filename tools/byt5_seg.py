import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from deva2iast import to_iast

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NAME = "chronbmm/sanskrit5-multitask"


class Seg:
    def __init__(self):
        self.tk = AutoTokenizer.from_pretrained(NAME)
        self.m = T5ForConditionalGeneration.from_pretrained(NAME, torch_dtype=torch.float32).to(DEV).eval()

    @torch.no_grad()
    def split(self, lines, beams=1):
        enc = self.tk(["S " + to_iast(x) for x in lines], return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
        out = self.m.generate(**enc, max_length=512, num_beams=beams)
        return [self.tk.decode(o, skip_special_tokens=True) for o in out]


if __name__ == "__main__":
    import time

    s = Seg()
    for t in sys.argv[1:] or ["सृष्टिस्थित्यप्ययेहानियतिदृशितमोबन्धमोक्षाश्च"]:
        t0 = time.time()
        print(t, "->", s.split([t])[0], f"({time.time()-t0:.1f}s)")
