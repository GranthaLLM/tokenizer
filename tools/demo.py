import os
import sys
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sentencepiece as spm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
SAMPLES = [
    "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥",
    "अविद्याध्यासनिमित्तसर्वव्यवहारप्रसिद्धिरेव ब्रह्मात्मैकत्वविज्ञानस्य प्रयोजनम्",
    "तत्त्वमसि श्वेतकेतो सोऽहमस्मि ब्रह्मैवाहम्",
    "राज्यपालस्य कार्यालयं न्यवेदयत् यत् आहतेषु जनेषु 19 रक्षिकाः आसन्।",
    "The Governor's office stated that 19 guards were among the injured.",
]
COLORS = [44, 45, 42, 43, 46, 41]
if os.path.exists("data/demo_samples.txt"):
    SAMPLES = [ln.strip() for ln in open("data/demo_samples.txt", encoding="utf-8") if ln.strip()] + SAMPLES


class Grantha:
    name = "Grantha 128k"

    def __init__(self):
        self.sp = spm.SentencePieceProcessor(model_file="data/tok_v2_dcsgr_128.model")

    def pieces(self, s):
        return [p.replace("▁", "") or "·" for p in self.sp.encode(s, out_type=str)]


class HF:
    def __init__(self, name, repo):
        self.name = name
        self.tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)

    def pieces(self, s):
        ids = self.tk.encode(s, add_special_tokens=False)
        return [self.tk.decode([i]).replace(" ", "") or "·" for i in ids]


def show(models, text):
    words = max(len(text.split()), 1)
    print(f"\n{text}\n{len(text.split())} words")
    for m in models:
        ps = m.pieces(text)
        chips = "".join(f"\033[{COLORS[i % len(COLORS)]};30m{p}\033[0m" for i, p in enumerate(ps))
        print(f"\n{m.name:<14} {len(ps):>3} tokens  {len(ps)/words:4.2f}/word")
        print("  " + chips)


def main():
    models = [
        Grantha(),
        HF("Gemma 3 / 4", "google/gemma-4-12B-it"),
        HF("Param2", "bharatgenai/Param2-17B-A2.4B-Thinking"),
        HF("Qwen3.5", "Qwen/Qwen3.5-9B"),
    ]
    if len(sys.argv) > 1:
        show(models, " ".join(sys.argv[1:]))
        return
    i = 0
    print("Enter: next sample. Type text to tokenize it. q: quit.")
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if line == "q":
            break
        if not line:
            line, i = SAMPLES[i % len(SAMPLES)], i + 1
        show(models, line)


if __name__ == "__main__":
    main()
