import json
import sys
from pathlib import Path

import sentencepiece as spm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from holdout_index import CORPUS, sanskrit_strings


def english_strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "english" and isinstance(v, str):
                yield v
            else:
                yield from english_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from english_strings(v)


def spm_enc(p):
    sp = spm.SentencePieceProcessor(model_file=p)
    return lambda texts: sum(len(x) for x in sp.encode(texts))


def hf_enc(n):
    tk = AutoTokenizer.from_pretrained(n)
    return lambda texts: sum(len(x) for x in tk(texts, add_special_tokens=False)["input_ids"])


def main():
    encs = {
        "mix50_128": spm_enc("data/tok_mix50_128.model"),
        "dcsgr_128": spm_enc("data/tok_v2_dcsgr_128.model"),
        "none_128": spm_enc("data/tok_v2_none_128.model"),
        "gemma4": hf_enc("google/gemma-4-12B-it"),
        "llama4": hf_enc("unsloth/Llama-4-Scout-17B-16E-Instruct"),
        "qwen3.5": hf_enc("Qwen/Qwen3.5-9B"),
        "param2": hf_enc("bharatgenai/Param2-17B-A2.4B-Thinking"),
        "mistral4": hf_enc("mistralai/Mistral-Large-4-1T-A52B"),
        "gpt-oss": hf_enc("openai/gpt-oss-20b"),
        "sarvam1": hf_enc("sarvamai/sarvam-1"),
    }
    names = list(encs)
    rows = []
    tot = {k: [0, 0, 0, 0] for k in names}
    for p in sorted(CORPUS.glob("*/*.json")):
        d = json.loads(p.read_text())
        if not isinstance(d, dict) or "units" not in d:
            continue
        san = [s for s in sanskrit_strings(d["units"]) if s.strip()]
        en = [s for s in english_strings(d["units"]) if s.strip()]
        sw = sum(len(s.split()) for s in san)
        ew = sum(len(s.split()) for s in en)
        if sw < 200:
            continue
        r = {"work": p.stem, "words": sw}
        for k, e in encs.items():
            ts, te = e(san), e(en)
            r[k] = (ts / sw, te / max(ew, 1))
            tot[k][0] += ts
            tot[k][1] += sw
            tot[k][2] += te
            tot[k][3] += ew
        rows.append(r)
    head = "| work | san words | " + " | ".join(f"{k} san | {k} eng" for k in names) + " |"
    lines = [head, "|" + "---|" * (2 + 2 * len(names))]
    for r in sorted(rows, key=lambda r: -r["mix50_128"][0]):
        lines.append(f"| {r['work']} | {r['words']} | " + " | ".join(f"{r[k][0]:.2f} | {r[k][1]:.2f}" for k in names) + " |")
    lines.append("| **all** | " + str(tot[names[0]][1]) + " | " + " | ".join(f"{tot[k][0]/tot[k][1]:.2f} | {tot[k][2]/tot[k][3]:.2f}" for k in names) + " |")
    Path("results/domain_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
