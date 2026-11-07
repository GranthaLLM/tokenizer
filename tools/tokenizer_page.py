"""Tokenizer inspector: type text, see Grantha's tokens next to any tokenizer in the local Hugging Face cache.

usage: python tools/tokenizer_page.py [--port 8770] [--no-open]
Every comparison tokenizer is the real one, loaded from ~/.cache/huggingface (offline).
"""
import argparse
import json
import os
import threading
import unicodedata
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

import sentencepiece as spm

MAX_CHARS = 3000
GRANTHA_MODEL = "release/grantha-128k.model"

LABELS = {
    "google/gemma-4-12B-it": ("Gemma 4", "Google"),
    "google/gemma-3-4b-it": ("Gemma 3", "Google"),
    "google/gemma-4-E2B-it": ("Gemma 4 E2B", "Google"),
    "unsloth/Llama-4-Scout-17B-16E-Instruct": ("Llama 4 Scout", "Meta"),
    "NousResearch/Meta-Llama-3-8B": ("Llama 3", "Meta"),
    "Qwen/Qwen3.5-9B": ("Qwen 3.5", "Qwen"),
    "Qwen/Qwen3.6-27B": ("Qwen 3.6", "Qwen"),
    "Qwen/Qwen3.8-Flash-Next": ("Qwen 3.8 Flash", "Qwen"),
    "Qwen/Qwen3-8B": ("Qwen 3", "Qwen"),
    "mistralai/Mistral-Large-4-1T-A52B": ("Mistral Large 4", "Mistral"),
    "deepseek-ai/DeepSeek-V4.1-Flash": ("DeepSeek V4.1", "DeepSeek"),
    "zai-org/GLM-5.3": ("GLM 5.3", "Z.ai"),
    "openai/gpt-oss-20b": ("gpt-oss 20B", "OpenAI"),
    "allenai/Olmo-3-7B-Instruct": ("Olmo 3", "Ai2"),
    "ibm-granite/granite-4.1-3b": ("Granite 4.1", "IBM"),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": ("Nemotron 3.5", "NVIDIA"),
    "CohereLabs/command-a-plus-05-2026-bf16": ("Command A+", "Cohere"),
    "sarvamai/sarvam-1": ("Sarvam-1", "Indic"),
    "sarvamai/sarvam-m": ("Sarvam-M", "Indic"),
    "sarvamai/sarvam-30b": ("Sarvam 30B", "Indic"),
    "bharatgenai/Param2-17B-A2.4B-Thinking": ("Param2", "Indic"),
    "krutrim-ai-labs/Krutrim-2-instruct": ("Krutrim-2", "Indic"),
    "theschoolofai/BrahmicTokenizer-131K": ("Brahmic 131K", "Indic"),
}

DEFAULT_TEXT = "वित्तव्याधिविकारविह्वलगिरां नामापि न श्रूयते"


def cached_repos():
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    found = {p.name[8:].replace("--", "/") for p in hub.glob("models--*")}
    return [r for r in LABELS if r in found]


def split_deltas(ids, decode):
    """Per-token text. Tokens that end inside a multi-byte letter show as the placeholder until the letter completes."""
    out, committed = [], ""
    for k in range(len(ids)):
        cur = decode(ids[: k + 1])
        if cur.endswith("�"):
            out.append({"t": "▯", "partial": True})
            continue
        out.append({"t": cur[len(committed) :], "partial": False})
        committed = cur
    return out


class Grantha:
    name = "Grantha 128k"

    def __init__(self):
        self.sp = spm.SentencePieceProcessor(model_file=GRANTHA_MODEL)

    def encode(self, text):
        ids = self.sp.encode(text)
        toks = split_deltas(ids, self.sp.decode)
        for t, i in zip(toks, ids):
            t["id"] = i
        return toks


class HFTok:
    def __init__(self, repo):
        from transformers import AutoTokenizer

        self.name = LABELS[repo][0]
        self.tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)

    def encode(self, text):
        ids = self.tk.encode(text, add_special_tokens=False)
        toks = split_deltas(ids, lambda x: self.tk.decode(x, clean_up_tokenization_spaces=False))
        for t, i in zip(toks, ids):
            t["id"] = i
        return toks


GR = None
CACHE = {}
LOCK = threading.Lock()


def get_other(repo):
    with LOCK:
        if repo not in CACHE:
            try:
                CACHE[repo] = HFTok(repo)
            except Exception as e:
                CACHE[repo] = str(e).split("\n")[0][:200]
        return CACHE[repo]


def run(text, repo):
    text = unicodedata.normalize("NFC", text)[:MAX_CHARS]
    words = max(len(text.split()), 1)
    res = {"words": len(text.split())}
    g = GR.encode(text)
    res["grantha"] = {"name": GR.name, "count": len(g), "tokens": g}
    if repo:
        o = get_other(repo)
        if isinstance(o, str):
            res["error"] = f"{LABELS[repo][0]} could not be loaded: {o}"
        else:
            t = o.encode(text)
            res["other"] = {"name": o.name, "count": len(t), "tokens": t}
    res["perWord"] = len(g) / words
    return res


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json; charset=utf-8", code=200):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/config":
            repos = [{"repo": r, "name": LABELS[r][0], "group": LABELS[r][1]} for r in cached_repos()]
            self._send(json.dumps({"tokenizers": repos, "default": DEFAULT_TEXT, "max": MAX_CHARS}, ensure_ascii=False))
        elif self.path in ("/", "/index.html"):
            page = (Path(__file__).parent / "tokenizer_page.html").read_text(encoding="utf-8")
            self._send(page, "text/html; charset=utf-8")
        else:
            self._send("{}", code=404)

    def do_POST(self):
        if self.path != "/api/tokenize":
            return self._send("{}", code=404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        repo = body.get("repo") or None
        if repo and repo not in LABELS:
            return self._send(json.dumps({"error": "unknown tokenizer"}), code=400)
        self._send(json.dumps(run(body.get("text", ""), repo), ensure_ascii=False))


def main():
    global GR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    GR = Grantha()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print("serving", url, flush=True)
    if not a.no_open:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
