import json
import os
import sys
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")

import sentencepiece as spm
from transformers import AutoTokenizer

PORT = 8765
BUILTIN = [
    ("Bhagavad Gita 2.47", "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥"),
    ("Tat tvam asi", "तत्त्वमसि श्वेतकेतो सोऽहमस्मि ब्रह्मैवाहम्"),
    ("FLORES sentence", "राज्यपालस्य कार्यालयं न्यवेदयत् यत् आहतेषु जनेषु 19 रक्षिकाः आसन्।"),
    ("English", "The Governor's office stated that 19 guards were among the injured."),
]


def samples():
    out = []
    try:
        texts = [l.strip() for l in open("data/demo_samples.txt", encoding="utf-8") if l.strip()]
        labels = [l.split(" | ")[0].strip() for l in open("data/demo_samples_labels.txt", encoding="utf-8") if l.strip()]
        out = list(zip(labels, texts))
    except OSError:
        pass
    return out + BUILTIN


class Grantha:
    name = "Grantha 128k"
    ours = True

    def __init__(self):
        self.sp = spm.SentencePieceProcessor(model_file="data/tok_v2_dcsgr_128.model")

    def spans(self, s):
        return [(p.replace("▁", "") or "·", p.startswith("▁")) for p in self.sp.encode(s, out_type=str)]


class HF:
    ours = False

    def __init__(self, name, repo):
        self.name = name
        self.tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)

    def spans(self, s):
        out = []
        for n, i in enumerate(self.tk.encode(s, add_special_tokens=False)):
            d = self.tk.decode([i])
            out.append((d.strip(" ") or "·", n == 0 or d.startswith(" ")))
        return out


MODELS = [
    Grantha(),
    HF("Gemma 3 / 4", "google/gemma-4-12B-it"),
    HF("Param2", "bharatgenai/Param2-17B-A2.4B-Thinking"),
    HF("Qwen3.5", "Qwen/Qwen3.5-9B"),
]


def tokenize(text):
    words = max(len(text.split()), 1)
    res = []
    for m in MODELS:
        sp = m.spans(text)
        res.append({"name": m.name, "ours": m.ours, "count": len(sp), "perWord": len(sp) / words, "pieces": [{"t": t, "w": w} for t, w in sp]})
    return {"words": len(text.split()), "models": res}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grantha Tokenizer</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Tiro+Devanagari+Sanskrit&display=swap">
<style>
:root {
  --paper:#eef1f4; --surface:#fbfcfd; --ink:#14202b; --muted:#566573; --rule:#cdd5dc;
  --accent:#2e3dbb; --accent-soft:#e0e4fa; --chip-a:#dfe5ec; --chip-b:#cfd8e2; --bar:#9aa5b1;
  --display:"Bricolage Grotesque","Helvetica Neue",Arial,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --deva:"Tiro Devanagari Sanskrit","Noto Sans Devanagari","Kohinoor Devanagari",serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark; --paper:#0f151b; --surface:#17202a; --ink:#e5ebf0; --muted:#93a1ae; --rule:#2b3844;
  --accent:#929eff; --accent-soft:#232b55; --chip-a:#202c37; --chip-b:#2a3846; --bar:#566573; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 var(--body); padding-inline:20px; }
main { max-width:1080px; margin-inline:auto; padding-block:36px 72px; display:flex; flex-direction:column; gap:28px; }
h1 { font:700 clamp(2rem,5vw,2.8rem)/1.05 var(--display); letter-spacing:-.02em; margin:0; text-wrap:balance; }
.eyebrow { font:500 .75rem var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
header { display:flex; flex-direction:column; gap:10px; }
header p { margin:0; color:var(--muted); max-width:62ch; }
.panel { background:var(--surface); border:1px solid var(--rule); border-radius:6px; padding:18px; display:flex; flex-direction:column; gap:14px; }
textarea { width:100%; min-height:96px; resize:vertical; border:1px solid var(--rule); border-radius:4px; background:var(--paper); color:var(--ink);
  font:1.5rem/1.5 var(--deva); padding:12px 14px; }
textarea:focus-visible, button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.samples { display:flex; flex-wrap:wrap; gap:8px; }
.samples button { font:500 .8rem var(--body); color:var(--ink); background:var(--paper); border:1px solid var(--rule); border-radius:4px; padding:6px 10px; cursor:pointer; }
.samples button:hover { border-color:var(--accent); }
.samples button[aria-pressed="true"] { background:var(--accent-soft); border-color:var(--accent); color:var(--accent); }
.hint { font:.78rem var(--mono); color:var(--muted); }
.verdict { font:600 1.15rem/1.3 var(--display); text-wrap:balance; }
.verdict b { color:var(--accent); }
.rows { display:flex; flex-direction:column; gap:14px; }
.row { display:grid; grid-template-columns:190px 1fr; gap:6px 22px; padding:16px; background:var(--surface); border:1px solid var(--rule); border-radius:6px; }
.row.ours { border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent); }
.meta { display:flex; flex-direction:column; gap:2px; }
.name { font:600 .95rem var(--body); }
.count { font:700 2.4rem/1 var(--display); font-variant-numeric:tabular-nums; }
.row.ours .count { color:var(--accent); }
.pw { font:.8rem var(--mono); color:var(--muted); font-variant-numeric:tabular-nums; }
.body { display:flex; flex-direction:column; gap:10px; min-width:0; }
.track { height:8px; background:var(--chip-a); border-radius:2px; overflow:hidden; }
.fill { height:100%; background:var(--bar); }
.row.ours .fill { background:var(--accent); }
.chips { display:flex; flex-wrap:wrap; gap:3px; }
.chip { font:1.2rem/1.35 var(--deva); padding:1px 7px; background:var(--chip-a); border-radius:3px; white-space:pre; }
.chip:nth-child(even) { background:var(--chip-b); }
.chip.w { box-shadow:inset 3px 0 0 var(--accent); padding-left:10px; }
.note { color:var(--muted); font-size:.85rem; max-width:80ch; }
@media (max-width:720px) { .row { grid-template-columns:1fr; } .meta { flex-direction:row; align-items:baseline; gap:14px; flex-wrap:wrap; } }
</style></head>
<body><main>
<header>
  <div class="eyebrow">GranthaLLM / tokenizer</div>
  <h1>Same text, four tokenizers</h1>
  <p>Each tokenizer cuts the text into pieces from its own vocabulary. Fewer pieces means cheaper to process. A blue edge marks a piece that starts a new word.</p>
</header>
<section class="panel">
  <label class="eyebrow" for="t">Paste Sanskrit or English</label>
  <textarea id="t" lang="sa" spellcheck="false" placeholder="कर्मण्येवाधिकारस्ते मा फलेषु कदाचन"></textarea>
  <div class="samples" id="samples" role="group" aria-label="Sample texts"></div>
  <div class="hint">Samples are drawn from the local corpus and not uploaded anywhere.</div>
</section>
<section class="stack" id="out" aria-live="polite"></section>
<p class="note">Fewer tokens is cheaper to process, not necessarily better quality. Grantha is trained on Sanskrit and English only, the others are general multilingual tokenizers. A <span class="pw">&#xFFFD;</span> means a character was cut into bytes.</p>
</main>
<script>
var T = document.getElementById("t"), OUT = document.getElementById("out"), S = document.getElementById("samples");
var timer = null, seq = 0;
function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function render(data) {
  OUT.textContent = "";
  var ms = data.models, ours = ms[0], best = null;
  ms.slice(1).forEach(function (m) { if (!best || m.count < best.count) best = m; });
  var v = el("div", "verdict");
  if (ours.count > 0 && best) {
    var r = best.count / ours.count;
    v.append("Grantha: ", el("b", "", ours.count + " pieces"), ". Best of the others: " + best.name + ", " + best.count + " pieces, " + (r >= 1 ? r.toFixed(1) + " times more." : "fewer, so Grantha loses here."));
  }
  OUT.appendChild(v);
  var rows = el("div", "rows");
  rows.style.marginTop = "14px";
  var max = Math.max.apply(null, ms.map(function (m) { return m.count; })) || 1;
  ms.forEach(function (m) {
    var row = el("div", "row" + (m.ours ? " ours" : ""));
    var meta = el("div", "meta");
    meta.appendChild(el("div", "name", m.name));
    meta.appendChild(el("div", "count", String(m.count)));
    meta.appendChild(el("div", "pw", m.perWord.toFixed(2) + " per word"));
    var body = el("div", "body"), track = el("div", "track"), fill = el("div", "fill");
    fill.style.width = (m.count / max * 100) + "%";
    track.appendChild(fill);
    var chips = el("div", "chips");
    m.pieces.forEach(function (p) { chips.appendChild(el("span", "chip" + (p.w ? " w" : ""), p.t)); });
    body.appendChild(track); body.appendChild(chips);
    row.appendChild(meta); row.appendChild(body); rows.appendChild(row);
  });
  OUT.appendChild(rows);
}
function run() {
  var text = T.value.trim(), my = ++seq;
  if (!text) { OUT.textContent = ""; return; }
  fetch("/api/tokenize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: text }) })
    .then(function (r) { return r.json(); })
    .then(function (d) { if (my === seq) render(d); });
}
T.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(run, 200); });
fetch("/api/samples").then(function (r) { return r.json(); }).then(function (list) {
  list.forEach(function (s, i) {
    var b = el("button", "", (i + 1) + ". " + s.label.replace(/ \[.*\]$/, "").replace(/ \(Shankara bhashya sutra\)/, ""));
    b.type = "button";
    b.addEventListener("click", function () {
      T.value = s.text;
      S.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", "false"); });
      b.setAttribute("aria-pressed", "true");
      run();
    });
    S.appendChild(b);
  });
  if (list.length) S.firstChild.click();
});
</script></body></html>
"""


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/samples":
            self._send(json.dumps([{"label": l, "text": t} for l, t in samples()], ensure_ascii=False), "application/json; charset=utf-8")
        elif self.path in ("/", "/index.html"):
            self._send(PAGE, "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/tokenize":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        text = json.loads(self.rfile.read(n) or b"{}").get("text", "")[:4000]
        self._send(json.dumps(tokenize(text), ensure_ascii=False), "application/json; charset=utf-8")

    def log_message(self, *a):
        pass


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}/"
    print("Grantha tokenizer demo at", url, "(Ctrl-C to stop)")
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
