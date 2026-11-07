import csv
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pickle

from akshara import bounds
from iast import to_devanagari
from uoh_parse import gold

PORT = 8766
SRC = Path("data/vedanta_to_label.csv")
OUT = Path("data/vedanta_labels.json")


def byt5_parts():
    labels = {}
    for p in ("data/teacher_labels.pkl", "data/resolve_labels.pkl"):
        if Path(p).exists():
            labels.update(pickle.load(open(p, "rb")))
    return labels


def parts_guess(labels, w, model_split):
    o = labels.get(w)
    if o:
        comps = [to_devanagari(c) for c in o.replace(" ", "_").split("_") if c]
        if comps:
            return "+".join(comps)
    return "+".join(model_split.split(" | "))


def preview(word, parts):
    comps = [c.strip() for c in parts.replace(" ", "+").split("+") if c.strip()]
    if len(comps) < 2:
        return {"cuts": [], "cands": []}
    targets, cands = gold(word, comps)
    return {"cuts": targets, "cands": cands}


def load_words():
    labels = byt5_parts()
    rows = []
    for r in csv.DictReader(open(SRC, encoding="utf-8")):
        w = r["word"]
        bs = bounds(w)
        aks = [w[a:b] for a, b in zip(bs, bs[1:])]
        model, pos = set(), 0
        for part in r["model_split"].split(" | ")[:-1]:
            pos += len(part)
            model.add(pos)
        gaps = [i for i, b in enumerate(bs[1:-1]) if b in model]
        rows.append({"word": w, "aksharas": aks, "model": gaps, "work": r["work"], "parts": parts_guess(labels, w, r["model_split"]), "bounds": bs})
    return rows


def load_labels():
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Compound labeling</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Tiro+Devanagari+Sanskrit&display=swap">
<style>
:root{--paper:#eef1f4;--surface:#fbfcfd;--ink:#14202b;--muted:#566573;--rule:#cdd5dc;--accent:#2e3dbb;--accent-soft:#e0e4fa;--ok:#1d7a4f;--ok-soft:#dcefe5;
--display:"Bricolage Grotesque","Helvetica Neue",Arial,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;--deva:"Tiro Devanagari Sanskrit","Noto Sans Devanagari","Kohinoor Devanagari",serif}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--paper:#0f151b;--surface:#17202a;--ink:#e5ebf0;--muted:#93a1ae;--rule:#2b3844;--accent:#929eff;--accent-soft:#232b55;--ok:#6fd2a0;--ok-soft:#183a2b}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 var(--body);padding-inline:20px}
main{max-width:900px;margin-inline:auto;padding-block:32px 64px;display:flex;flex-direction:column;gap:22px}
h1{font:700 1.7rem/1.1 var(--display);margin:0;letter-spacing:-.01em}
.eyebrow{font:500 .75rem var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.bar{height:6px;background:var(--rule);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font:.8rem var(--mono);color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:26px 22px;display:flex;flex-direction:column;gap:18px}
.word{display:flex;flex-wrap:wrap;align-items:center;row-gap:10px;font:2.2rem/1.3 var(--deva)}
.ak{padding:2px 1px}
.gap{width:18px;align-self:stretch;display:flex;align-items:center;justify-content:center;cursor:pointer;border-radius:3px;color:transparent;user-select:none;font:500 1.6rem var(--mono)}
.gap:hover{background:var(--accent-soft);color:var(--accent)}
.gap.on{color:var(--accent);background:var(--accent-soft)}
.result{font:1.25rem/1.5 var(--deva);color:var(--ink)}.parts{width:100%;font:1.5rem/1.4 var(--deva);padding:10px 14px;border:1px solid var(--rule);border-radius:4px;background:var(--paper);color:var(--ink)}
.result b{color:var(--accent);font-weight:400;padding-inline:.35em;font-family:var(--mono)}
.hint{color:var(--muted);font-size:.9rem;max-width:70ch}
.row{display:flex;flex-wrap:wrap;gap:10px}
button{font:500 .95rem var(--body);color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:4px;padding:9px 16px;cursor:pointer}
button:hover{border-color:var(--accent)}button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
@media (prefers-color-scheme:dark){button.primary{color:#0f151b}}
kbd{font:500 .75rem var(--mono);border:1px solid var(--rule);border-radius:3px;padding:1px 5px;margin-left:6px;opacity:.8}
.tag{font:.78rem var(--mono);padding:2px 8px;border-radius:3px;background:var(--ok-soft);color:var(--ok)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style></head>
<body><main>
<header style="display:flex;flex-direction:column;gap:10px">
 <div class="eyebrow">GranthaLLM / Vedanta compound set</div>
 <h1>Where does each part end?</h1>
 <p class="hint">Type the real words in the box, separated by +, as you would write them on their own (for example पूर्वम्+ऋजोः+ब्रह्मत्वतः for पूर्वमृजोर्ब्रह्मत्वतः). The page works out where they sit in the written letters. Where sandhi fused a letter into its neighbour, either side counts as right. You can also click a gap to cut or join by hand. The box starts with ByT5-Sanskrit's guess.</p>
</header>
<div class="bar"><i id="prog"></i></div>
<div class="meta"><span id="count"></span><span id="work"></span><span id="state"></span></div>
<section class="card">
 <div class="word" id="word" lang="sa"></div>
 <div class="result" id="result" lang="sa"></div>
 <label class="eyebrow" for="parts">Real words, separated by +</label>
 <input id="parts" class="parts" lang="sa" autocomplete="off" spellcheck="false">
 <div class="hint" id="note"></div>
 <div class="row">
  <button class="primary" id="ok">Save and next<kbd>Enter</kbd></button>
  <button id="none">No split, one word<kbd>N</kbd></button>
  <button id="reset">Reset to model guess<kbd>R</kbd></button>
  <button id="skip">Skip, unsure<kbd>S</kbd></button>
  <button id="back">Back<kbd>B</kbd></button>
 </div>
</section>
<p class="hint">Labels save to <code>data/vedanta_labels.json</code> after every action, so you can close this page and continue later.</p>
</main>
<script>
var W = [], L = {}, i = 0, cuts = new Set(), cands = null, timer = null, manual = false;
function el(t,c,x){var e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e}
function draw(){
  var w=W[i], box=document.getElementById("word"); box.textContent="";
  w.aksharas.forEach(function(a,k){
    box.appendChild(el("span","ak",a));
    if(k<w.aksharas.length-1){
      var g=el("span","gap"+(cuts.has(k)?" on":""),"|"); g.dataset.k=k;
      g.addEventListener("click",function(){var n=+this.dataset.k; if(cuts.has(n))cuts.delete(n); else cuts.add(n); cands=null; manual=true; draw();});
      box.appendChild(g);
    }
  });
  var parts=[],cur="";
  w.aksharas.forEach(function(a,k){cur+=a; if(cuts.has(k)){parts.push(cur);cur="";}});
  parts.push(cur);
  var r=document.getElementById("result"); r.textContent="";
  parts.forEach(function(p,k){ if(k)r.appendChild(el("b","","+")); r.appendChild(document.createTextNode(p)); });
  var done=Object.keys(L).length;
  document.getElementById("count").textContent="word "+(i+1)+" of "+W.length+", "+done+" saved";
  document.getElementById("work").textContent="from "+w.work;
  var st=document.getElementById("state"); st.textContent=""; var s=L[w.word];
  if(s){st.appendChild(el("span","tag",s.status));}
  document.getElementById("prog").style.width=(done/W.length*100)+"%";
}
function gapsOf(w,pos){return pos.map(function(p){return w.bounds.indexOf(p)-1;}).filter(function(g){return g>=0;});}
function preview(){
  var w=W[i], t=document.getElementById("parts").value;
  return fetch("/api/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({word:w.word,parts:t})}).then(function(r){return r.json();}).then(function(p){
    cuts=new Set(gapsOf(w,p.cuts)); cands=p.cands.length?p.cands:null;
    var fused=p.cands.filter(function(c){return c.length>1;}).length;
    document.getElementById("note").textContent=p.cuts.length?(p.cuts.length+" cut(s)"+(fused?", "+fused+" inside a fused letter, so either side counts":"")):"no cuts";
    draw();
  });
}
function load(){
  var w=W[i], s=L[w.word];
  document.getElementById("parts").value=s&&s.parts!==undefined?s.parts:w.parts;
  if(s&&s.gaps&&s.manual){cuts=new Set(s.gaps);cands=null;draw();document.getElementById("note").textContent="hand-set cuts";}
  else preview();
}
document.getElementById("parts").addEventListener("input",function(){clearTimeout(timer);timer=setTimeout(preview,200);});
function charPos(w,gaps){var p=0,out=[];w.aksharas.forEach(function(a,k){p+=a.length; if(gaps.indexOf(k)>=0)out.push(p);});return out;}
function save(status,gaps){
  var w=W[i], pos=charPos(w,gaps), entry={status:status,gaps:gaps,cuts:pos,cands:cands||pos.map(function(p){return [p];}),parts:document.getElementById("parts").value,manual:manual};
  L[w.word]=entry;
  return fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({word:w.word,entry:entry})}).then(function(){ manual=false; if(i<W.length-1){i++;} load(); });
}
document.getElementById("ok").onclick=function(){save("checked",Array.from(cuts).sort(function(a,b){return a-b;}));};
document.getElementById("none").onclick=function(){cuts=new Set();cands=null;document.getElementById("parts").value=W[i].word;save("no split",[]);};
document.getElementById("reset").onclick=function(){manual=false;document.getElementById("parts").value=W[i].parts;preview();};
document.getElementById("skip").onclick=function(){save("skipped",[]);};
document.getElementById("back").onclick=function(){if(i>0){i--;load();}};
document.addEventListener("keydown",function(e){
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  var k=e.key.toLowerCase();
  if(document.activeElement&&document.activeElement.id==="parts"){ if(e.key==="Enter"){document.getElementById("ok").click();e.preventDefault();} return; }
  if(e.key==="Enter"){document.getElementById("ok").click();e.preventDefault();}
  else if(k==="n")document.getElementById("none").click();
  else if(k==="r")document.getElementById("reset").click();
  else if(k==="s")document.getElementById("skip").click();
  else if(k==="b"||e.key==="ArrowLeft")document.getElementById("back").click();
});
Promise.all([fetch("/api/words").then(function(r){return r.json();}),fetch("/api/labels").then(function(r){return r.json();})]).then(function(x){
  W=x[0]; L=x[1];
  var first=W.findIndex(function(w){return !L[w.word];}); i=first<0?0:first; load();
});
</script></body></html>
"""


class H(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/words":
            self._send(json.dumps(WORDS, ensure_ascii=False))
        elif self.path == "/api/labels":
            self._send(json.dumps(load_labels(), ensure_ascii=False))
        elif self.path in ("/", "/index.html"):
            self._send(PAGE, "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        msg = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/preview":
            self._send(json.dumps(preview(msg["word"], msg["parts"]), ensure_ascii=False))
            return
        if self.path != "/api/save":
            self.send_error(404)
            return
        labels = load_labels()
        labels[msg["word"]] = msg["entry"]
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(OUT)
        self._send("{}")

    def log_message(self, *a):
        pass


WORDS = load_words()


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"{len(WORDS)} words, labeling page at {url} (Ctrl-C to stop)")
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == "__main__":
    main()
