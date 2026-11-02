from pathlib import Path

R = Path("results")
ORDER = [
    ("Grantha 128k", "tok_v2_dcsgr_128", "dcsgr_128", True),
    ("Param2", "param2", "param2", False),
    ("Gemma 3 / 4", "gemma4", "gemma4", False),
    ("Mistral Large 4", "mistral-large-4", "mistral4", False),
    ("Sarvam-1", "sarvam1", "sarvam1", False),
    ("Llama-4", "llama4", "llama4", False),
    ("Qwen3.5", "qwen3.5", "qwen3.5", False),
]


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def flores():
    out = {}
    for ln in (R / "fertility_bilingual.md").read_text().splitlines()[2:]:
        c = cells(ln)
        out[c[0]] = float(c[1])
    return out


def corpus():
    lines = (R / "domain_check.md").read_text().splitlines()
    head, allrow = cells(lines[0]), cells([l for l in lines if l.startswith("| **all**")][0])
    return {h.replace(" san", ""): float(v) for h, v in zip(head, allrow) if h.endswith(" san")}


def classical():
    d, g = {}, {}
    for ln in (R / "classical_eval.md").read_text().splitlines()[2:]:
        c = cells(ln)
        d[c[0]], g[c[0]] = float(c[1]), float(c[2])
    return d, g


f, c, (d, g) = flores(), corpus(), classical()
PANELS = [
    ("FLORES-200 devtest", [f[k] for _, k, _, _ in ORDER]),
    ("41 corpus works", [c[k] for _, _, k, _ in ORDER]),
    ("DCS held-out works", [d[k] for _, k, _, _ in ORDER]),
    ("GRETIL held-out works", [g[k] for _, k, _, _ in ORDER]),
]
PW, PH, LABEL, GAP, MAX, ROW = 360, 0, 98, 40, 6.0, 22
TOP = 66
panel_h = 30 + ROW * len(ORDER) + 26
W = 2 * PW + GAP + 40
H = TOP + 2 * panel_h + 20 + 62
out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Sanskrit tokens per word on four benchmarks, Grantha 128k against six other tokenizers">',
    "<style>",
    "text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;font-size:12.5px;fill:#1f2328}",
    ".mut{fill:#59636e}.grid{stroke:#d1d9e0;stroke-width:1}.us{fill:#3d4fc4}.alt{fill:#8d99dd}.them{fill:#9aa5b1}.b{font-weight:600}.t{font-size:15px;font-weight:600}.h{font-size:13px;font-weight:600}",
    "@media (prefers-color-scheme:dark){text{fill:#e6edf3}.mut{fill:#9198a1}.grid{stroke:#3d444d}.us{fill:#8b9bff}.alt{fill:#5a64a8}.them{fill:#59636e}}",
    "</style>",
    f'<text x="20" y="24" class="t">Tokens per Sanskrit word (cost, lower is better)</text>',
    f'<text x="20" y="44" class="mut">Grantha: 128k joint Sanskrit and English tokenizer.</text>',
]
for pi, (title, vals) in enumerate(PANELS):
    ox = 20 + (pi % 2) * (PW + GAP)
    oy = TOP + (pi // 2) * panel_h
    bx0, bw = ox + LABEL, PW - LABEL - 44

    def x(v):
        return bx0 + bw * v / MAX

    out.append(f'<text x="{ox}" y="{oy+12}" class="h">{title}</text>')
    for t in range(0, 7, 2):
        out.append(f'<line x1="{x(t):.1f}" y1="{oy+22}" x2="{x(t):.1f}" y2="{oy+22+ROW*len(ORDER)}" class="grid"/>')
        out.append(f'<text x="{x(t):.1f}" y="{oy+22+ROW*len(ORDER)+14}" text-anchor="middle" class="mut">{t}</text>')
    for i, ((name, _, _, kind), v) in enumerate(zip(ORDER, vals)):
        y = oy + 26 + i * ROW
        out.append(f'<text x="{bx0-8}" y="{y+11}" text-anchor="end"{" class=\"b\"" if kind is True else ""}>{name}</text>')
        out.append(f'<rect x="{bx0}" y="{y}" width="{x(v)-bx0:.1f}" height="14" class="{"us" if kind is True else "alt" if kind == "alt" else "them"}"/>')
        out.append(f'<text x="{x(v)+5:.1f}" y="{y+11}" class="mut">{v:.2f}</text>')
notes = [
    "Grantha is trained on Sanskrit and English only. The others are general multilingual tokenizers.",
    "On DCS and GRETIL, Grantha also trained on other works from those collections.",
    "Fewer tokens is cheaper to process, not necessarily better quality. No model has been trained on it yet.",
]
for i, n in enumerate(notes):
    out.append(f'<text x="20" y="{H-48+i*16}" class="mut">{n}</text>')
out.append("</svg>")
(R / "fertility.svg").write_text("\n".join(out) + "\n")
print(PANELS)
