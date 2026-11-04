import os
import re
import select
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(__file__))
from demo_web import MODELS
from rich.console import Console, Group
from rich.rule import Rule
from rich.text import Text

ACCENT = "#8b9bff"
GREY = "#7d8a97"
CHIP = ["on #26323f", "on #1d2731"]
con = Console()


PASTE_MARKS = re.compile(r"\x1b?\[20[01]~")


def label(piece):
    if piece and unicodedata.category(piece[0]).startswith("M"):
        return "◌" + piece
    return piece


def clean(line):
    line = PASTE_MARKS.sub("", line)
    line = "".join(c for c in line if c == "\t" or unicodedata.category(c) != "Cc")
    return unicodedata.normalize("NFC", line)


def chips(spans, ours, offset=0):
    t = Text()
    for i, (piece, start) in enumerate(spans, offset):
        if start:
            t.append("▏", style=ACCENT if ours else GREY)
        else:
            t.append(" ")
        t.append(f" {label(piece)}  ", style=CHIP[i % 2] + (" bold" if ours else ""))
    return t


def out(text, delay=0):
    con.print(text, end="", soft_wrap=True)
    if delay:
        con.file.flush()
        time.sleep(delay)


def render(text):
    words = max(len(text.split()), 1)
    results = [(m, m.spans(text)) for m in MODELS]
    mx = max(len(sp) for _, sp in results) or 1
    fast = sys.stdout.isatty()
    con.clear()
    con.print(Rule(Text("shloka"), style=ACCENT, align="left"))
    con.print(Text("▌ ", style=ACCENT) + Text(text, style="bold"), soft_wrap=True)
    con.print(Text(f"  {words} words", style=GREY))
    con.print()
    for m, sp in results:
        colour = ACCENT if m.ours else GREY
        con.print(Rule(Text(m.name), style=colour, align="left"))
        if fast:
            time.sleep(0.2)
        step = min(0.07, 0.7 / max(len(sp), 1)) if fast else 0
        for i, (piece, start) in enumerate(sp):
            out(chips([(piece, start)], m.ours, i), step)
        con.print()
        head = Text()
        head.append(f"{len(sp)} tokens", style=f"bold {ACCENT}" if m.ours else "bold")
        head.append(f"   {len(sp)/words:.2f} per word   ", style=GREY)
        out(head)
        n = max(1, round(len(sp) / mx * 40))
        for _ in range(n):
            out(Text("█", style=colour), 0.008 if fast else 0)
        con.print()
        con.print()
    ours = len(results[0][1])
    best = min((r for r in results if not r[0].ours), key=lambda r: len(r[1]))
    if ours:
        ratio = len(best[1]) / ours
        tail = f"{ratio:.1f} times more" if ratio >= 1 else "fewer, so Grantha loses here"
        v = Text()
        v.append("Grantha ", style="bold")
        v.append(f"{ours} pieces", style=f"bold {ACCENT}")
        v.append(f".  Best of the others: {best[0].name}, {len(best[1])} pieces, {tail}.")
        con.print(v)
    con.print(Text("fewer tokens is cheaper, not necessarily better quality. Grantha is trained on Sanskrit and English only.", style=GREY))


def read_paste():
    first = input()
    lines = [first]
    while select.select([sys.stdin], [], [], 0.08)[0]:
        ln = sys.stdin.readline()
        if not ln:
            break
        lines.append(ln.rstrip("\n"))
    return " ".join(clean(x).strip() for x in lines if clean(x).strip())


def main():
    if len(sys.argv) > 1:
        render(" ".join(sys.argv[1:]))
        return
    con.clear()
    con.print(Text("Grantha tokenizer", style=f"bold {ACCENT}"))
    con.print(Text("Paste text and press Enter. q quits.", style=GREY))
    while True:
        con.print(Text("\n> ", style=f"bold {ACCENT}"), end="")
        try:
            line = read_paste()
        except EOFError:
            break
        if line.lower() == "q":
            break
        if line:
            render(line)


if __name__ == "__main__":
    main()
