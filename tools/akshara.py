import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cbpe import AKSHARA


def bounds(s):
    out, pos = [0], 0
    for m in AKSHARA.finditer(s):
        for k in range(pos + 1, m.start() + 1):
            out.append(k)
        if m.end() > m.start():
            out.append(m.end())
        pos = m.end()
    for k in range(pos + 1, len(s) + 1):
        out.append(k)
    return sorted(set(out))
