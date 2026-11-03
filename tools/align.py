def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def junctions(surf, comps):
    comps = list(comps)
    pos, out = 0, []
    for i in range(len(comps) - 1):
        c = comps[i]
        k = lcp(surf[pos:], c)
        changed = k < len(c)
        j = pos + k
        n = comps[i + 1]
        best = (0, 0, 0)
        for d in range(0, 4):
            for s in range(0, 3):
                if s >= len(n):
                    continue
                m = lcp(surf[j + d :], n[s:])
                if m > best[0]:
                    best = (m, d, s)
        m, d, s = best
        if m < min(2, max(len(n) - s, 1)):
            d, s = 0, 0
        comps[i + 1] = n[s:]
        if 0 < j < len(surf):
            out.append((j, changed))
        pos = j + d
    return out


def pick(pos, bs, bset, used, changed, n):
    if pos in bset:
        t, alts = pos, {pos}
    else:
        lo = max(b for b in bs if b <= pos)
        hi = min(b for b in bs if b > pos)
        t, alts = lo, {lo, hi}
        if lo in used and hi not in used:
            t = hi
    if changed:
        nxt = [b for b in bs if b > t]
        if nxt:
            alts.add(nxt[0])
    return t, {a for a in alts if 0 < a < n}
