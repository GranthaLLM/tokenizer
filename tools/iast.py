import re
import unicodedata

VOW = {"a": "अ", "ā": "आ", "i": "इ", "ī": "ई", "u": "उ", "ū": "ऊ", "ṛ": "ऋ", "ṝ": "ॠ", "ḷ": "ऌ", "ḹ": "ॡ", "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ"}
MAT = {"a": "", "ā": "ा", "i": "ि", "ī": "ी", "u": "ु", "ū": "ू", "ṛ": "ृ", "ṝ": "ॄ", "ḷ": "ॢ", "ḹ": "ॣ", "e": "े", "ai": "ै", "o": "ो", "au": "ौ"}
CON = {
    "kh": "ख", "gh": "घ", "ch": "छ", "jh": "झ", "ṭh": "ठ", "ḍh": "ढ", "th": "थ", "dh": "ध", "ph": "फ", "bh": "भ",
    "k": "क", "g": "ग", "ṅ": "ङ", "c": "च", "j": "ज", "ñ": "ञ", "ṭ": "ट", "ḍ": "ड", "ṇ": "ण",
    "t": "त", "d": "द", "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल", "v": "व",
    "ś": "श", "ṣ": "ष", "s": "स", "h": "ह",
}
MISC = {"ṃ": "ं", "ṁ": "ं", "ḥ": "ः", "ꣳ": "ं", "m̐": "ँ", "'": "ऽ", "’": "ऽ", "ʼ": "ऽ", "|": "।", "||": "॥"}
DIG = dict(zip("0123456789", "०१२३४५६७८९"))
VIRAMA = "्"
IAST_CHARS = set("aāiīuūṛṝḷḹeoṃṁḥkgṅcjñṭḍṇtdnpbmyrlvśṣsh")
ALLOWED = IAST_CHARS | set(" \t|/-'’ʼ.,;:!?()[]0123456789\"“”‘")

_con_keys = sorted(CON, key=len, reverse=True)
_vow_keys = sorted(VOW, key=len, reverse=True)


def to_devanagari(s):
    s = unicodedata.normalize("NFC", s).lower()
    out, i, n = [], 0, len(s)
    prev_con = False
    while i < n:
        c = s[i]
        if c in "|/":
            j = i
            while j < n and s[j] in "|/":
                j += 1
            out.append("॥" if j - i >= 2 else "।")
            prev_con = False
            i = j
            continue
        if c == "-":
            i += 1
            continue
        hit = None
        for k in _con_keys:
            if s.startswith(k, i):
                hit = k
                break
        if hit:
            if prev_con:
                out.append(VIRAMA)
            out.append(CON[hit])
            prev_con = True
            i += len(hit)
            continue
        vh = None
        for k in _vow_keys:
            if s.startswith(k, i):
                vh = k
                break
        if vh:
            out.append(MAT[vh] if prev_con else VOW[vh])
            prev_con = False
            i += len(vh)
            continue
        if prev_con:
            out.append(VIRAMA)
            prev_con = False
        if c in MISC:
            out.append(MISC[c])
        elif c in DIG:
            out.append(DIG[c])
        else:
            out.append(c)
        i += 1
    if prev_con:
        out.append(VIRAMA)
    return "".join(out)


def iast_ratio(s):
    s = unicodedata.normalize("NFC", s).lower()
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in IAST_CHARS for c in letters) / len(letters)


if __name__ == "__main__":
    tests = {
        "athāgnipurāṇānukramaṇikā |": "अथाग्निपुराणानुक्रमणिका ।",
        "dharmakṣetre kurukṣetre samavetā yuyutsavaḥ ||": "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ॥",
        "tat-tvam asi": "तत्त्वम् असि".replace("तत्त्वम्", "तत्त्वम्"),
        "sa 'bravīt": "स ऽब्रवीत्",
        "kṛṣṇaḥ": "कृष्णः",
        "saṃskāra": "संस्कार",
        "ṝ": "ॠ",
    }
    for k, v in tests.items():
        got = to_devanagari(k)
        print("OK " if got == v else "BAD", k, "->", got, "" if got == v else f"(want {v})")
