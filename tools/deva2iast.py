import unicodedata

CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ", "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ", "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "ś", "ष": "ṣ", "स": "s", "ह": "h", "ळ": "ḷ"}
IV = {"अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū", "ऋ": "ṛ", "ॠ": "ṝ", "ऌ": "ḷ", "ॡ": "ḹ", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au"}
MAT = {"ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū", "ृ": "ṛ", "ॄ": "ṝ", "ॢ": "ḷ", "ॣ": "ḹ", "े": "e", "ै": "ai", "ो": "o", "ौ": "au"}
MISC = {"ं": "ṃ", "ः": "ḥ", "ँ": "m̐", "ऽ": "'", "।": "|", "॥": "||", "्": ""}
DIG = dict(zip("०१२३४५६७८९", "0123456789"))


def to_iast(s):
    s = unicodedata.normalize("NFC", s)
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in CONS:
            nxt = s[i + 1] if i + 1 < n else ""
            out.append(CONS[c])
            if nxt == "्":
                i += 2
                continue
            if nxt in MAT:
                out.append(MAT[nxt])
                i += 2
                continue
            out.append("a")
            i += 1
        elif c in IV:
            out.append(IV[c])
            i += 1
        elif c in MISC:
            out.append(MISC[c])
            i += 1
        elif c in DIG:
            out.append(DIG[c])
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


if __name__ == "__main__":
    for t in ["सृष्टिस्थित्यप्ययेहानियतिदृशितमोबन्धमोक्षाश्च", "कर्मण्येवाधिकारस्ते ॥"]:
        print(to_iast(t))
