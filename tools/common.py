import re
import unicodedata

STRIP = dict.fromkeys(map(ord, "​‌‍﻿­"))
DEV = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9०-९]")
OTHER = re.compile(r"[^\sऀ-ॿ -~]")
SKIP = re.compile(r"[\s\W\d_]+")
WS = re.compile(r"[ \t ]+")


def normalize(line):
    line = unicodedata.normalize("NFC", line).translate(STRIP)
    return WS.sub(" ", line).strip()


def stripped(line):
    return SKIP.sub("", line)


def good_line(line):
    n = len(line)
    if n < 8:
        return False
    dev = len(DEV.findall(line))
    if dev < 0.6 * n:
        return False
    if len(LATIN.findall(line)) > 0.05 * n:
        return False
    if len(OTHER.findall(line)) > 0.05 * n:
        return False
    if len(DIGIT.findall(line)) > 0.3 * n:
        return False
    return True
