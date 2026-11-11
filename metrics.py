"""Accuracy of the model on your verified lines: word and character error rates.

Text is normalized the usual way for Arabic ASR scoring (diacritics and tatweel
removed, alef/hamza forms collapsed, punctuation dropped) so spelling variants
that sound the same don't count as errors.
"""
from __future__ import annotations

import re

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # harakat, tatweel
_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalize(text: str) -> str:
    t = _DIACRITICS.sub("", text)
    t = re.sub("[أإآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    t = _PUNCT.sub(" ", t.lower())
    return " ".join(t.split())


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def score(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (model output, corrected text). Corpus-level WER/CER in percent."""
    we = wn = ce = cn = 0
    for hyp, ref in pairs:
        h, r = normalize(hyp), normalize(ref)
        we += edit_distance(h.split(), r.split())
        wn += len(r.split())
        ce += edit_distance(list(h.replace(" ", "")), list(r.replace(" ", "")))
        cn += len(r.replace(" ", ""))
    return {"wer": round(100 * we / wn, 1) if wn else None,
            "cer": round(100 * ce / cn, 1) if cn else None,
            "words": wn}
