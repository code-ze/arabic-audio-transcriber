"""Learned fixes: corrections you make by hand that the app can apply automatically.

Every word correction is counted as (model wrote X -> you changed it to Y).
When the same change has been made SUGGEST_AFTER times it is offered as a rule;
enabled rules are applied to every new transcript (and, on request, to old ones).

Stored in DATA_DIR/fixes.json:
  {"counts": {"X\tY": n}, "rules": [{"from": X, "to": Y}], "dismissed": ["X\tY"]}
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

SUGGEST_AFTER = 2
# punctuation that may stick to a word's edges; rules match the word without it
EDGE = ".,،؛;:!?؟\"'«»()[]…-"


def split_edges(word: str) -> tuple[str, str, str]:
    """'(سيدي،' -> ('(', 'سيدي', '،')"""
    core = word.strip(EDGE)
    if not core:
        return "", word, ""
    i = word.find(core)
    return word[:i], core, word[i + len(core):]


class Fixes:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data = {"counts": {}, "rules": [], "dismissed": []}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    # ---- learning ----
    def record(self, before: str, after: str) -> None:
        """Count one manual correction of a single word."""
        b, a = split_edges(before)[1], " ".join(split_edges(w)[1] for w in after.split())
        if not b or not a or b == a:
            return
        key = f"{b}\t{a}"
        with self.lock:
            self.data["counts"][key] = self.data["counts"].get(key, 0) + 1
            self._save()

    def rule_map(self) -> dict[str, str]:
        return {r["from"]: r["to"] for r in self.data["rules"]}

    def suggestions(self) -> list[dict]:
        rules, dismissed = self.rule_map(), set(self.data["dismissed"])
        out = []
        for key, n in self.data["counts"].items():
            frm, to = key.split("\t", 1)
            if n >= SUGGEST_AFTER and key not in dismissed and rules.get(frm) != to:
                out.append({"from": frm, "to": to, "count": n})
        return sorted(out, key=lambda s: -s["count"])

    def view(self) -> dict:
        counts = self.data["counts"]
        return {
            "suggestions": self.suggestions(),
            "rules": [r | {"count": counts.get(f"{r['from']}\t{r['to']}", 0)} for r in self.data["rules"]],
        }

    # ---- managing rules ----
    def add_rule(self, frm: str, to: str) -> None:
        frm, to = split_edges(frm.strip())[1], " ".join(to.split())
        if not frm or not to or frm == to:
            raise ValueError("A fix needs a word and a different replacement.")
        with self.lock:
            self.data["rules"] = [r for r in self.data["rules"] if r["from"] != frm] + [{"from": frm, "to": to}]
            self._save()

    def remove_rule(self, frm: str) -> None:
        with self.lock:
            self.data["rules"] = [r for r in self.data["rules"] if r["from"] != frm]
            self._save()

    def dismiss(self, frm: str, to: str) -> None:
        with self.lock:
            key = f"{frm}\t{to}"
            if key not in self.data["dismissed"]:
                self.data["dismissed"].append(key)
            self._save()

    # ---- applying ----
    def apply(self, item: dict) -> int:
        """Apply enabled rules to an item's words in place. Returns how many words changed.
        A rule may turn one word into several; the word's time span is shared between them."""
        rules = self.rule_map()
        if not rules:
            return 0
        changed = 0
        for seg in item.get("segments", []):
            out = []
            for w in seg["words"]:
                pre, core, post = split_edges(w["text"])
                to = rules.get(core)
                if to is None or w.get("fixed"):  # never override a manual correction
                    out.append(w)
                    continue
                parts = to.split()
                parts[0], parts[-1] = pre + parts[0], parts[-1] + post
                out += spread(w, parts, auto=True)
                changed += 1
            seg["words"] = out
            seg["text"] = " ".join(x["text"] for x in out)
        return changed


def realign(old: list[dict], new_text: str, seg_start: float, seg_end: float) -> tuple[list[dict], list[tuple[str, str]]]:
    """Give a retyped line word timings by lining it up with the old timed words.

    Unchanged words keep their exact times. Replaced/inserted words share the time
    between their unchanged neighbours. Returns (new words, one-for-one word swaps)
    so swaps can be counted as learned fixes.
    """
    from difflib import SequenceMatcher

    new = new_text.split()
    sm = SequenceMatcher(a=[split_edges(w["text"])[1] for w in old],
                         b=[split_edges(w)[1] for w in new], autojunk=False)
    out: list[dict] = []
    swaps: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                w = dict(old[i1 + k])
                w["text"] = new[j1 + k]  # punctuation may have changed
                if w["text"] != old[i1 + k]["text"]:
                    w["fixed"] = True
                out.append(w)
            continue
        if op == "replace" and i2 - i1 == j2 - j1:
            swaps += [(old[i1 + k]["text"], new[j1 + k]) for k in range(i2 - i1)]
        if j1 == j2:  # deletion
            continue
        # time window: the replaced words' span, else the gap between neighbours
        if i2 > i1:
            s, e = old[i1]["start"], old[i2 - 1]["end"]
        else:
            s = out[-1]["end"] if out else (old[i1 - 1]["end"] if i1 > 0 else seg_start)
            e = old[i1]["start"] if i1 < len(old) else seg_end
        if e - s < 0.08 * (j2 - j1):  # no room: borrow a little time after
            e = min(seg_end, s + 0.25 * (j2 - j1))
            s = min(s, e - 0.08 * (j2 - j1))
        out += spread({"start": s, "end": e}, new[j1:j2], fixed=True)
    return out, swaps


def spread(word: dict, parts: list[str], **flags) -> list[dict]:
    """Replace one timed word with len(parts) words sharing its time span evenly."""
    s, e = word["start"], word["end"]
    step = (e - s) / len(parts)
    return [{"start": round(s + i * step, 2), "end": round(s + (i + 1) * step, 2), "text": p, **flags}
            for i, p in enumerate(parts)]
