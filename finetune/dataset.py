"""Build a fine-tuning dataset from the app's library (DATA_DIR).

Sources
  - gold:   lines a person checked (✓ / Mark verified), from any recording
  - silver: lines from imported videos' human-made subtitles, after quality filtering

Quality filter for silver lines (subtitles can be mistimed, abridged or plain wrong):
  - 1.5 s <= duration <= 15 s and a plausible speaking rate (3..25 letters per second)
  - the aligner could place at least 75% of the words in the audio
  - subtitle and model transcript of the same audio don't disagree wildly (CER <= 60%);
    a line where they share almost nothing is usually a timing problem, not a dialect lesson

Spelling: subtitles often drop hamzas and write ه for ة. Where a subtitle word and the
model's word are the same after normalization, the model's (properly spelled) form is used,
so training teaches dialect words without teaching sloppy spelling.

Split: every recording goes wholly to train or test, except long recordings (> 30 min),
whose last 15% (by time) is held out as test so each source is also measured on unseen audio.
"""
from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import edit_distance, normalize  # noqa: E402

MIN_S, MAX_S = 1.5, 15.0
MAX_CER = 0.60
MIN_ALIGNED = 0.75


def _cer(a: str, b: str) -> float:
    a, b = normalize(a).replace(" ", ""), normalize(b).replace(" ", "")
    return edit_distance(list(a), list(b)) / max(1, len(b))


def merge_spelling(sub: str, model: str) -> str:
    """Subtitle words, but in the model's spelling wherever the two agree after normalization."""
    sw, mw = sub.split(), model.split()
    sm = SequenceMatcher(a=[normalize(w) for w in sw], b=[normalize(w) for w in mw], autojunk=False)
    out = list(sw)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = mw[j1 + k]
    return " ".join(out)


def build(data_dir: Path) -> dict:
    samples, dropped = [], {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for meta in sorted(data_dir.glob("*/meta.json")):
        it = json.loads(meta.read_text(encoding="utf-8"))
        if it.get("status") != "done" or not (meta.parent / "audio.m4a").exists():
            continue
        segs = it.get("segments", [])
        model_lines = (it.get("model_text") or "").split("\n")
        dur = it.get("duration") or (segs[-1]["end"] if segs else 0)
        long = dur > 1800
        for i, s in enumerate(segs):
            text = s["text"].strip()
            if not text:
                continue
            gold = bool(s.get("verified"))
            silver = bool(it.get("from_subtitles")) and not gold
            if not (gold or silver):
                continue
            length = s["end"] - s["start"]
            model = model_lines[i] if i < len(model_lines) else ""
            if silver:
                letters = len(normalize(text).replace(" ", ""))
                if not (MIN_S <= length <= MAX_S):
                    drop("length"); continue
                if not (3 <= letters / length <= 25):
                    drop("speaking rate"); continue
                placed = sum(1 for w in s["words"] if not w.get("guess")) / max(1, len(s["words"]))
                if placed < MIN_ALIGNED:
                    drop("not aligned"); continue
                if model and _cer(model, text) > MAX_CER:
                    drop("subtitle/audio mismatch"); continue
                target = merge_spelling(text, model) if model else text
            else:
                if length > 30:
                    drop("gold too long"); continue
                target = text
            words = target.split()
            if len(words) != len(s["words"]):  # timing list must match the words 1:1
                drop("word/timing mismatch"); continue
            test = (long and s["start"] > 0.85 * dur) or (not long and it["id"][-1] in "05")
            samples.append({
                "item": it["id"], "source": it["name"], "line": i, "kind": "gold" if gold else "silver",
                "start": s["start"], "end": s["end"], "text": target, "model_text": model,
                "word_starts": [round(w["start"] - s["start"], 3) for w in s["words"]],
                "aligner_timed": bool(it.get("from_subtitles")),
                "split": "test" if test else "train",
            })
    summary = {
        "train": sum(x["split"] == "train" for x in samples),
        "test": sum(x["split"] == "test" for x in samples),
        "train_hours": round(sum(x["end"] - x["start"] for x in samples if x["split"] == "train") / 3600, 2),
        "test_hours": round(sum(x["end"] - x["start"] for x in samples if x["split"] == "test") / 3600, 2),
        "gold": sum(x["kind"] == "gold" for x in samples),
        "dropped": dropped,
    }
    return {"samples": samples, "summary": summary}


if __name__ == "__main__":
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\AI\arabic-transcriber-data")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else data_dir / "finetune" / "dataset.json")
    ds = build(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ds, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ds["summary"], ensure_ascii=False, indent=1))
    print("written to", out)
