"""Local web app: upload or record Arabic voice notes, get text back with
word-by-word playback, a searchable history, editing and subtitle export.

Run:  start.bat   (or: python app.py)   then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

import importer
from aligner import Aligner
from fixes import Fixes, merge_alignment, realign, spread, timed_from_alignment
from metrics import score
from transcriber import SR, Transcriber, load_audio

HERE = Path(__file__).parent
# Every transcript lives in DATA_DIR/<id>/ : meta.json + audio.m4a
DATA_DIR = Path(os.environ.get("DATA_DIR", HERE / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
fixes = Fixes(DATA_DIR / "fixes.json")

app = FastAPI()
engine = Transcriber()
aligner = Aligner()  # loaded on the first re-sync
items: dict[str, dict] = {}
work: "queue.Queue[tuple[str, str]]" = queue.Queue()  # ("transcribe" | "align", item id)
align_queued: set[str] = set()
model_state = {"status": "loading", "error": None}
save_lock = threading.Lock()


# ---------- storage ----------

def item_dir(iid: str) -> Path:
    return DATA_DIR / iid


def save(item: dict) -> None:
    with save_lock:
        d = item_dir(item["id"])
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, d / "meta.json")


def load_all() -> None:
    for meta in DATA_DIR.glob("*/meta.json"):
        try:
            item = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items[item["id"]] = item
        segs = item.get("segments", [])
        if item.get("verified") and segs and not any("verified" in s for s in segs):
            for s in segs:  # notes verified before lines could be checked one by one
                s["verified"] = True
            save(item)
        if item["status"] in ("queued", "working", "downloading"):
            if item.get("source_url") and not find_source(item["id"]):
                item.update(status="queued", progress=0.0)  # import interrupted: start the download again
                work.put(("import", item["id"]))
            elif find_source(item["id"]):
                item.update(status="queued", progress=0.0)
                work.put(("transcribe", item["id"]))
            else:
                item.update(status="error", error="Interrupted before it finished; please upload again.")
            save(item)
        elif any(s.get("pending_align") for s in item.get("segments", [])):
            queue_align(item["id"])  # finish re-syncs interrupted by a restart


def find_source(iid: str) -> Path | None:
    return next((p for p in item_dir(iid).glob("source.*") if p.suffix not in (".vtt", ".part", ".ytdl")), None)


def full_text(item: dict) -> str:
    return "\n".join(s["text"] for s in item.get("segments", []))


def fmt_time(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ---------- worker ----------

def worker() -> None:
    try:
        engine.load()
        model_state["status"] = "ready"
    except Exception as e:  # noqa: BLE001
        model_state.update(status="error", error=str(e))
        return
    while True:
        kind, iid = work.get()
        try:
            if kind == "align":
                align_item(iid)
            elif kind == "import":
                import_item(iid)
            else:
                transcribe_item(iid)
        except Exception as e:  # noqa: BLE001  never let one job kill the worker
            print(f"{kind} job for {iid} failed: {e}")
        finally:
            # give cached GPU memory back between jobs so other apps (and training) can use it
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def queue_align(iid: str) -> None:
    """Re-sync edited lines to the audio in the background (batched per transcript)."""
    if iid not in align_queued:
        align_queued.add(iid)
        work.put(("align", iid))


def align_item(iid: str) -> None:
    align_queued.discard(iid)
    item = items.get(iid)
    if not item:
        return
    todo = [(i, s) for i, s in enumerate(item.get("segments", [])) if s.get("pending_align")]
    if not todo:
        return
    audio = load_audio(str(item_dir(iid) / "audio.m4a"))
    for i, seg in todo:
        text = seg["text"]
        spans = aligner.align(audio[int(seg["start"] * SR): int(seg["end"] * SR)], text)
        if seg["text"] != text:  # edited again meanwhile: a newer job will handle it
            continue
        off = seg["start"]
        aligned = [(round(x[0] + off, 2), round(x[1] + off, 2)) if x else None for x in spans]
        seg["words"] = merge_alignment(seg["words"], aligned, seg["end"])
        seg.pop("pending_align", None)
    save(item)


def import_item(iid: str) -> None:
    """Download from the item's link, then transcribe (from human subtitles when the video has them)."""
    item = items.get(iid)
    if not item:
        return
    item.update(status="downloading", progress=0.0)
    save(item)
    t0 = time.time()
    try:
        got = importer.fetch(item["source_url"], item_dir(iid), progress=lambda p: item.update(progress=p))
        item["name"] = got["title"][:200]
        cues = importer.parse_vtt(got["subtitles"]) if got["subtitles"] else []
        if got["subtitles"]:
            got["subtitles"].unlink(missing_ok=True)
        if cues:
            transcribe_from_subtitles(item, got["source"], importer.group_cues(cues), t0)
            return
        item["status"] = "queued"
        save(item)
    except Exception as e:  # noqa: BLE001
        item.update(status="error", error=("Couldn't import this link: " + str(e))[-500:],
                    seconds=round(time.time() - t0, 1))
        save(item)
        return
    transcribe_item(iid)  # no human subtitles: transcribe with the model as usual


def make_playback_copy(src: Path, iid: str) -> None:
    """Compact AAC copy for the player (plays in every browser)."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
         "-ac", "1", "-c:a", "aac", "-b:a", "64k", str(item_dir(iid) / "audio.m4a")],
        check=True, capture_output=True,
    )


def transcribe_from_subtitles(item: dict, src: Path, groups: list, t0: float) -> None:
    """Lines come from the video's human subtitles; the aligner times each word, and the
    model's own transcript of every line is kept so accuracy can be measured."""
    iid = item["id"]
    item.update(status="working", progress=0.0)
    save(item)
    try:
        audio = load_audio(str(src))
        make_playback_copy(src, iid)
        dur = len(audio) / SR
        groups = [(max(0.0, s - 0.15), min(dur, e + 0.15), t) for s, e, t in groups if s < dur]
        spans = [(int(s * SR), int(e * SR)) for s, e, _ in groups]
        n = len(spans)

        def prog(done, total):
            item["progress"] = 0.9 * done / total

        model = engine.run_spans(audio, spans, prog)
        segments = []
        for k, ((s, e, text), (mtext, _)) in enumerate(zip(groups, model)):
            aligned = aligner.align(audio[int(s * SR): int(e * SR)], text)
            segments.append({"start": round(s, 2), "end": round(e, 2), "text": text,
                             "words": timed_from_alignment(text, aligned, s, e), "model": mtext})
            item["progress"] = 0.9 + 0.1 * (k + 1) / n
        item["segments"] = segments
        item["model_text"] = "\n".join(seg.pop("model") for seg in segments)
        item.update(duration=round(dur, 2), from_subtitles=True, status="done")
        src.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        item.update(status="error", error=str(e)[-500:])
    finally:
        item["seconds"] = round(time.time() - t0, 1)
        save(item)


def transcribe_item(iid: str) -> None:
    item = items.get(iid)
    src = find_source(iid)
    if not item or not src:
        return
    item.update(status="working", progress=0.0)
    save(item)
    t0 = time.time()
    try:
        def prog(done, total):
            item["progress"] = done / total

        audio = load_audio(str(src))
        make_playback_copy(src, iid)
        segs = engine.transcribe(str(src), progress=prog, audio=audio)
        item["segments"] = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text,
             "words": [{"start": w.start, "end": w.end, "text": w.text} for w in s.words]}
            for s in segs
        ]
        item["duration"] = round(len(audio) / SR, 2)
        item["model_text"] = full_text(item)  # untouched model output, kept for measuring accuracy later
        item["auto_fixes"] = fixes.apply(item)
        item["status"] = "done"
        src.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        item.update(status="error", error=str(e)[-500:])
    finally:
        item["seconds"] = round(time.time() - t0, 1)
        save(item)


# ---------- API ----------

@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


def verified_lines() -> list[tuple[dict, int, dict]]:
    """(item, line index, line) for every line a person has checked: the training data."""
    return [(it, i, s) for it in items.values() if it["status"] == "done"
            for i, s in enumerate(it.get("segments", [])) if s.get("verified") and s["text"].strip()]


def sync_item_verified(it: dict) -> None:
    segs = [s for s in it.get("segments", []) if s["text"].strip()]
    it["verified"] = bool(segs) and all(s.get("verified") for s in segs)


@app.get("/api/status")
def status():
    lines = verified_lines()
    return {**model_state,
            "queued": sum(1 for i in items.values() if i["status"] in ("queued", "working")),
            "verified_lines": len(lines),
            "verified_seconds": round(sum(s["end"] - s["start"] for _, _, s in lines), 1),
            "suggestions": len(fixes.suggestions())}


TRAINING_GOAL_S = 3600  # about an hour of checked speech before fine-tuning is worth trying


def model_line(it: dict, i: int) -> str | None:
    lines = (it.get("model_text") or "").split("\n")
    return lines[i] if i < len(lines) else None


@app.get("/api/training")
def training():
    lines = verified_lines()
    secs = sum(s["end"] - s["start"] for _, _, s in lines)
    pairs = [(m, s["text"]) for it, i, s in lines if (m := model_line(it, i)) is not None]
    tops = sorted(fixes.data["counts"].items(), key=lambda kv: -kv[1])[:12]
    return {
        "lines": len(lines),
        "notes": len({it["id"] for it, _, _ in lines}),
        "seconds": round(secs, 1),
        "goal_seconds": TRAINING_GOAL_S,
        "accuracy": score(pairs),
        "unchecked_lines": sum(1 for it in items.values() if it["status"] == "done"
                               for s in it.get("segments", []) if s["text"].strip() and not s.get("verified")),
        "top_corrections": [{"from": k.split("\t")[0], "to": k.split("\t")[1], "count": n} for k, n in tops],
    }


@app.post("/api/training/export")
def training_export():
    """Write the checked lines as a training set: 16 kHz WAV clips + metadata.jsonl.
    Whole recordings go to either train or test (about 1 in 10 to test), never both."""
    import hashlib

    import soundfile as sf

    lines = verified_lines()
    if not lines:
        raise HTTPException(400, "No verified lines yet.")
    out = DATA_DIR / "exports" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (out / "audio").mkdir(parents=True)
    audio_cache: dict[str, object] = {}
    rows = []
    for it, i, s in lines:
        if it["id"] not in audio_cache:
            audio_cache[it["id"]] = load_audio(str(item_dir(it["id"]) / "audio.m4a"))
        clip = audio_cache[it["id"]][int(s["start"] * SR): int(s["end"] * SR)]
        rel = f"audio/{it['id']}_{i:04d}.wav"
        sf.write(out / rel, clip, SR, subtype="PCM_16")
        split = "test" if int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 10 == 0 else "train"
        rows.append({"audio": rel, "text": s["text"], "duration": round(s["end"] - s["start"], 2),
                     "split": split, "source": it["name"], "line": i, "model_text": model_line(it, i)})
    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"folder": str(out), "lines": len(rows),
            "train": sum(r["split"] == "train" for r in rows), "test": sum(r["split"] == "test" for r in rows)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    iid = uuid.uuid4().hex[:12]
    suffix = (Path(file.filename or "audio").suffix or ".bin").lower()[:10]
    d = item_dir(iid)
    d.mkdir(parents=True)
    with open(d / f"source{suffix}", "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    item = {"id": iid, "name": file.filename or "recording", "created": datetime.now().isoformat(timespec="microseconds"),
            "status": "queued", "progress": 0.0}
    items[iid] = item
    save(item)
    work.put(("transcribe", iid))
    return {"id": iid}


@app.post("/api/import")
def import_link(body: dict = Body(...)):
    """Queue a link (YouTube, Ayn, ...) for download + transcription."""
    url = str(body.get("url", "")).strip()
    if not re.match(r"^https?://\S+$", url):
        raise HTTPException(400, "Paste a full link starting with http:// or https://")
    iid = uuid.uuid4().hex[:12]
    item_dir(iid).mkdir(parents=True)
    item = {"id": iid, "name": url, "source_url": url, "created": datetime.now().isoformat(timespec="microseconds"),
            "status": "queued", "progress": 0.0}
    items[iid] = item
    save(item)
    work.put(("import", iid))
    return {"id": iid}


@app.post("/api/items/{iid}/retry")
def retry(iid: str):
    """Try a failed import again (audio already downloaded is reused)."""
    it = get_item(iid)
    if it["status"] != "error" or not it.get("source_url"):
        raise HTTPException(409, "Only failed imports can be retried.")
    it.update(status="queued", progress=0.0, error=None)
    save(it)
    work.put(("import", iid))
    return {"ok": True}


@app.get("/api/items")
def list_items(q: str = ""):
    q = q.strip()
    out = []
    for it in sorted(items.values(), key=lambda i: i["created"], reverse=True):
        text = full_text(it)
        if q and q not in text and q.lower() not in it["name"].lower():
            continue
        hit = ""
        if q and q in text:
            k = text.index(q)
            hit = ("…" if k > 40 else "") + text[max(0, k - 40): k + len(q) + 40] + "…"
        out.append({k: it.get(k) for k in ("id", "name", "created", "status", "progress", "duration", "seconds",
                                           "error", "verified", "auto_fixes", "from_subtitles", "source_url")}
                   | {"preview": hit or text[:140],
                      "syncing": sum(1 for s in it.get("segments", []) if s.get("pending_align"))})
    return out


def get_item(iid: str) -> dict:
    if iid not in items:
        raise HTTPException(404)
    return items[iid]


@app.get("/api/items/{iid}")
def item(iid: str):
    return get_item(iid)


@app.get("/api/items/{iid}/audio")
def item_audio(iid: str):
    get_item(iid)
    f = item_dir(iid) / "audio.m4a"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="audio/mp4")


@app.patch("/api/items/{iid}")
def update_item(iid: str, body: dict = Body(...)):
    """Rename ({"name": ...}) and/or mark as checked by a person ({"verified": true})."""
    it = get_item(iid)
    name = str(body.get("name", "")).strip()
    if name:
        it["name"] = name[:200]
    if "verified" in body:
        if it["status"] != "done":
            raise HTTPException(409, "Only finished transcripts can be verified.")
        for s in it.get("segments", []):  # whole note = every line checked
            s["verified"] = bool(body["verified"])
        sync_item_verified(it)
        it["verified_at"] = datetime.now().isoformat(timespec="seconds") if it["verified"] else None
    save(it)
    return {"ok": True, "verified": it.get("verified", False)}


@app.put("/api/items/{iid}/segment/verified")
def verify_line(iid: str, body: dict = Body(...)):
    """Mark one line as checked: {"s": line index, "verified": true}."""
    it = get_item(iid)
    try:
        seg = it["segments"][int(body["s"])]
    except (KeyError, IndexError, ValueError, TypeError):
        raise HTTPException(400, "bad line reference")
    seg["verified"] = bool(body.get("verified", True))
    sync_item_verified(it)
    save(it)
    return {"verified": seg["verified"], "item_verified": it["verified"]}


@app.put("/api/items/{iid}/word")
def edit_word(iid: str, body: dict = Body(...)):
    """Correct one word: {"s": segment index, "w": word index, "text": new text ('' deletes it)}."""
    it = get_item(iid)
    try:
        seg = it["segments"][int(body["s"])]
        words = seg["words"]
        w = int(body["w"])
        words[w]  # bounds check
    except (KeyError, IndexError, ValueError, TypeError):
        raise HTTPException(400, "bad word reference")
    parts = str(body.get("text", "")).split()
    before = words[w]["text"]
    if parts:
        # several words typed = the model missed some; they share this word's time span
        words[w:w + 1] = spread(words[w], parts, fixed=True, guess=len(parts) > 1)
        fixes.record(before, " ".join(parts))
        if len(parts) > 1:  # added words only have estimated times until re-synced
            seg["pending_align"] = True
    else:
        words.pop(w)
    seg["text"] = " ".join(x["text"] for x in words)
    it["edited"] = True
    save(it)
    if seg.get("pending_align"):
        queue_align(iid)
    return seg


@app.post("/api/items/{iid}/resync")
def resync(iid: str):
    """Re-sync every edited line of a transcript to its audio (all corrected words get re-timed)."""
    it = get_item(iid)
    if it["status"] != "done":
        raise HTTPException(409, "Not finished transcribing yet.")
    n = 0
    for seg in it.get("segments", []):
        edited = [w for w in seg["words"] if w.get("fixed")]
        if edited:
            for w in edited:
                w["guess"] = True
            seg["pending_align"] = True
            n += 1
    save(it)
    if n:
        queue_align(iid)
    return {"lines": n}


@app.put("/api/items/{iid}/segment")
def edit_segment(iid: str, body: dict = Body(...)):
    """Retype a whole line: {"s": segment index, "text": corrected text}. Word timings are re-fitted."""
    it = get_item(iid)
    try:
        seg = it["segments"][int(body["s"])]
    except (KeyError, IndexError, ValueError, TypeError):
        raise HTTPException(400, "bad segment reference")
    text = " ".join(str(body.get("text", "")).split())
    if text == seg["text"]:
        return seg
    words, swaps = realign(seg["words"], text, seg["start"], seg["end"])
    for before, after in swaps:
        fixes.record(before, after)
    seg["words"], seg["text"] = words, text
    it["edited"] = True
    if any(w.get("guess") for w in words):
        seg["pending_align"] = True  # added words have guessed times; the worker re-syncs them to the audio
    save(it)
    if seg.get("pending_align"):
        queue_align(iid)
    return seg


# ---------- learned fixes ----------

@app.get("/api/fixes")
def fixes_view():
    return fixes.view()


@app.post("/api/fixes")
def fixes_add(body: dict = Body(...)):
    try:
        fixes.add_rule(str(body.get("from", "")), str(body.get("to", "")))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return fixes.view()


@app.post("/api/fixes/remove")
def fixes_remove(body: dict = Body(...)):
    fixes.remove_rule(str(body.get("from", "")))
    return fixes.view()


@app.post("/api/fixes/dismiss")
def fixes_dismiss(body: dict = Body(...)):
    fixes.dismiss(str(body.get("from", "")), str(body.get("to", "")))
    return fixes.view()


@app.post("/api/fixes/apply")
def fixes_apply():
    """Apply enabled fixes to past transcripts (verified ones are left exactly as you checked them)."""
    words_changed = notes_changed = 0
    for it in list(items.values()):
        if it["status"] != "done" or it.get("verified"):
            continue
        n = fixes.apply(it)
        if n:
            it["auto_fixes"] = it.get("auto_fixes", 0) + n
            save(it)
            words_changed += n
            notes_changed += 1
    return {"words": words_changed, "notes": notes_changed}


@app.delete("/api/items/{iid}")
def delete(iid: str):
    it = get_item(iid)
    if it["status"] == "working":
        raise HTTPException(409, "Still transcribing; delete it when it's done.")
    items.pop(iid)
    shutil.rmtree(item_dir(iid), ignore_errors=True)
    return {"ok": True}


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt(item: dict) -> str:
    """Subtitle cues of up to ~8 words / 4 s, broken at sentence ends (and commas once a cue has 4+ words)."""
    cues, cur = [], []
    for seg in item.get("segments", []):
        for w in seg["words"]:
            cur.append(w)
            long = len(cur) >= 8 or cur[-1]["end"] - cur[0]["start"] >= 4
            end_mark = w["text"][-1:]
            if long or end_mark in ".؟?!" or (end_mark in "،," and len(cur) >= 4):
                cues.append(cur)
                cur = []
        if cur:
            cues.append(cur)
            cur = []
    lines = []
    for n, c in enumerate(cues, 1):
        start, end = c[0]["start"], c[-1]["end"]
        nxt = cues[n][0]["start"] if n < len(cues) else end + 1.0
        end = min(max(end, start + 1.0), nxt)  # keep short cues on screen ~1 s
        lines += [str(n), f"{srt_time(start)} --> {srt_time(end)}",
                  " ".join(w["text"] for w in c), ""]
    return "\n".join(lines)


@app.get("/api/items/{iid}/export")
def export(iid: str, fmt: str = "txt"):
    it = get_item(iid)
    stem = Path(it["name"]).stem or "transcript"
    if fmt == "srt":
        body, mime = srt(it), "application/x-subrip"
    elif fmt == "json":
        body, mime = json.dumps(it.get("segments", []), ensure_ascii=False, indent=1), "application/json"
    else:
        fmt, body, mime = "txt", full_text(it), "text/plain"
    # RFC 5987 filename so Arabic names survive
    from urllib.parse import quote
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(stem)}.{fmt}"}
    return PlainTextResponse("﻿" + body if fmt != "json" else body, media_type=f"{mime}; charset=utf-8",
                             headers=headers)


if __name__ == "__main__":
    load_all()
    threading.Thread(target=worker, daemon=True).start()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    print(f"Data folder: {DATA_DIR}")
    print(f"Open http://127.0.0.1:{port} in your browser")
    uvicorn.run(app, host=host, port=port, log_level="warning")
