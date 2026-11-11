"""Local web app: upload or record Arabic voice notes, get text back with
word-by-word playback, a searchable history, editing and subtitle export.

Run:  start.bat   (or: python app.py)   then open http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import os
import queue
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

from fixes import Fixes, spread
from transcriber import SR, Transcriber, load_audio

HERE = Path(__file__).parent
# Every transcript lives in DATA_DIR/<id>/ : meta.json + audio.m4a
DATA_DIR = Path(os.environ.get("DATA_DIR", HERE / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
fixes = Fixes(DATA_DIR / "fixes.json")

app = FastAPI()
engine = Transcriber()
items: dict[str, dict] = {}
work: "queue.Queue[str]" = queue.Queue()
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
        if item["status"] in ("queued", "working"):
            if find_source(item["id"]):
                item.update(status="queued", progress=0.0)
                work.put(item["id"])
            else:
                item.update(status="error", error="Interrupted before it finished; please upload again.")
            save(item)


def find_source(iid: str) -> Path | None:
    return next(iter(item_dir(iid).glob("source.*")), None)


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
        iid = work.get()
        item = items.get(iid)
        src = find_source(iid)
        if not item or not src:
            continue
        item["status"] = "working"
        save(item)
        t0 = time.time()
        try:
            def prog(done, total):
                item["progress"] = done / total

            audio = load_audio(str(src))
            # compact copy for playback (AAC plays in every browser)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
                 "-ac", "1", "-c:a", "aac", "-b:a", "64k", str(item_dir(iid) / "audio.m4a")],
                check=True, capture_output=True,
            )
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


@app.get("/api/status")
def status():
    verified = [i for i in items.values() if i.get("verified")]
    return {**model_state,
            "queued": sum(1 for i in items.values() if i["status"] in ("queued", "working")),
            "verified": len(verified),
            "verified_seconds": round(sum(i.get("duration", 0) for i in verified), 1),
            "suggestions": len(fixes.suggestions())}


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
    work.put(iid)
    return {"id": iid}


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
                                           "error", "verified", "auto_fixes")}
                   | {"preview": hit or text[:140]})
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
        it["verified"] = bool(body["verified"])
        it["verified_at"] = datetime.now().isoformat(timespec="seconds") if it["verified"] else None
    save(it)
    return {"ok": True, "verified": it.get("verified", False)}


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
        words[w:w + 1] = spread(words[w], parts, fixed=True)
        fixes.record(before, " ".join(parts))
    else:
        words.pop(w)
    seg["text"] = " ".join(x["text"] for x in words)
    it["edited"] = True
    save(it)
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
