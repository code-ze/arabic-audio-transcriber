"""Local web app: upload Arabic voice notes, get text back.

Run:  .venv\\Scripts\\python app.py   then open http://127.0.0.1:8765
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from transcriber import Transcriber

HERE = Path(__file__).parent
OUT_DIR = HERE / "transcripts"
OUT_DIR.mkdir(exist_ok=True)

app = FastAPI()
engine = Transcriber()
jobs: dict[str, dict] = {}
work: "queue.Queue[str]" = queue.Queue()
model_state = {"status": "loading", "error": None}


def fmt_time(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def worker() -> None:
    try:
        engine.load()
        model_state["status"] = "ready"
    except Exception as e:  # noqa: BLE001
        model_state.update(status="error", error=str(e))
        return
    while True:
        jid = work.get()
        job = jobs[jid]
        job["status"] = "working"
        t0 = time.time()
        try:
            def prog(done, total):
                job["progress"] = done / total
            segs = engine.transcribe(job["path"], progress=prog)
            job["text"] = "\n".join(s.text for s in segs)
            job["segments"] = [{"start": fmt_time(s.start), "end": fmt_time(s.end), "text": s.text} for s in segs]
            job["status"] = "done"
            stem = Path(job["name"]).stem
            (OUT_DIR / f"{stem}.txt").write_text(job["text"], encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            job.update(status="error", error=str(e))
        finally:
            job["seconds"] = round(time.time() - t0, 1)
            try:
                os.remove(job["path"])
            except OSError:
                pass


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/status")
def status():
    return model_state


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio").suffix or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    jid = uuid.uuid4().hex
    jobs[jid] = {"id": jid, "name": file.filename, "path": path, "status": "queued", "progress": 0.0}
    work.put(jid)
    return {"id": jid}


@app.get("/api/job/{jid}")
def job(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    return {k: v for k, v in jobs[jid].items() if k != "path"}


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    print(f"Open http://127.0.0.1:{port} in your browser")
    uvicorn.run(app, host=host, port=port, log_level="warning")
