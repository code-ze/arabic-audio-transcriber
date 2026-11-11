"""Import audio from a link (YouTube, Ayn/Oman TV, and anything yt-dlp supports).

If the video has human-made Arabic subtitles they are downloaded too and used as
the starting transcript (YouTube's automatic captions are ignored: they are
machine output, the very thing we're trying to improve on).

Media you import stays on this computer. Only import what you're allowed to use;
broadcasts and YouTube videos belong to their owners.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from pathlib import Path


def resolve(url: str) -> str:
    """Ayn (ayn.om) pages embed a Mangomolo player that yt-dlp understands directly."""
    host = urllib.parse.urlparse(url).netloc.lower()
    if host.endswith("ayn.om"):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        m = re.search(r'src="(https://player\.mangomolo\.com/v1/[^"]+)"', html)
        if m:
            return m.group(1).replace("&amp;", "&")
    return url


def fetch(url: str, dest: Path, progress=None) -> dict:
    """Download the best audio (+ human Arabic subtitles) into dest. Returns
    {"source": path, "subtitles": path | None, "title": str, "duration": seconds}."""
    import yt_dlp

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(min(d.get("downloaded_bytes", 0) / total, 1.0))

    opts = {
        # audio-only if offered; otherwise the smallest video that has sound (e.g. Ayn's HLS streams)
        "format": "bestaudio/worst[acodec!=none]/worst",
        "outtmpl": str(dest / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "retries": 3,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(resolve(url), download=True)
    source = next((p for p in dest.glob("source.*") if p.suffix not in (".vtt", ".part")), None)
    if source is None:
        raise RuntimeError("Download finished but no audio file was found.")
    # human-made Arabic subtitles only (automatic captions are machine output)
    subs = None
    tracks = info.get("subtitles") or {}
    lang = next((k for k in tracks if k.lower() == "ar"), None) or next((k for k in tracks if k.lower().startswith("ar")), None)
    if lang:
        sub_url = next((f["url"] for f in tracks[lang] if f.get("ext") == "vtt"), None)
        if sub_url:
            subs = dest / f"source.{lang}.vtt"
            subs.write_bytes(_get_with_backoff(sub_url))
    return {"source": source, "subtitles": subs, "title": info.get("title") or url,
            "duration": info.get("duration")}


def _get_with_backoff(url: str, waits=(0, 30, 120, 300)) -> bytes:
    """Subtitle hosts rate-limit (HTTP 429) after many requests; wait and retry."""
    import time
    import urllib.error

    last = None
    for w in waits:
        time.sleep(w)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 503):
                break
    raise RuntimeError(f"Couldn't fetch the subtitles ({last}). The site may be rate-limiting; try again later.")


def _ts(t: str) -> float:
    parts = t.replace(",", ".").split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_vtt(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cues = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        for k, line in enumerate(lines):
            if "-->" in line:
                a, b = line.split("-->")[:2]
                start, end = _ts(a.strip().split()[0]), _ts(b.strip().split()[0])
                body = " ".join(re.sub(r"<[^>]+>", "", l) for l in lines[k + 1:])
                body = " ".join(body.replace("&nbsp;", " ").replace("&amp;", "&").split())
                if body and end > start:
                    cues.append((start, end, body))
                break
    # drop exact repeats that some uploaders leave in (rolling captions)
    out: list[tuple[float, float, str]] = []
    for c in cues:
        if out and c[2] == out[-1][2]:
            out[-1] = (out[-1][0], c[1], c[2])
        else:
            out.append(c)
    return out


def group_cues(cues, max_len: float = 12.0, max_gap: float = 1.0):
    """Join neighbouring subtitle cues into lines of up to ~12 s (the size the model and training use)."""
    groups: list[tuple[float, float, str]] = []
    for s, e, t in cues:
        if groups and s - groups[-1][1] <= max_gap and e - groups[-1][0] <= max_len:
            gs, _, gt = groups[-1]
            groups[-1] = (gs, e, gt + " " + t)
        else:
            groups.append((s, e, t))
    return groups
