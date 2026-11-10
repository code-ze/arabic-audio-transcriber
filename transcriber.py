"""Arabic speech-to-text engine built on Mistral's Voxtral-Mini-4B-Realtime-Arabic
(MSA + 15 dialects incl. Omani and Gulf; runs fully locally).

- Decodes any audio/video format via ffmpeg (WhatsApp .opus/.ogg, .m4a, .mp3, .wav, .mp4 ...)
- Splits long recordings at quiet points into <= MAX_CHUNK_S pieces
- Transcribes each piece on the GPU and joins the text
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass

import numpy as np

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-Arabic"
SR = 16000
# Generation is CPU-launch-bound (~110 ms per 80 ms-token regardless of batch size),
# so we cut recordings at pauses into short pieces and decode them in one batch.
MAX_CHUNK_S = 12.0   # hard upper bound for one piece
MIN_CHUNK_S = 6.0    # look for a pause between MIN and MAX
MIN_TAIL_S = 3.0     # never leave a tiny last piece
BATCH = 16           # pieces decoded together on the GPU


def load_audio(path: str) -> np.ndarray:
    """Decode any file ffmpeg understands into 16 kHz mono float32."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg could not read this file: " + proc.stderr.decode(errors="ignore")[-400:])
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("The file contains no audio.")
    return audio


def split_on_silence(audio: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) sample ranges, cutting at the quietest 30 ms frame
    between MIN_CHUNK_S and MAX_CHUNK_S of each piece."""
    frame = int(0.03 * SR)
    n_frames = len(audio) // frame
    if len(audio) <= MAX_CHUNK_S * SR or n_frames == 0:
        return [(0, len(audio))]
    rms = np.sqrt(np.mean(audio[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1) + 1e-12)
    # smooth so we cut in the middle of a pause, not on one quiet frame inside a word
    rms = np.convolve(rms, np.ones(7) / 7, mode="same")

    spans, start = [], 0
    while len(audio) - start > MAX_CHUNK_S * SR:
        lo = (start + int(MIN_CHUNK_S * SR)) // frame
        hi = min(start + int(MAX_CHUNK_S * SR), len(audio) - int(MIN_TAIL_S * SR)) // frame
        hi = max(hi, lo + 1)
        cut = (lo + int(np.argmin(rms[lo:hi]))) * frame
        spans.append((start, cut))
        start = cut
    spans.append((start, len(audio)))
    return spans


@dataclass
class Segment:
    start: float
    end: float
    text: str


class Transcriber:
    def __init__(self) -> None:
        self._model = None
        self._proc = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

        self._proc = AutoProcessor.from_pretrained(MODEL_ID)
        sr = self._proc.feature_extractor.sampling_rate
        if sr != SR:
            raise RuntimeError(f"Model expects {sr} Hz audio, engine is set up for {SR} Hz")
        self._model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto"
        )

    def _run(self, pieces: list[np.ndarray]) -> list[str]:
        import torch

        inputs = self._proc(pieces, return_tensors="pt")
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)
        # one text token per 80 ms of audio, plus headroom
        max_new = int(max(len(p) for p in pieces) / SR / 0.08) + 64
        with torch.inference_mode():
            out = self._model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
        return [t.strip() for t in self._proc.batch_decode(out, skip_special_tokens=True)]

    def transcribe(self, path: str, progress=None) -> list[Segment]:
        audio = load_audio(path)
        spans = [(s, e) for s, e in split_on_silence(audio)
                 if np.sqrt(np.mean(audio[s:e] ** 2)) >= 1e-4]  # drop digital silence
        texts: list[str] = []
        with self._lock:  # one GPU job at a time
            self.load()
            for i in range(0, len(spans), BATCH):
                group = spans[i:i + BATCH]
                texts += self._run([audio[s:e] for s, e in group])
                if progress:
                    progress(min(i + BATCH, len(spans)), len(spans))
        return [Segment(s / SR, e / SR, t) for (s, e), t in zip(spans, texts) if t]
