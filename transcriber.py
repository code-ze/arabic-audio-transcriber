"""Arabic speech-to-text engine built on Mistral's Voxtral-Mini-4B-Realtime-Arabic
(MSA + 15 dialects incl. Omani and Gulf; runs fully locally).

- Decodes any audio/video format via ffmpeg (WhatsApp .opus/.ogg, .m4a, .mp3, .wav, .mp4 ...)
- Splits long recordings at quiet points into <= MAX_CHUNK_S pieces
- Transcribes each piece on the GPU and joins the text
"""
from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass, field

import numpy as np

# A local folder (e.g. D:\AI\models\voxtral-arabic) or the Hugging Face repo id
MODEL_ID = os.environ.get("ASR_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-Arabic")
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
        raise RuntimeError("Couldn't read any audio from this file. Is it a voice note or another audio/video file?")
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
class Word:
    start: float
    end: float
    text: str


# Voxtral Realtime emits exactly one token per 80 ms of (padded) audio: text tokens
# appear where the word is heard, with [STREAMING_PAD]/[STREAMING_WORD] filling the gaps.
# The prompt is BOS + 32 left-pad + 6 delay tokens; measured against speech onsets,
# a word's token lands ~3 steps after it starts, so start ≈ (pos - prompt_len - 3) * 80 ms.
TOKEN_S = 0.08
WORD_LEAD_TOKENS = 3


def words_from_tokens(ids: list[int], prompt_len: int, decode, dur: float) -> list[Word]:
    words: list[list] = []  # [start_pos, last_pos, [token ids]]
    for pos in range(prompt_len, len(ids)):
        piece = decode([ids[pos]])
        if not piece:  # padding / word marker / eos
            continue
        if not words or piece[0].isspace():
            words.append([pos, pos, [ids[pos]]])
        else:
            words[-1][1] = pos
            words[-1][2].append(ids[pos])

    def t(pos: int) -> float:
        return min(max((pos - prompt_len - WORD_LEAD_TOKENS) * TOKEN_S, 0.0), dur)

    out: list[Word] = []
    for i, (p0, p1, toks) in enumerate(words):
        start = t(p0)
        nxt = t(words[i + 1][0]) if i + 1 < len(words) else dur
        end = min(max(t(p1) + 0.4, start + 0.2), nxt, dur)
        text = decode(toks).strip()
        if text:
            out.append(Word(round(start, 2), round(max(end, start), 2), text))
    return out


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


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
        if torch.cuda.is_available():
            # Hard cap so we get a (recoverable) out-of-memory error instead of Windows silently
            # moving GPU memory into system RAM, which makes everything ~50x slower.
            frac = float(os.environ.get("GPU_MEMORY_FRACTION", "0.75"))
            torch.cuda.set_per_process_memory_fraction(frac)
        self._model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto"
        )

    def _decode(self, ids: list[int]) -> str:
        return self._proc.tokenizer.decode(ids, skip_special_tokens=True)

    def _run(self, pieces: list[np.ndarray]) -> list[tuple[str, list[Word]]]:
        import torch

        inputs = self._proc(pieces, return_tensors="pt")
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)
        prompt_len = inputs["input_ids"].shape[1]
        # one text token per 80 ms of audio, plus headroom
        max_new = int(max(len(p) for p in pieces) / SR / TOKEN_S) + 64
        with torch.inference_mode():
            out = self._model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
        results = []
        for row, piece in zip(out.tolist(), pieces):
            text = self._decode(row[prompt_len:]).strip()
            words = words_from_tokens(row, prompt_len, self._decode, len(piece) / SR)
            results.append((text, words))
        return results

    def _run_safe(self, pieces: list[np.ndarray]) -> list[tuple[str, list[Word]]]:
        """_run, but if the GPU runs out of memory, free the cache and retry in halves."""
        import torch

        try:
            return self._run(pieces)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(pieces) == 1:
                raise
            mid = len(pieces) // 2
            return self._run_safe(pieces[:mid]) + self._run_safe(pieces[mid:])

    def run_spans(self, audio: np.ndarray, spans: list[tuple[int, int]], progress=None) -> list[tuple[str, list[Word]]]:
        """Transcribe the given (start, end) sample ranges, batched on the GPU."""
        import torch

        results: list[tuple[str, list[Word]]] = []
        with self._lock:  # one GPU job at a time
            self.load()
            for i in range(0, len(spans), BATCH):
                group = spans[i:i + BATCH]
                results += self._run_safe([audio[s:e] for s, e in group])
                # Batches differ in size; without this the cache fragments and keeps growing on long
                # recordings until Windows spills GPU memory into system RAM and everything crawls.
                torch.cuda.empty_cache()
                if progress:
                    progress(min(i + BATCH, len(spans)), len(spans))
        return results

    def transcribe(self, path: str, progress=None, audio: np.ndarray | None = None) -> list[Segment]:
        if audio is None:
            audio = load_audio(path)
        spans = [(s, e) for s, e in split_on_silence(audio)
                 if np.sqrt(np.mean(audio[s:e] ** 2)) >= 1e-4]  # drop digital silence
        results = self.run_spans(audio, spans, progress)
        segments = []
        for (s, e), (text, words) in zip(spans, results):
            if not text:
                continue
            off = s / SR
            words = [Word(round(w.start + off, 2), round(w.end + off, 2), w.text) for w in words]
            segments.append(Segment(off, e / SR, text, words))
        return segments
