"""Forced alignment: find when each word of a (corrected) text is spoken.

Uses Meta's MMS forced-alignment model (torchaudio MMS_FA, ~1.2 GB, downloaded on
first use into TORCH_HOME). Words are romanized with uroman, so Arabic, dialect
spellings and French/English loanwords are matched by sound against the audio.
This is content-aware, so it handles heavy rewrites where the word count changes.
"""
from __future__ import annotations

import os
import threading

import numpy as np

SR = 16000


class Aligner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        import uroman
        from torchaudio.pipelines import MMS_FA

        if os.environ.get("TORCH_HOME"):
            torch.hub.set_dir(os.path.join(os.environ["TORCH_HOME"], "hub"))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = MMS_FA.get_model(with_star=False).to(self._device).eval()
        self._dict = MMS_FA.get_dict(star=None)
        self._aligner = MMS_FA.get_aligner()
        self._uroman = uroman.Uroman()

    def _romanize(self, word: str) -> str:
        r = self._uroman.romanize_string(word).lower().replace("’", "'")
        return "".join(c for c in r if c in self._dict and c not in ("-", "|"))

    def align(self, piece: np.ndarray, text: str) -> list[tuple[float, float] | None]:
        """(start, end) in seconds for every word of `text` (split on spaces), relative to
        the piece; None for words that can't be aligned (e.g. only punctuation)."""
        import torch

        words = text.split()
        if not words or len(piece) < SR // 10:
            return [None] * len(words)
        with self._lock:
            self._load()
            roman = [self._romanize(w) for w in words]
            keep = [i for i, r in enumerate(roman) if r]
            if not keep:
                return [None] * len(words)
            tokens = [[self._dict[c] for c in roman[i]] for i in keep]
            wav = torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32)).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                emission, _ = self._model(wav)
            n_frames = emission.size(1)
            # CTC needs at least one frame per token (+ blanks between repeats); bail out if the text is too long
            need = sum(len(t) for t in tokens) + sum(1 for t in tokens for a, b in zip(t, t[1:]) if a == b)
            if need > n_frames:
                return [None] * len(words)
            spans = self._aligner(emission[0], tokens)
        sec = len(piece) / SR / n_frames
        out: list[tuple[float, float] | None] = [None] * len(words)
        for i, sp in zip(keep, spans):
            out[i] = (round(sp[0].start * sec, 2), round(sp[-1].end * sec, 2))
        return out
