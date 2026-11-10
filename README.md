# Arabic Transcriber · تفريغ الصوت

Local, private transcription for Arabic voice notes, dialects included. Drop in WhatsApp voice notes (or record from the mic) and get text back with **word-by-word playback**. Everything runs on your own GPU; nothing is uploaded anywhere.

## Features

- **Upload or record**: WhatsApp `.opus`/`.ogg`, `.m4a`, `.mp3`, `.wav`, `.aac`, video files, or record straight from the microphone. Several files queue up automatically.
- **Word-by-word playback**: the word being spoken lights up as the audio plays. Click any word to jump there. Playback speed goes from 0.75× to 2×, and Space plays/pauses.
- **Fix mistakes**: double-click a word to correct it (Enter saves, Esc cancels, an empty word deletes it). Corrected words are underlined.
- **History and search**: every transcript and its audio is kept. Search across all of them in Arabic or Latin script.
- **Export**: copy the text, or download `.txt` or `.srt` subtitles (from the word timings).

## Model

[`mistralai/Voxtral-Mini-4B-Realtime-Arabic`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-Arabic) (Apache 2.0). Mistral's Arabic fine-tune covers MSA plus 15 dialects, including Omani, Gulf, Najdi, Egyptian, Levantine, Iraqi and Maghrebi. It was chosen over Jais-ASR, Cohere Transcribe Arabic and Whisper because the base Voxtral model scored best on Omani voice messages in the independent [Almieyar benchmark](https://arxiv.org/html/2609.35564v1) (30% WER vs 51% for Whisper-large-v3).

The model emits one token per 80 ms of audio, so word timings come straight from token positions; no separate alignment model is needed.

**Speed** on an RTX 3090: about 17 s for a 25 s note and about 17 s for a 76 s recording. Long audio is cut at natural pauses into 6–12 s pieces that are decoded in parallel, so length barely matters. Needs about 9 GB of GPU memory.

## Run on Windows (no Docker)

Requires Python 3.11, an NVIDIA GPU, and `ffmpeg` on PATH (`winget install ffmpeg`).

```bash
py -3.11 -m venv D:\AI\venvs\arabic-transcriber
D:\AI\venvs\arabic-transcriber\Scripts\python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
D:\AI\venvs\arabic-transcriber\Scripts\python -m pip install -r requirements.txt
D:\AI\venvs\arabic-transcriber\Scripts\hf download mistralai/Voxtral-Mini-4B-Realtime-Arabic --exclude consolidated.safetensors --local-dir D:\AI\models\voxtral-arabic
```

Then double-click **`start.bat`**. It opens http://127.0.0.1:8765.

`--exclude consolidated.safetensors` skips a second 9 GB copy of the weights that only vLLM uses. PyTorch is pinned to 2.8 because 2.11 failed to load on Windows (`WinError 1114` in `c10.dll`).

## Run with Docker (GPU)

Needs Docker Desktop with the WSL 2 backend and an NVIDIA driver. Download the model once as above, then:

```bash
docker compose up -d --build
```

`docker-compose.yml` mounts the model folder read-only and the transcript library, defaulting to `D:/AI/models/voxtral-arabic` and `D:/AI/arabic-transcriber-data`. Override them with `MODEL_DIR` / `DATA_DIR` in a `.env` file.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ASR_MODEL` | `mistralai/Voxtral-Mini-4B-Realtime-Arabic` | Local model folder or Hugging Face repo id |
| `DATA_DIR` | `./data` | Where transcripts and audio are stored |
| `HOST` / `PORT` | `127.0.0.1` / `8765` | Web server address |

## Files

- `transcriber.py`: audio decoding (ffmpeg), pause-based chunking, batched GPU inference, word timings
- `app.py`: FastAPI server, job queue, library storage, editing, SRT export
- `static/index.html`: the whole UI (single file, no build step)
