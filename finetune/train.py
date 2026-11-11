"""LoRA fine-tuning of Voxtral-Mini-4B-Realtime-Arabic on the app's checked + subtitle lines.

How the training targets are built
  Voxtral Realtime emits one token per 80 ms of audio; at each position the audio frame
  embedding is ADDED to the token embedding, so the token sequence must have exactly one
  position per audio frame. The model's own output format (verified by rebuilding its outputs,
  see selftest()) is:
    - [STREAMING_PAD] while nothing new is said
    - when a word starts and nothing is pending: [STREAMING_WORD], then the word's tokens
    - a word that starts while earlier tokens are still being written just queues behind them
  A word that starts at t seconds (in the app's timing convention) begins at position
  prompt_len + 2 + t / 0.08.

Usage
  python finetune/train.py --data D:/AI/arabic-transcriber-data/finetune/dataset.json --out D:/AI/models/voxtral-omani-lora
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metrics import score  # noqa: E402
from transcriber import SR, load_audio  # noqa: E402

TOKEN_S = 0.08
ALIGNER_OFFSET_S = 0.10  # MMS aligner times words ~0.1 s earlier than the model's own convention


class Builder:
    def __init__(self, proc, model):
        self.proc = proc
        tok = proc.tokenizer
        self.tok = tok
        self.PAD = tok.convert_tokens_to_ids("[STREAMING_PAD]")
        self.WORD = tok.convert_tokens_to_ids("[STREAMING_WORD]")
        self.EOS = model.generation_config.eos_token_id

    def word_tokens(self, text: str) -> list[list[int]]:
        return [self.tok.encode(" " + w, add_special_tokens=False) for w in text.split()]

    def sequence(self, prompt: list[int], L: int, begins: list[int], words: list[list[int]]) -> list[int] | None:
        seq = list(prompt) + [self.PAD] * (L - len(prompt))
        queue, wi = [], 0
        for p in range(len(prompt), L):
            starting = []
            while wi < len(words) and begins[wi] <= p:
                starting.append(words[wi]); wi += 1
            if starting and not queue:
                seq[p] = self.WORD
                for s in starting:
                    queue += s
                continue
            for s in starting:
                queue += s
            seq[p] = queue.pop(0) if queue else self.PAD
        if queue or wi < len(words):
            return None  # text doesn't fit in the audio (too fast / bad timing)
        return seq


def load_clips(samples, data_dir: Path):
    cache, clips = {}, []
    for s in samples:
        if s["item"] not in cache:
            cache[s["item"]] = load_audio(str(data_dir / s["item"] / "audio.m4a"))
        a = cache[s["item"]]
        clips.append(a[int(s["start"] * SR): int(s["end"] * SR)])
    return clips


def make_batch(builder, samples, clips, device, dtype):
    inputs = builder.proc(clips, return_tensors="pt")
    P = inputs["input_ids"].shape[1]
    L = int(inputs["input_features"].shape[-1] // 8)
    prompt = inputs["input_ids"][0].tolist()
    seqs, labels, keep = [], [], []
    for b, s in enumerate(samples):
        off = ALIGNER_OFFSET_S if s.get("aligner_timed") else 0.0
        begins = [P + 2 + int(round((t + off) / TOKEN_S)) for t in s["word_starts"]]
        own_L = min(L, P + int(math.ceil(len(clips[b]) / SR / TOKEN_S)) + 24)  # audio + right padding of this clip
        seq = builder.sequence(prompt, own_L, begins, builder.word_tokens(s["text"]))
        if seq is None:
            continue
        lab = [-100] * P + seq[P:] + [-100] * (L - own_L)
        seq = seq + [builder.PAD] * (L - own_L)
        seqs.append(seq); labels.append(lab); keep.append(b)
    if not keep:
        return None
    feats = inputs["input_features"][keep]
    return {
        "input_ids": torch.tensor(seqs, device=device),
        "labels": torch.tensor(labels, device=device),
        "input_features": feats.to(device, dtype=dtype),
        "num_delay_tokens": inputs["num_delay_tokens"],
    }


@torch.inference_mode()
def transcribe(model, proc, clips, bs=16):
    out = []
    for i in range(0, len(clips), bs):
        group = clips[i:i + bs]
        inputs = proc(group, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        P = inputs["input_ids"].shape[1]
        max_new = int(max(len(c) for c in group) / SR / TOKEN_S) + 64
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
        out += [t.strip() for t in proc.batch_decode(gen[:, P:], skip_special_tokens=True)]
    return out


def evaluate(model, proc, test, clips, label):
    t0 = time.time()
    hyps = transcribe(model, proc, clips)
    res = {"all": score(list(zip(hyps, [s["text"] for s in test])))}
    for kind in ("gold", "silver"):
        idx = [i for i, s in enumerate(test) if s["kind"] == kind]
        if idx:
            res[kind] = score([(hyps[i], test[i]["text"]) for i in idx])
    print(f"[{label}] {json.dumps(res, ensure_ascii=False)}  ({time.time()-t0:.0f}s)", flush=True)
    return res, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--library", default=r"D:\AI\arabic-transcriber-data")
    ap.add_argument("--model", default=r"D:\AI\models\voxtral-arabic")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gold_weight", type=int, default=3, help="repeat checked lines this many times")
    ap.add_argument("--max_test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    random.seed(args.seed); torch.manual_seed(args.seed)
    ds = json.loads(Path(args.data).read_text(encoding="utf-8"))
    train = [s for s in ds["samples"] if s["split"] == "train"]
    test = [s for s in ds["samples"] if s["split"] == "test"]
    random.shuffle(test)
    test = sorted(test[:args.max_test], key=lambda s: (s["item"], s["start"]))
    train = train + [s for s in train if s["kind"] == "gold"] * (args.gold_weight - 1)
    print(f"train {len(train)} lines, test {len(test)} lines", flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    lib = Path(args.library)
    test_clips = load_clips(test, lib)
    base, base_hyps = evaluate(model, proc, test, test_clips, "base model")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    builder = Builder(proc, model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = int(math.ceil(len(train) / args.bs) * args.epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / steps))

    train_clips_cache: dict = {}
    step, skipped, t0 = 0, 0, time.time()
    model.train()
    while step < steps:
        order = list(range(len(train)))
        random.shuffle(order)
        # group similar lengths together so batches waste little padding
        chunks = [order[i:i + args.bs * 20] for i in range(0, len(order), args.bs * 20)]
        for ch in chunks:
            ch.sort(key=lambda i: train[i]["end"] - train[i]["start"])
            for k in range(0, len(ch), args.bs):
                if step >= steps:
                    break
                bs = [train[i] for i in ch[k:k + args.bs]]
                clips = []
                for s in bs:
                    key = (s["item"], s["line"])
                    if key not in train_clips_cache:
                        train_clips_cache[key] = load_clips([s], lib)[0]
                    clips.append(train_clips_cache[key])
                batch = make_batch(builder, bs, clips, model.device, torch.bfloat16)
                if batch is None:
                    skipped += len(bs); continue
                skipped += len(bs) - batch["input_ids"].shape[0]
                loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1:
                    print(f"step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"skipped {skipped} ({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    tuned, tuned_hyps = evaluate(model, proc, test, test_clips, "fine-tuned")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    report = {"base": base, "tuned": tuned, "args": vars(args), "train_lines": len(train), "test_lines": len(test),
              "skipped_lines": skipped,
              "examples": [{"ref": s["text"], "base": b, "tuned": t, "source": s["source"]}
                           for s, b, t in list(zip(test, base_hyps, tuned_hyps))[:40]]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved adapter + report to", out, flush=True)


def selftest():
    """Rebuild the model's own outputs from its word timings; the format is right if they match."""
    from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

    from transcriber import split_on_silence, words_from_tokens

    mdir = r"D:\AI\models\voxtral-arabic"
    proc = AutoProcessor.from_pretrained(mdir)
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(mdir, dtype=torch.bfloat16, device_map="cuda")
    b = Builder(proc, model)
    audio = load_audio(str(Path(mdir) / "assets" / "bank-take-2.wav"))
    total = same = 0
    for s, e in split_on_silence(audio):
        inputs = proc([audio[s:e]], return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        P = inputs["input_ids"].shape[1]
        L = int(inputs["input_features"].shape[-1] // 8)
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=L)[0].tolist()[:L]
        dec = lambda ids: proc.tokenizer.decode(ids, skip_special_tokens=True)
        words = words_from_tokens(out, P, dec, (e - s) / SR)  # app timing convention
        begins = [P + 2 + int(round(w.start / TOKEN_S)) for w in words]
        rebuilt = b.sequence(out[:P], len(out), begins, b.word_tokens(" ".join(w.text for w in words)))
        eq = sum(1 for p in range(P, len(out)) if rebuilt and rebuilt[p] == out[p])
        total += len(out) - P; same += eq
        print(f"span {s/SR:5.1f}-{e/SR:5.1f}s  positions {len(out)} (audio frames {L})  rebuilt match {eq}/{len(out)-P}")
    print(f"TOKEN MATCH {100*same/total:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
