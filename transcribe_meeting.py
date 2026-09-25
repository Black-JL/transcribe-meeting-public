#!/usr/bin/env python3
"""
transcribe_meeting.py — robust local meeting transcription + speaker diarization
for Apple Silicon.

Pipeline:
  1. ffmpeg  -> 16 kHz mono WAV (what every ASR model wants anyway)
  2. parakeet-mlx (NVIDIA Parakeet on the Mac GPU) -> word-level transcript.
     Non-autoregressive, so it does NOT fall into Whisper's repeat-loop and does
     not drop long sections — the whole reason we use it for meetings.
  3. sherpa-onnx diarization (segmentation + speaker embeddings) -> who spoke
     when, using ONNX models downloaded from GitHub (no Hugging Face account).
  4. Merge: assign each word to the overlapping speaker, group into turns.
  5. Write a clean Markdown transcript (+ plain .txt) with timestamps + speakers.

Diarization models live in ~/.config/transcribe-meeting/models. If they are
missing, the script still produces a high-quality transcript, just without
speaker labels.

Usage:
  transcribe_meeting.py INPUT [--out DIR] [--no-diarize]
                        [--speakers N | --min-speakers N --max-speakers N]
                        [--title "Meeting name"]

Env:
  HF_TOKEN   Hugging Face read token (or stored in ~/.cache/huggingface/token)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# GUI launches (double-clicked .app) don't inherit the shell PATH, so make sure
# the tools we shell out to (ffmpeg, parakeet-mlx, python) are findable.
for _p in ("/opt/homebrew/bin", "/usr/local/bin",
           "/Library/Frameworks/Python.framework/Versions/3.12/bin",
           str(Path.home() / ".local" / "bin")):
    if _p not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = _p + ":" + os.environ.get("PATH", "")

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
# Local, GitHub-sourced diarization models (no Hugging Face account needed).
MODELS_DIR = Path(os.environ.get(
    "TM_MODELS_DIR", str(Path.home() / ".config" / "transcribe-meeting" / "models")))
SEG_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS_DIR / "nemo_en_titanet_small.onnx"


import datetime

LOG_FILE = Path.home() / "Library" / "Logs" / "transcribe-meeting.log"


def log(msg):
    line = f"[transcribe-meeting] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as fh:
            fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")
    except Exception:
        pass  # never let logging break the run


def fmt_ts(seconds):
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def to_wav(src: Path, dst: Path):
    """Decode anything ffmpeg understands to 16 kHz mono PCM."""
    log(f"converting audio -> 16 kHz mono wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("ffmpeg failed:\n" + r.stderr[-2000:])
        sys.exit(1)


def run_parakeet(wav: Path, workdir: Path):
    """Transcribe with parakeet-mlx, return list of word dicts {text,start,end}."""
    log("transcribing with parakeet-mlx (GPU)…")
    cmd = [
        "parakeet-mlx", str(wav),
        "--model", PARAKEET_MODEL,
        "--output-format", "json",
        "--output-dir", str(workdir),
        # robust long-form handling: chunk with overlap, local attention
        "--chunk-duration", "120",
        "--overlap-duration", "15",
        "--local-attention",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("parakeet-mlx failed:\n" + (r.stderr or r.stdout)[-2000:])
        sys.exit(1)
    out = workdir / (wav.stem + ".json")
    if not out.exists():
        cands = list(workdir.glob("*.json"))
        if not cands:
            log("parakeet produced no json output")
            sys.exit(1)
        out = cands[0]
    data = json.loads(out.read_text())
    words = []
    for sent in data.get("sentences", []):
        for tok in sent.get("tokens", []):
            words.append({"text": tok["text"], "start": tok["start"], "end": tok["end"]})
    if not words:  # fallback: sentence-level if no tokens
        for sent in data.get("sentences", []):
            words.append({"text": sent["text"], "start": sent["start"], "end": sent["end"]})
    log(f"transcribed {len(words)} words")
    return words, data.get("text", "")


def run_diarization(wav: Path, speakers, threshold):
    """Return list of {start,end,speaker} turns via sherpa-onnx (local), or None.

    Uses GitHub-sourced ONNX models — no Hugging Face account or token required.
    """
    if not SEG_MODEL.exists() or not EMB_MODEL.exists():
        log("diarization models not found — producing transcript without speakers.")
        log(f"  expected segmentation model at: {SEG_MODEL}")
        log("  run the installer (install-transcribe-meeting.sh) to fetch them from GitHub.")
        return None
    try:
        import soundfile as sf
        import sherpa_onnx
    except Exception as e:
        log(f"sherpa-onnx import failed: {e}")
        return None
    log("diarizing speakers (sherpa-onnx, fully local)…")
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG_MODEL))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speakers if speakers else -1,
            threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    try:
        sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    except Exception as e:
        log(f"could not initialize diarization: {e}")
        return None
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if sr != sd.sample_rate:
        log(f"warning: sample rate {sr} != model {sd.sample_rate}")
    res = sd.process(audio).sort_by_start_time()
    # Drop micro-turns (<0.5s) — usually overlap blips that create phantom
    # speakers and attach stray punctuation to the wrong person.
    turns = [{"start": s.start, "end": s.end, "speaker": s.speaker}
             for s in res if (s.end - s.start) >= 0.5]
    log(f"found {len(set(t['speaker'] for t in turns))} speakers across {len(turns)} turns")
    return turns


def speaker_for(word, turns):
    """Pick the diarization turn with the most overlap for this word."""
    ws, we = word["start"], word["end"]
    best, best_ov = None, 0.0
    for t in turns:
        ov = min(we, t["end"]) - max(ws, t["start"])
        if ov > best_ov:
            best_ov, best = ov, t["speaker"]
    if best is None:  # no overlap: snap to nearest turn by midpoint
        mid = (ws + we) / 2
        best = min(turns, key=lambda t: abs(((t["start"] + t["end"]) / 2) - mid))["speaker"]
    return best


def merge(words, turns):
    """Group consecutive words by speaker into labeled turns."""
    # stable speaker relabel: SPEAKER_00 -> "Speaker 1" by first appearance
    order, mapping = [], {}
    def label(raw):
        if raw not in mapping:
            order.append(raw)
            mapping[raw] = f"Speaker {len(order)}"
        return mapping[raw]

    segments, cur = [], None
    for w in words:
        spk = label(speaker_for(w, turns)) if turns else None
        if cur and cur["speaker"] == spk:
            cur["text"] += w["text"]
            cur["end"] = w["end"]
        else:
            if cur:
                segments.append(cur)
            cur = {"speaker": spk, "text": w["text"], "start": w["start"], "end": w["end"]}
    if cur:
        segments.append(cur)

    # Coalesce trivial fragments (punctuation/1-char) into the neighbor so a
    # momentary mis-label doesn't spawn a "Speaker N: ." line.
    cleaned = []
    for seg in segments:
        if cleaned and len(seg["text"].strip().strip(".,!?;:—-")) <= 1:
            cleaned[-1]["text"] += seg["text"]
            cleaned[-1]["end"] = seg["end"]
        else:
            cleaned.append(seg)
    # Re-coalesce consecutive same-speaker turns created by the cleanup.
    merged = []
    for seg in cleaned:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] += seg["text"]
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def write_outputs(segments, full_text, out_md, out_txt, title, diarized):
    lines = [f"# {title}", ""]
    if not diarized:
        lines.append("_(no speaker labels — diarization was skipped or unavailable)_")
        lines.append("")
    for seg in segments:
        ts = fmt_ts(seg["start"])
        text = seg["text"].strip()
        if seg["speaker"]:
            lines.append(f"**{seg['speaker']}** [{ts}]  {text}")
        else:
            lines.append(f"[{ts}]  {text}")
        lines.append("")
    out_md.write_text("\n".join(lines))
    # plain text version
    tlines = []
    for seg in segments:
        ts = fmt_ts(seg["start"])
        text = seg["text"].strip()
        prefix = f"{seg['speaker']} [{ts}]: " if seg["speaker"] else f"[{ts}] "
        tlines.append(prefix + text)
    out_txt.write_text("\n".join(tlines))


def main():
    ap = argparse.ArgumentParser(description="Local meeting transcription + diarization")
    ap.add_argument("input", help="audio or video file")
    ap.add_argument("--out", help="output directory (default: alongside input)")
    ap.add_argument("--no-diarize", action="store_true", help="skip speaker labels")
    ap.add_argument("--speakers", type=int, help="exact number of speakers if known")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="diarization cluster threshold (lower=more speakers); default 0.5")
    ap.add_argument("--title", help="title for the transcript")
    ap.add_argument("--no-notes", action="store_true",
                    help="skip the local-AI meeting-notes summarization step")
    ap.add_argument("--notes-model", help="Ollama model for notes (default gpt-oss:20b)")
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        log(f"input not found: {src}")
        sys.exit(1)
    outdir = Path(args.out).expanduser().resolve() if args.out else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    title = args.title or src.stem

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav = td / "audio.wav"
        to_wav(src, wav)
        words, full_text = run_parakeet(wav, td)

        turns = None
        if not args.no_diarize:
            turns = run_diarization(wav, args.speakers, args.threshold)

        segments = merge(words, turns)
        out_md = outdir / (src.stem + ".transcript.md")
        out_txt = outdir / (src.stem + ".transcript.txt")
        write_outputs(segments, full_text, out_md, out_txt, title, diarized=bool(turns))

    log(f"transcript done -> {out_md}")

    # Stage 2: turn the transcript into meeting notes with a local LLM.
    final_path = out_md
    if not args.no_notes:
        notes_md = outdir / (src.stem + ".notes.md")
        notes_py = str(Path(__file__).with_name("meeting_notes.py"))
        mdate = datetime.date.fromtimestamp(src.stat().st_mtime).isoformat()
        cmd = [sys.executable, notes_py, str(out_md), "--title", title,
               "--out", str(notes_md), "--date", mdate]
        if args.notes_model:
            cmd += ["--model", args.notes_model]
        log("generating meeting notes (local AI)…")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and notes_md.exists():
                log(f"notes done -> {notes_md}")
                final_path = notes_md
            else:
                log("notes step did not complete; transcript is still available. "
                    + (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""))
        except Exception as e:
            log(f"notes step error (transcript still available): {e}")

    log(f"done -> {final_path}")
    print(str(final_path))


if __name__ == "__main__":
    log(f"invoked: {' '.join(sys.argv[1:])}")
    try:
        main()
        log("completed OK")
    except SystemExit:
        raise
    except Exception:
        import traceback
        log("FATAL:\n" + traceback.format_exc())
        raise
