#!/usr/bin/env python3
"""
transcribe_meeting.py — robust local meeting transcription + speaker diarization
for Apple Silicon.

Pipeline:
  1. ffmpeg  -> 16 kHz mono WAV (what every ASR model wants anyway)
  2. Voz (`da voz`, Desert Ant Labs: NVIDIA Parakeet TDT 0.6B v3 on the Neural
     Engine) -> word-level transcript. Non-autoregressive, so it does NOT fall
     into Whisper's repeat-loop and does not drop long sections. Runs as its own
     process (~0.5 GB), which exits and frees its memory when the step ends.
  3. sherpa-onnx diarization (segmentation + speaker embeddings) -> who spoke
     when, using ONNX models downloaded from GitHub (no Hugging Face account).
     — or, for an online meeting recorded as two tracks (--system), the mic is
     labeled "Me" and the Mac's system audio "Others", with no guessing.
  4. Merge: assign each whole word to the overlapping speaker, group into turns.
  5. Write a clean Markdown transcript (+ plain .txt) with timestamps + speakers.
  6. Meeting notes: what was discussed, names, due-outs (Claude by default;
     a local LLM via Ollama when offline or --sensitive),
     skipped when too little was said, plus a Word copy (.notes.docx).

Diarization models live in ~/.config/transcribe-meeting/models. If they are
missing, the script still produces a high-quality transcript, just without
speaker labels.

Usage:
  transcribe_meeting.py INPUT [--out DIR] [--no-diarize] [--speakers N]
                        [--title "Meeting name"] [--no-notes] [--notes-model M]
  transcribe_meeting.py MIC --system SYSTEM_AUDIO [...]      # online meeting

No accounts or tokens: models come from GitHub (speakers) and a one-time
download of Voz by the `da` CLI (speech), after which it runs offline.
"""

import argparse
import json
import re
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# GUI launches (double-clicked .app) don't inherit the shell PATH, so make sure
# the tools we shell out to (ffmpeg, da, python) are findable.
for _p in ("/opt/homebrew/bin", "/usr/local/bin",
           "/Library/Frameworks/Python.framework/Versions/3.12/bin",
           str(Path.home() / ".local" / "bin")):
    if _p not in os.environ.get("PATH", "").split(":"):
        os.environ["PATH"] = _p + ":" + os.environ.get("PATH", "")

DA_INSTALL = "brew install desert-ant-labs/tap/desertant && da pull voz"
# owner_check.py's second listen uses Whisper (a different model on purpose).
OWNER_CHECK_MODEL = "mlx-community/whisper-large-v3-turbo"


def _hf_cached(repo):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snap = hub / ("models--" + repo.replace("/", "--")) / "snapshots"
    return any(snap.glob("*/*.safetensors")) or any(snap.glob("*/*.npz"))


# Once the second-listen model is on disk, never touch the network for it.
if _hf_cached(OWNER_CHECK_MODEL):
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Local, GitHub-sourced diarization models (no Hugging Face account needed).
MODELS_DIR = Path(os.environ.get(
    "TM_MODELS_DIR", str(Path.home() / ".config" / "transcribe-meeting" / "models")))
SEG_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS_DIR / "nemo_en_titanet_small.onnx"
# Enrolled voices (enroll_voice.py): <Name>.npy, one or more voiceprints each.
VOICES_DIR = Path(os.environ.get("TM_VOICES_DIR", str(Path.home() / ".config" / "transcribe-meeting" / "voices")))
# Same person across recordings scored 0.70-0.98, different people ~0.0 (2026-10-22).
VOICE_MATCH = 0.6


def load_voices():
    import numpy as np
    out = {}
    if VOICES_DIR.is_dir():
        for f in sorted(VOICES_DIR.glob("*.npy")):
            out[f.stem] = np.atleast_2d(np.load(f))
    return out


def name_clusters(centroids):
    """Map each speaker group to an enrolled person when the voice matches.
    Several groups can map to one person (diarization often splits a voice)."""
    voices = load_voices()
    if not voices:
        return {}
    names = {}
    for c, v in centroids.items():
        best, score = None, VOICE_MATCH
        for name, prints in voices.items():
            sc = float(max(prints @ v))
            if sc >= score:
                best, score = name, sc
        if best:
            names[c] = best
    if names:
        summary = {}
        for n in names.values():
            summary[n] = summary.get(n, 0) + 1
        log("recognized voices: " + ", ".join(f"{n} ({k} group{'s' if k > 1 else ''})"
                                               for n, k in summary.items()))
    return names


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


def run_voz(wav: Path, workdir: Path):
    """Transcribe with Voz (`da voz`), return (list of word dicts {text,start,end}, text).

    `da` is a separate process on the Neural Engine: ~0.5 GB peak on a 59-minute
    file (vs 3.8 GB for parakeet-mlx in-process), and its memory is released the
    moment it exits. Voz only — if `da` is missing or fails, stop with the fix.
    """
    import resource
    import shutil
    log("transcribing with Voz (Neural Engine)…")
    if not shutil.which("da"):
        log("Voz is not installed (the `da` command was not found). Install it with:\n"
            f"    {DA_INSTALL}")
        sys.exit(1)
    r = subprocess.run(["da", "voz", str(wav), "--json", "--quiet", "--no-color"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log("Voz failed (exit %d):\n%s\nIf Voz is missing or broken, reinstall with:\n    %s"
            % (r.returncode, (r.stderr or r.stdout)[-2000:], DA_INSTALL))
        sys.exit(1)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        log("Voz returned output that is not JSON:\n" + r.stdout[-1000:])
        sys.exit(1)
    # Voz words carry no leading space; add one so whole_words() keeps them apart.
    words = [{"text": " " + w["text"].strip(), "start": float(w["start"]), "end": float(w["end"])}
             for w in data.get("words", []) if w.get("text", "").strip()]
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9  # bytes on macOS
    log(f"transcribed {len(words)} words (Voz peak {peak:.1f} GB, released)")
    return words, data.get("text", "")


DIAR_THREADS = max(2, min(8, (os.cpu_count() or 4)))


def run_diarization(wav: Path, speakers, threshold=0.65):
    """Who spoke when, fully local (sherpa-onnx).

    Forcing sherpa's clustering to exactly N speakers is unreliable: on a
    45-minute two-person session it put 1 stray segment in one cluster and the
    other 264 in the other. Instead: cluster automatically with a looser
    threshold, then refine with our own voice fingerprints — take the N biggest
    clusters (by talk time) as the real speakers and move every segment to the
    closest one. With N unknown (meetings), clusters with under 3% of the talk
    time are folded into the nearest real speaker.
    """
    if not SEG_MODEL.exists() or not EMB_MODEL.exists():
        log("speaker models not found — transcript will have no speaker labels")
        return None
    try:
        import numpy as np
        import soundfile as sf
        import sherpa_onnx
    except Exception as e:
        log(f"sherpa-onnx import failed: {e}")
        return None
    log("diarizing speakers (sherpa-onnx, fully local)…")
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG_MODEL)),
            num_threads=DIAR_THREADS),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(EMB_MODEL), num_threads=DIAR_THREADS),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    try:
        sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    except Exception as e:
        log(f"could not initialize diarization: {e}")
        return None
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    # Progress every ~5% so a watcher (the app's watchdog) sees the job is alive.
    last_pct = [-5]
    def progress(done, total):
        pct = int(100 * done / max(total, 1))
        if pct >= last_pct[0] + 5:
            last_pct[0] = pct
            log(f"diarizing: {pct}%")
        return 0
    segs = [{"start": x.start, "end": x.end, "speaker": x.speaker}
            for x in sd.process(audio, callback=progress).sort_by_start_time()
            if (x.end - x.start) >= 0.5]
    if not segs:
        return None

    # Voice fingerprint for each segment (skip very short ones: unreliable).
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=DIAR_THREADS))
    embs = {}
    for i, g in enumerate(segs):
        if g["end"] - g["start"] < 1.0:
            continue
        st = ext.create_stream()
        st.accept_waveform(sr, audio[int(g["start"] * sr):int(g["end"] * sr)])
        st.input_finished()
        if ext.is_ready(st):
            v = np.array(ext.compute(st)); embs[i] = v / (np.linalg.norm(v) + 1e-9)

    talk = {}
    for g in segs:
        talk[g["speaker"]] = talk.get(g["speaker"], 0) + g["end"] - g["start"]
    total = sum(talk.values())
    ranked = sorted(talk, key=talk.get, reverse=True)
    keep = ranked[:speakers] if speakers else [c for c in ranked if talk[c] / total >= 0.03] or ranked[:1]

    def centroids(labels):
        cs = {}
        for c in labels:
            vs = [embs[i] for i, g in enumerate(segs) if g["speaker"] == c and i in embs]
            if vs:
                m = np.mean(vs, axis=0); cs[c] = m / (np.linalg.norm(m) + 1e-9)
        return cs

    cs = centroids(keep)
    for _ in range(3):                       # a few k-means-style refinement passes
        if len(cs) < 2:
            break
        for i, g in enumerate(segs):
            if i in embs:
                g["speaker"] = max(cs, key=lambda c: float(embs[i] @ cs[c]))
        cs = centroids(list(cs))
    # segments too short to fingerprint: take the speaker of the nearest neighbour in time
    kept = set(cs) or set(keep)
    for i, g in enumerate(segs):
        if g["speaker"] not in kept:
            near = min((h for h in segs if h["speaker"] in kept),
                       key=lambda h: abs(h["start"] - g["start"]), default=None)
            if near:
                g["speaker"] = near["speaker"]
    # Put names on groups whose voice matches an enrolled person.
    named = name_clusters(centroids(list(kept)))
    for g in segs:
        if g["speaker"] in named:
            g["speaker"] = "@" + named[g["speaker"]]
    counts = {}
    for g in segs:
        counts[g["speaker"]] = counts.get(g["speaker"], 0) + 1
    log(f"found {len(counts)} speakers across {len(segs)} turns")
    return segs


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


def whole_words(tokens):
    """Join sub-word tokens into whole words: a new word starts with a leading
    space, so one word is never split between two speakers ("tim" | "es").
    Voz already returns whole words; run_voz() gives each a leading space."""
    out = []
    for t in tokens:
        if out and not t["text"].startswith((" ", "\n")) and t["text"].strip():
            out[-1]["text"] += t["text"]; out[-1]["end"] = t["end"]
        else:
            out.append(dict(t))
    return out


def smooth_speakers(words, spks, majority=0.6, short_words=4, short_secs=1.5):
    """Stop turns from breaking mid-sentence.

    Speaker labels are assigned word by word, so a brief mis-label splits a
    sentence between two people. Two passes:
      1. Within each sentence, if one speaker holds >= `majority` of its spoken
         time, the whole sentence goes to that speaker. (A real interruption,
         where talk time is split more evenly, is left alone.)
      2. A run shorter than `short_words` words AND `short_secs` seconds is
         folded into the previous run (or the next one, at the start).
    """
    n = len(words)
    if n == 0:
        return spks
    spks = list(spks)
    # pass 1: sentence majority by duration
    start = 0
    for i, w in enumerate(words):
        if w["text"].strip().endswith((".", "?", "!")) or i == n - 1:
            dur = {}
            for k in range(start, i + 1):
                d = max(words[k]["end"] - words[k]["start"], 0.01)
                dur[spks[k]] = dur.get(spks[k], 0.0) + d
            top = max(dur, key=dur.get)
            if dur[top] / sum(dur.values()) >= majority:
                for k in range(start, i + 1):
                    spks[k] = top
            start = i + 1
    # pass 2: fold very short runs into a neighbour
    for _ in range(2):
        runs, a = [], 0
        for i in range(1, n + 1):
            if i == n or spks[i] != spks[a]:
                runs.append((a, i)); a = i
        for j, (a, b) in enumerate(runs):
            secs = words[b - 1]["end"] - words[a]["start"]
            if (b - a) < short_words and secs < short_secs and len(runs) > 1:
                nb = spks[runs[j - 1][0]] if j > 0 else spks[runs[j + 1][0]]
                for k in range(a, b):
                    spks[k] = nb
    return spks


def merge(words, turns):
    """Group consecutive words by speaker into labeled turns."""
    words = whole_words(words)
    # stable speaker relabel: SPEAKER_00 -> "Speaker 1" by first appearance
    order, mapping = [], {}
    def label(raw):
        if isinstance(raw, str) and raw.startswith("@"):
            return raw[1:]                     # an enrolled, recognized voice
        if raw not in mapping:
            order.append(raw)
            mapping[raw] = f"Speaker {len(order)}"
        return mapping[raw]

    spks = [label(speaker_for(w, turns)) if turns else None for w in words]
    if turns:
        spks = smooth_speakers(words, spks)

    segments, cur = [], None
    for w, spk in zip(words, spks):
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


def label_two_tracks(mic_words, sys_words, mic_label="Me", sys_label="Others"):
    """Online meeting: interleave separately transcribed mic + system tracks."""
    tagged = [dict(w, speaker=mic_label) for w in whole_words(mic_words)] + \
             [dict(w, speaker=sys_label) for w in whole_words(sys_words)]
    tagged.sort(key=lambda w: w["start"])
    segments, cur = [], None
    for w in tagged:
        if cur and cur["speaker"] == w["speaker"]:
            cur["text"] += w["text"]; cur["end"] = w["end"]
        else:
            if cur:
                segments.append(cur)
            cur = {"speaker": w["speaker"], "text": w["text"], "start": w["start"], "end": w["end"]}
    if cur:
        segments.append(cur)
    return segments


MIN_NOTES_WORDS = 100   # below this, a model tends to invent a meeting


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
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="speaker clustering threshold (higher = fewer speakers); default 0.65")
    ap.add_argument("--title", help="title for the transcript")
    ap.add_argument("--no-notes", action="store_true",
                    help="skip the local-AI meeting-notes summarization step")
    ap.add_argument("--notes-model",
                    help='notes writer: "claude" (default, online) or an Ollama model name (offline)')
    ap.add_argument("--sensitive", action="store_true",
                    help="involves students or other protected information: notes stay offline (never Claude)")
    ap.add_argument("--attendees", help="who was in the room (passed to the notes step)")
    ap.add_argument("--type", help="meeting type for the notes: general, handover, aar, committee, debrief")
    ap.add_argument("--my-notes", help="your own key points: a file or the text (passed to the notes step)")
    ap.add_argument("--system", help="second track: the Mac's system audio (online meeting)")
    ap.add_argument("--tracker", action="store_true",
                    help="carry forward the program due-out tracker's open items (on automatically "
                         "for meetings whose title matches TM_TRACKER_TITLES)")
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        log(f"input not found: {src}")
        sys.exit(1)
    outdir = Path(args.out).expanduser().resolve() if args.out else src.parent
    # Hold marker: the folder watcher (tm-watch) leaves this meeting alone while
    # we work on it. The app also creates it at Stop, while you add notes.
    (outdir / (src.stem + ".later")).unlink(missing_ok=True)   # processing it now
    hold = outdir / (src.stem + ".hold")
    try:
        hold.touch()
        import atexit
        atexit.register(lambda: hold.exists() and hold.unlink())
    except OSError:
        pass
    outdir.mkdir(parents=True, exist_ok=True)
    title = args.title or src.stem

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if args.system:
            (td / "mic").mkdir(); (td / "sys").mkdir()
            to_wav(src, td / "mic" / "mic.wav")
            to_wav(Path(args.system).expanduser().resolve(), td / "sys" / "sys.wav")
            mic_words, _ = run_voz(td / "mic" / "mic.wav", td / "mic")
            sys_words, _ = run_voz(td / "sys" / "sys.wav", td / "sys")
            segments = label_two_tracks(mic_words, sys_words)
            full_text, labeled = "", True
        else:
            wav = td / "audio.wav"
            to_wav(src, wav)
            # 16 kHz mono 16-bit = 32,000 bytes/s. Under half a second there is
            # nothing to transcribe, and the GPU model crashes on empty input.
            if wav.stat().st_size < 16000 + 44:
                log("no audio to transcribe (recording is shorter than half a second)")
                sys.exit(7)
            words, full_text = run_voz(wav, td)
            turns = None
            if not args.no_diarize:
                turns = run_diarization(wav, args.speakers, args.threshold)
            segments = merge(words, turns)
            labeled = bool(turns)
        out_md = outdir / (src.stem + ".transcript.md")
        out_txt = outdir / (src.stem + ".transcript.txt")
        write_outputs(segments, full_text, out_md, out_txt, title, diarized=labeled)
    spoken = sum(len(seg["text"].split()) for seg in segments)

    log(f"transcript done -> {out_md}")

    # Stage 2: turn the transcript into meeting notes with a local LLM.
    final_path = out_md
    if not args.no_notes and spoken < MIN_NOTES_WORDS:
        log(f"notes skipped: only {spoken} words were spoken — too little for minutes "
            "(a model would invent content). Transcript is available.")
    elif not args.no_notes:
        notes_md = outdir / (src.stem + ".notes.md")
        notes_py = str(Path(__file__).with_name("meeting_notes.py"))
        mdate = datetime.date.fromtimestamp(src.stat().st_mtime).isoformat()
        cmd = [sys.executable, notes_py, str(out_md), "--title", title,
               "--out", str(notes_md), "--date", mdate]
        if args.notes_model:
            cmd += ["--model", args.notes_model]
        if args.attendees:
            cmd += ["--attendees", args.attendees]
        if args.type and args.type != "auto":
            cmd += ["--type", args.type]
        if args.my_notes:
            cmd += ["--my-notes", args.my_notes]
        if args.sensitive:
            cmd += ["--sensitive"]
        if args.tracker or (os.environ.get("TM_TRACKER_TITLES") and re.search(os.environ["TM_TRACKER_TITLES"], title, re.I)):
            cmd += ["--tracker"]
        online = (not args.sensitive) and (args.notes_model or "claude").startswith("claude")
        log("generating meeting notes (Claude, online)…" if online else "generating meeting notes (local AI)…")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and notes_md.exists():
                log(f"notes done -> {notes_md}")
                final_path = notes_md
                # Second listen on the owner lines (local Whisper); never fails the run.
                oc = Path(__file__).with_name("owner_check.py")
                if oc.exists() and not os.environ.get("TM_NO_OWNER_CHECK"):
                    log("second listen on due-out owner lines (local)…")
                    try:
                        rc = subprocess.run([sys.executable, str(oc), str(notes_md), str(out_md),
                                             str(src)], capture_output=True, text=True)
                        for ln in rc.stderr.strip().splitlines()[-2:]:
                            log(ln)
                    except Exception as e:
                        log(f"owner check skipped: {e}")
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
