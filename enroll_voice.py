#!/usr/bin/env python3
"""Enroll a person's voice so transcripts label them by name instead of "Speaker 3".

Everything stays on this Mac: a voiceprint is a list of numbers saved in
~/.config/transcribe-meeting/voices/<Name>.npy. It cannot be turned back into audio.

  enroll-voice "Pat Lee" clip.m4a                    # whole clip is that person
  enroll-voice "Pat Lee" meeting.m4a --from 12:30 --to 14:05
  enroll-voice "Sam Ortiz" --transcript "X.transcript.md" --speaker "Speaker 3"
  enroll-voice --list
  enroll-voice --remove "Pat Lee"

Enrolling the same person again ADDS a voiceprint (useful for each recorder or
room); matching uses the best one.
"""
import argparse, re, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcribe_meeting as tm  # noqa: E402

AUDIO_EXT = (".m4a", ".wav", ".mp3", ".WAV", ".MP3", ".M4A", ".mp4", ".mov")
MIN_SECONDS = 20


def secs(t):
    parts = [float(x) for x in t.split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


def spans_from_transcript(path, speaker):
    """Each turn of `speaker` runs from its timestamp to the next turn's."""
    raw = Path(str(path).replace(".transcript.", ".transcript.raw."))
    text = (raw if raw.exists() else Path(path)).read_text(encoding="utf-8")
    turns = re.findall(r"^\**([^*\[\n]+?)\**\s*\[(\d+:\d+(?::\d+)?)\]", text, re.M)
    spans = []
    for i, (who, ts) in enumerate(turns):
        if who.strip() == speaker:
            end = secs(turns[i + 1][1]) if i + 1 < len(turns) else secs(ts) + 30
            spans.append((secs(ts), end))
    return spans


def audio_for(transcript):
    base = str(transcript)
    for suf in (".transcript.md", ".transcript.txt"):
        if base.endswith(suf):
            base = base[:-len(suf)]
    for ext in AUDIO_EXT:
        if Path(base + ext).exists():
            return Path(base + ext)
    return None


def voiceprint(wav, spans):
    import numpy as np, soundfile as sf, sherpa_onnx
    audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(tm.EMB_MODEL)))
    win = int(3 * sr)
    vs, total = [], 0.0
    for a, b in spans or [(0, len(audio) / sr)]:
        seg = audio[int(a * sr):int(b * sr)]
        total += len(seg) / sr
        for i in range(0, max(len(seg) - win, 0) + 1, win):
            chunk = seg[i:i + win]
            if len(chunk) < sr:
                continue
            st = ext.create_stream(); st.accept_waveform(sr, chunk); st.input_finished()
            if ext.is_ready(st):
                v = np.array(ext.compute(st)); vs.append(v / (np.linalg.norm(v) + 1e-9))
    if not vs:
        return None, total
    m = np.mean(vs, axis=0)
    return m / (np.linalg.norm(m) + 1e-9), total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--transcript")
    ap.add_argument("--speaker")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--remove")
    a = ap.parse_args()
    import numpy as np

    if a.list:
        for n, p in tm.load_voices().items():
            print(f"{n}: {len(p)} voiceprint(s)")
        return
    if a.remove:
        f = tm.VOICES_DIR / f"{a.remove}.npy"
        if f.exists():
            f.unlink(); print(f"removed {a.remove}")
        else:
            sys.exit(f"no enrolled voice called {a.remove!r}")
        return
    if not a.name:
        ap.error("give a name")

    if a.transcript:
        if not a.speaker:
            ap.error("--transcript needs --speaker, e.g. --speaker \"Speaker 3\"")
        audio = Path(a.audio) if a.audio else audio_for(a.transcript)
        if not audio:
            sys.exit("couldn't find the recording next to the transcript; give it as the second argument")
        spans = spans_from_transcript(a.transcript, a.speaker)
        if not spans:
            sys.exit(f"no turns labelled {a.speaker!r} in that transcript")
    else:
        if not a.audio:
            ap.error("give an audio file, or --transcript and --speaker")
        audio = Path(a.audio)
        spans = [(secs(a.start or "0"), secs(a.end) if a.end else 10**9)]

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(audio),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        vp, total = voiceprint(wav, spans)
    if vp is None or total < MIN_SECONDS:
        sys.exit(f"only {total:.0f} s of speech found; need at least {MIN_SECONDS} s for a reliable voiceprint")

    tm.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    f = tm.VOICES_DIR / f"{a.name}.npy"
    old = np.atleast_2d(np.load(f)) if f.exists() else np.zeros((0, len(vp)))
    others = {n: p for n, p in tm.load_voices().items() if n != a.name}
    for n, p in others.items():
        sc = float(max(p @ vp))
        if sc >= tm.VOICE_MATCH:
            print(f"warning: this voice also matches {n!r} ({sc:.2f}). Check the clip is the right person.")
    np.save(f, np.vstack([old, vp]))
    print(f"enrolled {a.name} from {total:.0f} s of speech ({len(old) + 1} voiceprint(s))")


if __name__ == "__main__":
    main()
