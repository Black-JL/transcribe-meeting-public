# Transcribe Meeting

A native macOS app that turns a meeting into **meeting notes** — not just a
transcript. Record (or drop a file) and it transcribes with **speaker labels**
and summarizes into structured notes (summary, themes, decisions, and action
items by person). Built for long, multi-speaker, in-room recordings where plain
Whisper drops audio and loops on repeated words.

**Everything runs on your Mac. No cloud, no accounts, no API keys** — the
summarization uses a local LLM via Ollama.

## How it works
1. **Transcription:** [parakeet-mlx](https://github.com/senstella/parakeet-mlx)
   (NVIDIA Parakeet) on the Apple GPU via MLX. Non-autoregressive, so it can't
   fall into Whisper's repeat-loop and doesn't drop long sections.
2. **Speaker labels:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
   diarization with ONNX models from **GitHub** (no Hugging Face).
3. **Meeting notes:** a local LLM via [Ollama](https://ollama.com) (default
   `gpt-oss:20b`) summarizes the transcript into markdown notes — summary, key
   themes, decisions, and **action items grouped by person** — applying your
   terminology glossary so speech-to-text errors get fixed automatically.
4. **Recording:** AVFoundation (a real signed app, so the macOS mic-permission
   prompt works correctly). The mic is released the moment the app closes.

Each meeting produces three files: the recording, `<name>.transcript.md`, and
`<name>.notes.md` (the headline deliverable).

## Install
```
cd transcribe-meeting-kit
./install.sh            # add --watch to also enable the auto-transcribe folder
```
Installs the Python pipeline, downloads the diarization models from GitHub, and
builds **`~/Applications/Transcribe Meeting.app`**. The parakeet model (~1–2 GB)
downloads once on first transcription, then is cached.

Requires: Apple Silicon Mac, Python 3, `ffmpeg`, and Xcode Command Line Tools
(`xcode-select --install`) for the Swift build.

## Use it
Open **Transcribe Meeting.app**:
- **Record New Meeting** — optionally type a name, hit record, talk, hit Stop.
  It transcribes, diarizes, and writes meeting notes automatically.
- **Transcribe a File…** — pick any audio/video file.
- **Recent Meetings** — every meeting is listed; open the **notes** or the
  **transcript**, copy text, or reveal in Finder. **Open Folder** jumps straight
  to where everything is saved.

Everything lives in **`~/Documents/Transcribed Meetings/`**: the recording,
`<name>.transcript.md` (speaker-labeled, timestamped), and `<name>.notes.md`.

### CLI
```
transcribe-meeting /path/to/recording.m4a          # transcript + notes
transcribe-meeting recording.m4a --speakers 4      # if you know the count
transcribe-meeting recording.m4a --no-diarize      # fastest, no speaker labels
transcribe-meeting recording.m4a --no-notes        # transcript only
meeting-notes some.transcript.md                   # (re)generate notes only
```

### Meeting-notes glossary
Notes generation applies corrections from
`~/.config/transcribe-meeting/glossary.md` — edit it to fix the acronyms and
proper nouns your meetings use (speech-to-text mangles them). A starter is
seeded on install.

### Auto-transcribe watch-folder (optional)
```
./install-watcher.sh           # enable    (./install-watcher.sh --uninstall to remove)
```
Then **drop any audio/video into `~/Documents/Transcribed Meetings/`** (or record
into it with Voice Memos) and a transcript appears automatically. It's also a
safety net: if an in-app transcription ever fails, the watcher retries it.

## Tips for best results
1. **A USB conference mic** in the center of the table beats any software change
   for a roomful of people.
2. Use `--speakers N` when you know the headcount.
3. If speakers get over-split, raise the diarization threshold; if merged, lower
   it (CLI `--threshold`, default 0.5).

## Troubleshooting
- Logs: `~/Library/Logs/transcribe-meeting.log`
- Silent recording → check **System Settings ▸ Sound ▸ Input** (right mic, not
  muted) and **Privacy & Security ▸ Microphone** (Transcribe Meeting enabled).

## Files
- `TranscribeMeetingApp.swift` — the native SwiftUI app
- `transcribe_meeting.py` — transcription + diarization pipeline
- `meeting_notes.py` — local-LLM meeting-notes generator
- `config/glossary.example.md` — starter terminology glossary
- `build-app.sh` — compiles & signs the app
- `install.sh` — installer · `install-watcher.sh` — watch-folder agent
- `tm-watch` — the auto-process scanner (transcribe → notes)
- `icon.svg` / `icon.png` / `AppIcon.icns` — app icon
