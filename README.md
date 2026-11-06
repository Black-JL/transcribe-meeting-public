# Transcribe Meeting

A native macOS app that turns a meeting into **readable meeting notes**, not
just a transcript. Record in the room or on a video call (or drop in a file) and
it transcribes with **speaker labels**, then writes notes a colleague who wasn't
there can read in three minutes: what was talked about, by topic, with names and
the specifics (numbers, dates, rooms, deadlines); a **due-outs** table (who,
what, when); and open questions. You get markdown plus a Word copy. It's built
for long, multi-speaker recordings where plain Whisper drops audio and loops on
repeated words.

**Transcription always runs on your Mac.** The notes are written by **Claude**
(online, the default) or by a **local LLM through Ollama** (offline). Tick
**"Involves students — keep notes offline"** (or pass `--sensitive`) and the
notes never leave the Mac.

## How it works
1. **Transcription:** [Voz](https://desertant.com/models/voz/) via the Desert Ant
   CLI (`da voz`): NVIDIA Parakeet TDT 0.6B v3, converted to run on the Mac's
   Neural Engine. It's non-autoregressive, so it can't fall into Whisper's
   repeat-loop and doesn't drop long sections. A 59-minute file takes ~12 s at
   **~0.5 GB peak**, and the `da` process exits (freeing that memory) when the
   step ends. Model weights are under Desert Ant's source-available license.
2. **Speaker labels:**
   - **In the room:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
     diarization, using ONNX models from **GitHub** (no Hugging Face). Words are
     kept whole, so a word is never split between two speakers.
   - **Online:** your mic and the Mac's system audio (the call) are recorded as
     two tracks and labeled **Me** / **Others**, so no guessing is needed. This
     works with any video app (Zoom, Teams, a browser call).
3. **Notes:** written by Claude through the local `claude` CLI (no tools, no
   saved session), or by a local LLM via [Ollama](https://ollama.com)
   (`gpt-oss:20b` when offline, when Claude can't be reached, or with
   `--sensitive`). The same instructions are used for every model:
   - Organised by topic, each opening with a bold takeaway sentence.
   - Keeps names, roles, numbers, dates, rooms, and who pushed back or hedged.
   - Never invents attendees, owners, dates or decisions. Inferred due-outs are
     labelled as inferred. Uncertain names that matter are flagged to confirm.
   - About two pages for an hour-long meeting.
   - Your terminology glossary is applied. Optional `profile.md` (who records,
     colleagues' names and ranks) and `notes-example.md` (an approved set of
     notes, used as a style example) in `~/.config/transcribe-meeting/`.

   - Each due-out and decision carries a rough time, e.g. `(~12:40)`, so you
     can jump to that point in the recording.
   - Documents walked through in the meeting get a **Documents discussed**
     table. Goals people state, how sure the speaker sounded, implied due-outs,
     and gaps in a handover are all kept.
   - A **meeting type** (general, handover, AAR, committee, debrief) is
     detected or set, and shapes what the notes stress.
   - **Your own notes** (pasted, dictated, or a photo of handwritten notes) are
     shown first, marked ✎, and expanded from the recording.
   - **Since last time:** if an earlier meeting with the same name exists, its
     open due-outs are checked off as done, still open, or not mentioned.

   A local model is **unloaded when the notes are done**, so its memory is freed.
4. **Transcript clean-up:** before the notes are written, misheard glossary
   terms and names are fixed in the transcript, and any password or code read
   aloud is replaced with `[spoken code removed]`. The untouched original is
   kept as `<name>.transcript.raw.md`. A *Transcript hygiene* line in the notes
   says what was changed.
5. **Fact-check:** Claude lists the claims in the draft notes, checks each one
   against the transcript without seeing the draft, and fixes anything wrong,
   overstated or understated. The report is saved as `<name>.check.md`. It is
   skipped for local models, which can't do it reliably yet
   (`TM_VERIFY_LOCAL=1` forces it).
6. **Guard:** if fewer than ~100 words were spoken (a mic test, a false start),
   no notes are written, because a model would invent a meeting. The
   transcript is still kept.
7. **Recording:** AVFoundation, saved as compressed **AAC `.m4a`** (~15 MB per
   hour, versus ~115 MB for WAV). System audio uses a Core Audio process tap
   (macOS 14.2+).

Each meeting produces the recording, `<name>.transcript.md` (cleaned) and
`<name>.transcript.raw.md`, `<name>.notes.md` (the notes),
`<name>.notes.docx`, and `<name>.check.md` (the fact-check report).

## Install
```
cd transcribe-meeting-kit
./install.sh            # add --watch to also enable the auto-transcribe folder
```
This installs the Python pipeline (plus `python-docx`), downloads the
diarization models from GitHub, and builds
**`~/Applications/Transcribe Meeting.app`**. The installer downloads the Voz model (~340 MB)
once; after that, everything runs offline.

Requires: an Apple Silicon Mac on **macOS 14.2+**, Python 3, `ffmpeg`, Ollama,
and Xcode Command Line Tools (`xcode-select --install`) for the Swift build.

## Use it
Open **Transcribe Meeting.app**:
- **In the room / Online (video call):** choose before recording. For online
  meetings, wear headphones so the call doesn't echo into your mic. The first
  time, macOS asks to allow **System Audio Recording**; click Allow.
- **Record New Meeting:** optionally type a name, hit Record, talk, hit Stop.
  The mic in use is shown, with a **Change…** link. While recording you can
  jot a few points in the notes box.
- **After Stop (or after picking a file):** a short panel asks who was there,
  how many speakers, the meeting type, and your top 4–6 points. Type, paste,
  **Dictate**, or **Add Photo of Notes…** (text is read from the photo on the
  Mac). **Write Notes** goes ahead, **Skip Questions** goes ahead without
  them, **Not Now** leaves the recording for later.
- **Transcribe a File…:** pick any audio or video file.
- **Import from Recorder:** with a USB voice recorder plugged in (folders
  `VOICE/A`–`D`, e.g. Philips VoiceTracer), copies new recordings to
  `Transcribed Meetings/Recorder/`, checks each copy byte for byte, and queues
  them. Nothing is ever deleted from the recorder.
- **Result:** Open Notes (markdown), **Open in Word**, Open Transcript, Copy,
  Reveal, and *File a copy* to one of your destinations.
- **Recent Meetings:** open notes or transcripts. **Open Folder** jumps to the
  meetings folder.

Everything lives in **`~/Documents/Transcribed Meetings/`**. Quitting the app
stops any recording and any running pipeline.

### CLI
```
transcribe-meeting /path/to/recording.m4a          # transcript + notes (+ .docx)
transcribe-meeting mic.m4a --system call.m4a       # online meeting: Me / Others
transcribe-meeting recording.m4a --speakers 4      # if you know the count
transcribe-meeting recording.m4a --no-diarize      # fastest, no speaker labels
transcribe-meeting recording.m4a --no-notes        # transcript only
transcribe-meeting recording.m4a --notes-model llama3.2:3b
transcribe-meeting rec.m4a --attendees "Pat Lee, Sam Ortiz" --type handover
transcribe-meeting rec.m4a --my-notes points.txt   # or the text itself
meeting-notes some.transcript.md                   # (re)generate notes only
meeting-notes X.transcript.md --previous Y.notes.md   # or --no-previous
meeting-notes X.transcript.md --no-clean --no-verify  # skip clean-up / fact-check
meeting-notes --verify-only X.notes.md             # fact-check existing notes
meeting-notes --brief "Weekly sync"                # open items before the next one
enroll-voice "Pat Lee" clip.m4a                    # see "Voice names" below
```

### Terminology glossary
The notes apply corrections from `~/.config/transcribe-meeting/glossary.md`.
Edit it to fix the acronyms and proper nouns your meetings use, since
speech-to-text mangles them. A starter copy is created on install.

### Voice names (optional, experimental)
Enroll a person once and their speaker label becomes their name instead of
"Speaker 3". Needs at least 20 s of their speech:
```
enroll-voice "Pat Lee" clip.m4a --from 12:30 --to 14:05
enroll-voice "Pat Lee" --transcript X.transcript.md --speaker "Speaker 3"
enroll-voice --list            # enroll-voice --remove "Pat Lee"
```
Voiceprints are lists of numbers in `~/.config/transcribe-meeting/voices/`
(override with `TM_VOICES_DIR`); they can't be turned back into audio. With
none enrolled, nothing changes.

### Auto-transcribe watch-folder (optional)
```
./install-watcher.sh           # enable    (./install-watcher.sh --uninstall to remove)
```
Then **drop any audio or video into `~/Documents/Transcribed Meetings/`** (or
record into it with Voice Memos), and a transcript plus notes appear
automatically. The watcher is also a safety net: if an in-app run fails, it
retries.
- It ignores `*.system.m4a` call tracks, which are processed together with
  their mic file.
- It doesn't retry transcripts that were too short for notes (it marks them
  with a hidden `.<name>.notes-skipped` file).
- It leaves a file alone while the app or a run is working on it (a
  `<name>.hold` marker, ignored after 12 hours) and skips anything you set
  aside with **Not Now** (`<name>.later`).

## Tips for best results
1. **A USB conference mic** in the center of the table helps a roomful of
   people more than any software change.
2. Use `--speakers N` when you know the headcount.
3. If speakers get over-split, raise the diarization threshold; if they get
   merged, lower it (CLI `--threshold`, default 0.8).
4. Smaller notes models (3–4B) miss decisions and assignments on long
   meetings. `gpt-oss:20b` is the recommended default on 32 GB Macs.

## Troubleshooting
- Logs: `~/Library/Logs/transcribe-meeting.log`
- Silent recording: check **System Settings ▸ Sound ▸ Input** (right mic, not
  muted) and **Privacy & Security ▸ Microphone** (Transcribe Meeting enabled).
- Online mode can't record the call: **Privacy & Security ▸ Screen & System
  Audio Recording** must allow Transcribe Meeting.

## Files
- `TranscribeMeetingApp.swift`: the native SwiftUI app
- `SystemAudioRecorder.swift`: system-audio (call) recording via a Core Audio tap
- `transcribe_meeting.py`: transcription, speaker labels, and two-track mode
- `meeting_notes.py`: transcript clean-up, meeting notes via Claude or a
  local LLM, and the fact-check
- `enroll_voice.py`: enroll a voice so speakers are labelled by name
- `notes_docx.py`: notes to Word (Times New Roman 10 pt, tables)
- `config/glossary.example.md`: starter terminology glossary
- `build-app.sh`: compiles and signs the app
- `install.sh`: installer
- `install-watcher.sh`: watch-folder agent
- `tm-watch`: the auto-process scanner (transcribe → notes)
- `icon.svg` / `icon.png` / `AppIcon.icns`: app icon
