#!/usr/bin/env python3
"""Second listen on due-out owner lines.

Speech-to-text and speaker labels are weakest on exactly the words that decide
who owns a task. For every due-out row that carries a timestamp (~MM:SS), this
re-transcribes that moment with a second, independent local model (Whisper
large-v3-turbo via mlx-whisper) and compares it with the transcript:

  - owner names in the row that the second listen does not hear,
  - names the second listen hears that the transcript window lacks,
  - overall word agreement for the moment.

Writes <stem>.ownercheck.md next to the notes. If any row is flagged, appends a
short "Owner lines to confirm" section to the notes. Fully local; nothing leaves
the Mac.

Usage: owner_check.py NOTES.md TRANSCRIPT.md AUDIO [--names-file directory.yml]
"""
import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile

WHISPER = "mlx-community/whisper-large-v3-turbo"
PRE, POST = 8, 22                 # seconds of audio around the timestamp
ROW_TS = re.compile(r"\(~\s*(\d{1,2}:\d{2}(?::\d{2})?)\)")
TITLES = {"SGT", "LTC", "COL", "MAJ", "CPT", "Mr", "Ms", "Mrs", "Dr", "Chaplain",
          "Colonel", "Sergeant", "Major", "The", "And", "All", "Group"}


def log(msg):
    print(f"[owner-check] {msg}", file=sys.stderr, flush=True)


def secs(ts):
    p = [int(x) for x in ts.split(":")]
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]


def words(text):
    return [w for w in (re.sub(r"[^a-z0-9']", "", t.lower()) for t in text.split()) if w]


def names_in(text):
    toks = re.findall(r"\b[A-Z][a-zA-Zé'’-]{2,}\b", text)
    return {t.strip("'’") for t in toks if t not in TITLES}


def due_rows(notes):
    """Yield (row_text, owner_cell, seconds) for due-out table rows with a timestamp."""
    lines = notes.splitlines()
    owner_col = None
    for line in lines:
        if not line.startswith("|"):
            owner_col = None if not line.strip() else owner_col
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        low = [c.lower() for c in cells]
        if "who" in low or "owner" in low:
            owner_col = low.index("who") if "who" in low else low.index("owner")
            continue
        if owner_col is None or set(line.replace("|", "").strip()) <= set("-: "):
            continue
        m = ROW_TS.search(line)
        if m and owner_col < len(cells):
            yield line, cells[owner_col].replace("*", ""), secs(m.group(1))


def transcript_window(transcript, t, pad=25):
    turns = []
    for m in re.finditer(r"\*\*(.+?)\*\* \[([\d:]+)\]\s+(.*)", transcript):
        turns.append((secs(m.group(2)), m.group(3)))
    return " ".join(txt for start, txt in turns if t - pad - 60 <= start <= t + pad)


def second_listen(audio, t, prompt):
    import mlx_whisper
    with tempfile.TemporaryDirectory() as d:
        clip = os.path.join(d, "clip.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(max(t - PRE, 0)),
                        "-t", str(PRE + POST), "-i", audio, "-ac", "1", "-ar", "16000", clip],
                       check=True)
        r = mlx_whisper.transcribe(clip, path_or_hf_repo=WHISPER, language="en",
                                   initial_prompt=prompt or None)
    return r.get("text", "").strip()


def agreement(window_words, heard_words):
    if not heard_words:
        return 0.0
    sm = difflib.SequenceMatcher(None, window_words, heard_words, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(heard_words)


def directory_names(path):
    if not path or not os.path.exists(path):
        return []
    return re.findall(r'-\s*name:\s*"([^"]+)"', open(path, encoding="utf-8").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notes"); ap.add_argument("transcript"); ap.add_argument("audio")
    ap.add_argument("--names-file", default=os.path.expanduser(
        "~/.config/transcribe-meeting/directory.yml"))
    ap.add_argument("--no-append", action="store_true", help="report only; leave notes unchanged")
    a = ap.parse_args()

    notes = open(a.notes, encoding="utf-8").read()
    transcript = open(a.transcript, encoding="utf-8").read()
    known = directory_names(a.names_file)
    known_tokens = {w for n in known for w in n.replace(".", "").split() if len(w) > 2}
    # No name hint: Whisper copies prompt text into its output when audio is unclear,
    # which would invent exactly the names we are trying to check.
    prompt = ""

    rows = list(due_rows(notes))
    if not rows:
        log("no timestamped due-out rows; nothing to check")
        return
    log(f"second listen on {len(rows)} due-out row(s)")
    report = ["# Owner lines, second listen", "",
              f"Each due-out's moment was re-transcribed with {WHISPER} (local) and compared "
              "with the transcript. Flagged rows need a human check before the notes go out.", "",
              "| Due-out (owner) | ~Time | Agreement | Flag | Second listen heard |",
              "|---|---|---|---|---|"]
    flagged = []
    for line, owner, t in rows:
        try:
            heard = second_listen(a.audio, t, prompt)
        except Exception as e:
            log(f"clip at {t}s failed: {e}"); continue
        win = transcript_window(transcript, t)
        agree = agreement(words(win), words(heard))
        owner_names = {n for n in names_in(owner) if n in known_tokens or n in names_in(win)}
        missing = sorted(n for n in owner_names if n.lower() not in heard.lower())
        extra = sorted(n for n in names_in(heard) & known_tokens if n.lower() not in win.lower())
        flags = []
        if agree < 0.70:
            flags.append(f"low agreement ({agree:.0%})")
        if extra:
            flags.append("second listen heard " + ", ".join(extra))
        if missing and agree < 0.85:
            flags.append("owner not heard: " + ", ".join(missing))
        mm = f"{t // 60:02d}:{t % 60:02d}" if t < 3600 else f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"
        what = re.sub(r"\s+", " ", owner)[:40]
        report.append(f"| {what} | {mm} | {agree:.0%} | {'; '.join(flags) or 'ok'} | "
                      f"{heard[:220].replace('|', '/')} |")
        if flags:
            flagged.append((what, mm, "; ".join(flags)))
    stem = a.notes[:-len(".notes.md")] if a.notes.endswith(".notes.md") else os.path.splitext(a.notes)[0]
    out = stem + ".ownercheck.md"
    open(out, "w", encoding="utf-8").write("\n".join(report) + "\n")
    log(f"report: {out} ({len(flagged)} flagged of {len(rows)})")
    if flagged and not a.no_append:
        add = ["", "## Owner lines to confirm", "",
               "A second, independent transcription disagreed at these moments. Listen "
               "before relying on the owner named:", ""]
        add += [f"- **{w}** at ~{mm}: {f}" for w, mm, f in flagged]
        open(a.notes, "w", encoding="utf-8").write(notes.rstrip() + "\n" + "\n".join(add) + "\n")
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import notes_docx
            notes_docx.write_md(open(a.notes, encoding="utf-8").read(), a.notes[:-3] + ".docx")
        except Exception as e:
            log(f"word copy not refreshed: {e}")


if __name__ == "__main__":
    main()
