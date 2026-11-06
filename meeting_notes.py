#!/usr/bin/env python3
"""Turn a meeting transcript into readable meeting notes: what was talked
about (by topic, with names and specifics), who owes what (a due-outs table),
and open questions. Never invents attendees, owners, dates or decisions; marks
uncertain names and inferred due-outs. Long meetings use map-reduce. A Word
copy (.docx) is written next to the markdown, and the model is unloaded when
done. Fewer than ~100 spoken words -> no notes (exit 6).

Optional personal context in ~/.config/transcribe-meeting/: profile.md (who
recorded it) and notes-example.md (approved notes used as a style example).

Applies terminology corrections from a glossary so speech-to-text errors
(acronyms, names, jargon) are fixed automatically.

Usage:
    meeting-notes <transcript.(md|txt)> [--out notes.md] [--title T]
                  [--date YYYY-MM-DD] [--model gpt-oss:20b]
                  [--glossary path] [--host http://localhost:11434]
                  [--single]          # force one pass (no map-reduce)
                  [--chunk-words N]    # section size for long meetings (default 3000)
"""
import argparse, datetime, json, os, re, shutil, subprocess, sys, tempfile, urllib.request, urllib.error

# "claude" = Claude via the local `claude` CLI (online; the default for ordinary
# meetings). "claude:opus" / "claude:sonnet" pick a Claude model. Anything else
# is an Ollama model name (offline).
DEFAULT_MODEL = "claude"
LOCAL_FALLBACK = "gpt-oss:20b"   # used when Claude is unreachable or --sensitive
DEFAULT_HOST = "http://localhost:11434"
GLOSSARY = os.path.expanduser("~/.config/transcribe-meeting/glossary.md")
LOGFILE = os.path.expanduser("~/Library/Logs/transcribe-meeting.log")

# A meeting longer than this (in words) is processed section-by-section.
MAPREDUCE_THRESHOLD = 3500
# Below this many spoken words there is nothing to minute; a model would invent it.
MIN_SPOKEN_WORDS = 100
DEFAULT_CHUNK_WORDS = 3000


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} [notes] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        with open(LOGFILE, "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


# ---- Prompts -----------------------------------------------------------------
# The house style is plain, readable notes: what was talked about, who said it,
# and who owes what. Not Robert's Rules. Written for any model (local or Claude).

QUALITY_BAR = """\
WHAT GOOD NOTES ARE
Notes for a busy colleague who was not there. They should be able to read them
in three minutes and know what was discussed, what the important details were,
who said what, and who owes what to whom.

CONTENT RULES
- Organise by topic, not by the order things were said. Merge repeated mentions.
- Open each topic with a bold lead sentence that states the takeaway, then 2-5
  sentences of plain prose with the specifics.
- Timestamps: end each bold topic lead with where it starts in the
  recording, e.g. "(~12:40)", and end each due-out's "What" with the moment it
  was said or implied. Use the transcript's own timestamps; never invent one.
  The reader uses them to check the notes against the audio. Include them even
  though the style example has none.
- Keep the specifics: numbers, dates, deadlines, room numbers, systems,
  document names, contract names, capacities, time limits. These are the most
  useful part of the notes and the first thing weak notes drop.
- Keep the nuance: who pushed back, who was frustrated, who hedged, what was
  promised in the room versus afterwards, what was "no timeline, no promises".
- Attribute: say who said or proposed what, by name where the transcript makes
  it clear. Use roles too (e.g. "Pat Lee (IT office)").
- Stated goals and design intent: when the person leading says what they want
  or why ("what I would like is...", "the goal is...", "my biggest concern
  is..."), make it a bold topic lead. Keep both the goal and the concern, not
  just the concern.
- Keep each speaker's level of certainty. Do not soften it ("I'm almost
  certain" is not "possibly"; "I don't want it on here" is not "may belong
  elsewhere") and do not harden a hedge into a decision. When certainty
  matters, quote the short phrase.
- Documents: when people walk through specific documents (a handover, a
  document review, "here's the new version of X"), the "Documents discussed"
  table is REQUIRED and is the most important part of the notes, because it
  feeds a knowledge base. Put each document's details in the table rather than
  in a bulleted topic; the prose may refer to it briefly. The style example
  below may lack this table; include it anyway. Documents that are only
  mentioned, signed, cited or "to be attached" go on the short "Documents:"
  line only. In that case leave the table out.
- When a speaker enumerates items (a process, a list of problems), keep it as a
  short bulleted list under that topic.
- Length: about two printed pages for an hour-long meeting (roughly 900-1,200
  words before the due-outs table), less for shorter meetings. Merge small
  topics; drop pleasantries, logistics nobody will act on, and repetition.
- Plain English. No meeting jargon such as "motion", "disposition", "adopted",
  "call to order", "adjournment", unless the meeting actually used them.

ACCURACY RULES (these matter more than completeness)
- Never invent anything: attendees, decisions, owners, due dates, numbers.
- "In the room" lists only people the transcript says were present. Someone who
  signed or wrote a document that was mentioned is NOT an attendee; list them
  under documents.
- A due date is only a date that was stated. Otherwise write "Not stated".
- A due-out someone explicitly took on or was given goes in the table as is.
  A due-out you inferred (the next step is obvious but nobody assigned it) may
  go in the table, but list which rows are inferred in a one-line note below it.
  Conditional or future statements are implied due-outs: "it will once we get
  that document in there", "I can open up the login for you". Add them as
  inferred rows, even when the step would happen right after the meeting
  (that is its "When"); do not narrate it as already done.
- Handover gaps: when someone who is leaving describes a task they do
  personally ("I keep those on file", "I send them up to X") and no successor
  is named, add an open question: "Who does <task> after <person> leaves?"
- Spoken credentials: never write a PIN, password, passcode, login code or
  account number that was read aloud, not even partly. Instead end the notes
  with one line: "**Transcript hygiene:** a spoken code appears at ~MM:SS;
  redact the transcript and audio before sharing." (use the timestamp). If the
  transcript shows "[spoken code removed]", say instead: "a spoken code at
  ~MM:SS was removed from the transcript; the audio still contains it."
- Speech-to-text gets names, acronyms and numbers wrong. Apply the GLOSSARY.
  If a name, number or room is uncertain AND it matters (it changes a fact
  someone will rely on, or who owns a due-out), give it as heard, mark it
  with *, and add a footnote line starting "*" saying what to confirm. At most
  about five footnotes; silently leave out unclear passages that do not matter.
- Speaker labels ("Speaker 1") come from automatic software and are often
  wrong; a single talker may be split across two labels. Use names from the
  content, not the labels, and never write "Speaker N" in the notes unless no
  name can be found. A label that is a person's name (not "Speaker N") comes
  from voice recognition of an enrolled voice and is reliable.
- If the recording is one person dictating notes about a meeting (a debrief),
  say so in one italic line under the heading and write the notes about the
  meeting itself, not about the dictation.
- If the transcript does not support a section, leave the section out rather
  than filling it.
- MY NOTES, if given, are the recorder's judgment of what mattered. Every point
  in them must appear in the notes, expanded with the transcript's detail. Put
  the topics they cover first and start each such bold lead with "✎ ". A point
  the transcript does not cover stays in, as the recorder wrote it. If a point
  conflicts with the transcript, keep both and add an open question. Add one
  italic line under the heading: "*✎ marks points from the organizer's notes.*"
- Audience: the notes go to everyone who attended and to people who were not
  there. Never address the reader as "you" and never write "your notes"; say
  "the organizer's notes". Neutral tone, no asides to the recorder.
- Names: use full names and roles from the PROGRAM DIRECTORY when a transcript
  name matches one (first mention: full name and role; later: short name).
  Never invent a surname; flag an unknown full name as an open question.
- Attribution: credit a statement to a named person only when the transcript
  makes the speaker clear (they are named, they name themselves, or the
  content makes it unambiguous). Otherwise write "one participant" or "the
  group". Speaker labels alone are not enough.
- Due-outs: capture every task someone took on or was assigned, including
  routine ones (accounts, in-processing, room requests, sending a file), not
  only the large items. "When" is a date or deadline actually said; if none
  was said write "Not stated" - never infer one. Mark any inferred owner with *.
- ATTENDEES, if given, are who was in the room. Do not add or drop anyone.
- PREVIOUS MEETING'S OPEN ITEMS, if given: add a "## Since last time" section
  before "What we talked about" with a table | Due-out from last time | Status |
  where Status is "Done", "In progress", "Dropped" or "Not discussed", plus a
  few words and (~MM:SS) when it came up. Base the status only on this
  transcript or MY NOTES; never mark something done unless someone said so.
  Previous due-outs still open go into this meeting's due-outs only if they
  were re-assigned or discussed. An open question answered in this meeting
  goes in the same table with "Answered: ..."."""

MEETING_TYPES = """\
MEETING TYPE
Decide which type this meeting is (or use the TYPE given) and apply its extra
rules on top of everything above. Do not name the type in the notes.
- general: no extra rules.
- handover: someone is passing their duties or documents to others (often
  before leaving). The "Documents discussed" table is required when documents
  were walked through; every task the departing person does personally with
  no successor named becomes an open question; separate what they will finish
  themselves from what others must pick up. Group the Due-outs table by the
  departing person's functional areas: add a first column "Area" and keep rows
  for one area together.
- aar: an after action review. Under "What we talked about", organise by the
  AAR questions: what was supposed to happen, what happened, why, and then
  "Sustain" and "Improve" as two short bulleted lists, attributed.
- committee: a standing committee or board doing business. Record each
  decision as a bold lead starting "Decided:", with who proposed it and any
  dissent or vote count stated; use "motion"/"vote" only if the meeting did.
- debrief: one person dictating notes about a meeting they attended (already
  covered above): write about the meeting itself.
"""

FINAL_FORMAT = """\
OUTPUT - markdown in this structure:

# {A short descriptive title: who + subject, e.g. "Facilities & IT - building Wi-Fi meeting"}
**{DATE}** · {Hosted by X, if stated} · {Led by **Name** (role), if stated}
*{Only if a debrief or otherwise unusual: one line saying what the recording is.}*

**In the room:** {**Name** (role); ... only people stated to be present; include
the recorder's owner if they were there}
**Documents:** {documents mentioned, attached or to be attached, with who
signed or wrote them - omit if none}

---

## What we talked about

**{Takeaway sentence for topic 1.}** (~MM:SS) {Prose with specifics, attribution, nuance.}

**{Takeaway sentence for topic 2.}** {...}
- {bulleted items only where someone enumerated them}

{...one paragraph per topic; a final "**Follow-up:**" line if a next meeting
was set}

## Decisions
{Only if something was decided or explicitly tabled. Numbered list. Each item:
bold one-line decision, who proposed or agreed, (~MM:SS). Tabled items start
with "**Tabled:**" and say who will settle them. Use "motion"/"vote" only if
the meeting did.}

## Documents discussed
{Only when people walked through what specific documents contain or why they
exist; otherwise leave this section out. One row per document. Use only what was said; write
"Not stated" for anything else. Keep the short "Documents:" line above too.}

| Document | What it is / why it exists | Owner / author | Status | Belongs to | Links to | Follow-ups |
|---|---|---|---|---|---|---|
| {name; file name if said} | {purpose} | {who made or keeps it} | {new / updated / replaces X, and what changed} | {program function or process} | {related forms or documents} | {what still needs doing to it} |

## Due-outs

| Who | What | When |
|---|---|---|
| **{Name or group}** | {Specific action, with enough context to act on} (~MM:SS) | {Stated date, or "Not stated"} |

*{Rows N and M are inferred, not assigned in the meeting. - only if any}*

## Open questions
- {Unresolved items, gaps nobody owns, things to confirm}

{Footnote lines starting with "*" for anything to confirm.}

{"**Transcript hygiene:** ..." line, only if a code was spoken aloud.}"""

SYSTEM_SINGLE = ("You write meeting notes from a transcript made by "
                 "speech-to-text software.\n\n" + QUALITY_BAR + "\n\n" + MEETING_TYPES
                 + "\n\n" + FINAL_FORMAT)

SYSTEM_SECTION = ("You are given ONE SECTION of a longer meeting transcript, made by "
                  "speech-to-text software. Extract everything the final notes "
                  "will need from this section. Do not write the final notes.\n\n"
                  + QUALITY_BAR + "\n\n"
                  "OUTPUT - markdown with these parts (write \"none\" if empty):\n"
                  "### People\nNames, roles, whether present, and which speaker label they seem to be.\n"
                  "### Documents\n- name (file name if said) | what it is / why it exists | "
                  "owner or author | status (new / updated / replaces X, what changed) | "
                  "belongs to (program function or process) | links to (related forms or "
                  "documents) | follow-ups. Write \"not stated\" for any field not said.\n"
                  "### Goals and concerns\n- what the leader said they want, and why; quote "
                  "phrases that show how certain they were\n"
                  "### Topics\n- **topic** (~MM:SS where it starts): what was said, by whom, with every specific "
                  "(numbers, dates, rooms, systems) and any nuance (pushback, hedging, frustration)\n"
                  "### Due-outs\n- who: what (~MM:SS) - when (stated date or not stated) - assigned or "
                  "inferred (include conditional or future statements)\n"
                  "### Handover gaps\n- tasks a departing person does personally with no "
                  "successor named\n"
                  "### Spoken codes\n- timestamp only, never the value\n"
                  "### Uncertain\n- names, numbers or terms that may be mis-transcribed\n"
                  "### Follow-up\n- next meeting, if set")

SYSTEM_REDUCE = ("You are given notes extracted from consecutive sections of ONE "
                 "meeting, in order. Merge them into one set of meeting notes. "
                 "Combine each topic into one paragraph; a due-out mentioned in "
                 "several sections appears once.\n\n"
                 + QUALITY_BAR + "\n\n" + MEETING_TYPES + "\n\n" + FINAL_FORMAT)

# Optional personal context, kept outside the repo:
#   profile.md - who owns the recorder and their organisation, so a debrief
#                ("a meeting I just went to") is attributed to the right person.
#   notes-example.md - an approved set of notes, used as a style example only.
CONFIG_DIR = os.path.expanduser("~/.config/transcribe-meeting")


def with_context(system):
    ctx = personal_context()
    return system + ("\n\n" + ctx if ctx else "")


DIRECTORY = os.environ.get("TM_DIRECTORY", os.path.expanduser(
    "~/.config/transcribe-meeting/directory.yml"))


def directory_context():
    """Compact 'Name - role' list from the program contact directory (no emails,
    no phones), so notes use full names and correct roles."""
    if not os.path.exists(DIRECTORY):
        return ""
    rows, name = [], None
    for line in open(DIRECTORY, encoding="utf-8"):
        m = re.match(r'\s*-\s*name:\s*"([^"]+)"', line)
        if m:
            name = m.group(1); continue
        m = re.match(r'\s*role:\s*"([^"]+)"', line)
        if m and name:
            rows.append(f"- {name} - {m.group(1)}"); name = None
    return ("PROGRAM DIRECTORY (current names and roles; spellings are authoritative):\n"
            + "\n".join(rows[:120])) if rows else ""


def personal_context():
    parts = []
    d = directory_context()
    if d:
        parts.append(d)
    prof = os.path.join(CONFIG_DIR, "profile.md")
    if os.path.exists(prof):
        parts.append("ABOUT THE PERSON WHO RECORDED THIS:\n" + open(prof, encoding="utf-8").read().strip())
    ex = os.path.join(CONFIG_DIR, "notes-example.md")
    if os.path.exists(ex) and not os.environ.get("TM_NO_EXAMPLE"):
        parts.append("STYLE EXAMPLE - approved notes from a DIFFERENT meeting. Match the "
                     "format, tone and level of detail. Copy none of its content, names "
                     "or facts:\n<<<EXAMPLE\n" + open(ex, encoding="utf-8").read().strip()
                     + "\nEXAMPLE>>>")
    return "\n\n".join(parts)


# ---- Intake: what the person recording tells us ------------------------------
# Set from --attendees / --my-notes. Attendees are authoritative for who was in
# the room; MY NOTES are the recorder's judgment of what mattered (Granola-style).
INTAKE = {"attendees": "", "my_notes": "", "previous": "", "type": ""}


def meeting_key(name):
    """'G6 & DGE Meeting — 2026-10-21 15.39.notes.md' -> 'g6 & dge meeting'"""
    name = os.path.basename(name)
    name = re.split(r"\s+—\s+|\.notes|\.transcript", name)[0]
    # Ignore dates so "G6 & DGE Meeting 21 Oct 2026" matches "... 15 Dec 2026".
    name = re.sub(r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b"
                  r"|\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b", "", name)
    return re.sub(r"\s+", " ", name).strip(" -–—").lower()


def find_previous(folder, key, exclude=()):
    """Most recent notes in `folder` for the same meeting name (reviewed notes win
    over generated ones because they are written later)."""
    if not key or not os.path.isdir(folder):
        return None
    ex = {os.path.abspath(e) for e in exclude}
    cands = []
    for f in os.listdir(folder):
        full = os.path.join(folder, f)
        if (f.endswith("notes.md") and meeting_key(f) == key
                and os.path.abspath(full) not in ex and not f.startswith("~$")):
            cands.append(full)
    return max(cands, key=os.path.getmtime) if cands else None


def open_items(notes_text):
    """The Due-outs table and Open questions of a set of notes, verbatim."""
    keep = []
    for sec in ("Due-outs", "Open questions"):
        m = re.search(r"^## " + sec + r"\s*\n(.*?)(?=^## |\Z)", notes_text, re.S | re.M)
        if m:
            keep.append("## " + sec + "\n" + m.group(1).strip())
    return "\n\n".join(keep)


TRACKER_DATA = os.environ.get("TM_TRACKER_DATA", os.path.expanduser("~/.config/transcribe-meeting/tracker-data.js"))


def tracker_items(path):
    """The open items on the program due-out tracker, as a Due-outs table, so the
    notes report on each by its exact wording (tracker_propose.py then matches
    them one to one)."""
    try:
        js = ("global.window={};require(process.argv[1]);"
              "process.stdout.write(JSON.stringify(window.DUE_OUTS))")
        out = subprocess.run(["node", "-e", js, path], capture_output=True, text=True,
                             check=True, timeout=30).stdout
        items = [d for d in json.loads(out) if d.get("status") != "done"]
    except Exception as e:
        log(f"due-out tracker not read ({e})")
        return ""
    if not items:
        return ""
    rows = "\n".join(f"| **{d['who']}** | {d['what']} | {'Partly done' if d['status'] == 'partly' else 'Open'} |"
                      for d in items)
    return ("(from: the program due-out tracker. In \"Since last time\", list each item below that "
            "came up, using its wording exactly; leave out items nobody mentioned.)\n"
            "## Due-outs\n\n| Who | What | Status |\n|---|---|---|\n" + rows)


def intake_block():
    parts = []
    if INTAKE["type"]:
        parts.append("TYPE (chosen by the person who recorded): " + INTAKE["type"])
    if INTAKE["attendees"]:
        parts.append("ATTENDEES (given by the person who recorded; this list is correct "
                     "for who was in the room - use these names and spellings):\n"
                     + INTAKE["attendees"])
    if INTAKE["previous"]:
        parts.append("PREVIOUS MEETING'S OPEN ITEMS (from the last notes for this "
                     "meeting):\n" + INTAKE["previous"])
    if INTAKE["my_notes"]:
        parts.append("MY NOTES (the recorder's own key points, written during or right "
                     "after the meeting):\n<<<MY_NOTES\n" + INTAKE["my_notes"] + "\nMY_NOTES>>>")
    return "\n\n".join(parts)


# ---- Claude (online) --------------------------------------------------------

def is_claude(model):
    return model == "claude" or model.startswith("claude:")


def claude_bin():
    for c in (shutil.which("claude"), os.path.expanduser("~/.local/bin/claude"),
              "/opt/homebrew/bin/claude", "/usr/local/bin/claude"):
        if c and os.path.exists(c):
            return c
    return None


def call_claude(model, system, user):
    """One request through the `claude` CLI: no tools, no MCP servers, no saved
    session, run from a temp dir so no project instructions are picked up."""
    exe = claude_bin()
    if not exe:
        raise RuntimeError("claude CLI not found")
    cmd = [exe, "-p", "--system-prompt", system, "--tools", "",
           "--strict-mcp-config", "--no-session-persistence", "--output-format", "text"]
    if ":" in model:
        cmd += ["--model", model.split(":", 1)[1]]
    r = subprocess.run(cmd, input=user, capture_output=True, text=True,
                       cwd=tempfile.gettempdir(), timeout=1800)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-300:] or f"exit {r.returncode}")
    return r.stdout.strip()


def gen_claude(model, transcript, glossary, title, date):
    user = []
    if glossary:
        user.append("GLOSSARY (apply these corrections):\n" + glossary.strip())
    user.append(f"MEETING TITLE: {title}\nMEETING DATE: {date}")
    if intake_block():
        user.append(intake_block())
    user.append("TRANSCRIPT:\n" + transcript)
    user.append("Reply with the notes in markdown only - no preamble, no closing remarks.")
    sysmsg = with_context(SYSTEM_SINGLE.replace("{TITLE}", title).replace("{DATE}", date))
    log(f"notes via Claude ({model})")
    return call_claude(model, sysmsg, "\n\n".join(user))


# ---- Transcript clean-up ------------------------------------------------------
# Fixes go in as exact find -> replace pairs; the transcript is never rewritten
# by a model, so nothing can be dropped or paraphrased. Raw copy is kept.

CODE_MARK = "[spoken code removed]"

CLEAN_PROMPT = """\
You are given a meeting transcript made by speech-to-text software, a GLOSSARY
of known terms and a PROFILE of known people. List corrections as JSON only:
[{"from": "...", "to": "...", "kind": "term" | "name" | "code"}]
- "from" must be copied EXACTLY from the transcript (same case and spacing)
  and specific enough that replacing EVERY occurrence is correct; include a
  neighbouring word when the bare word could be right elsewhere.
- "term"/"name": a mis-heard acronym, system, product or person's name, fixed
  to the glossary or profile spelling. Only fix what you are confident of.
  Never change ordinary words, grammar, filler, or meaning.
- "code": a PIN, password, passcode, login code or account number read aloud.
  "from" is exactly the spoken digits or characters; "to" is "%s".
Output [] if there is nothing to fix. JSON only, no commentary.""" % CODE_MARK


TERMS_OF_REFERENCE = os.environ.get("TM_TERMS_OF_REFERENCE", os.path.expanduser("~/.config/transcribe-meeting/terms-of-reference.md"))
GLOSSARY_BUILDER = os.environ.get("TM_GLOSSARY_BUILDER", os.path.expanduser("~/.config/transcribe-meeting/build_glossary.py"))


def refresh_glossary(path):
    """One glossary: Terms of Reference.md (Box) is the list people edit. When it
    is newer than the meeting glossary, rebuild the glossary from it first."""
    try:
        if path != GLOSSARY or not (os.path.exists(TERMS_OF_REFERENCE) and os.path.exists(GLOSSARY_BUILDER)):
            return
        if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(TERMS_OF_REFERENCE):
            return
        import subprocess
        subprocess.run([sys.executable, GLOSSARY_BUILDER, "--meeting-out", path,
                        "--site-out", os.devnull], check=True, capture_output=True, timeout=60)
        log("glossary rebuilt from Terms of Reference.md")
    except Exception as e:                       # never block the notes on this
        log(f"glossary not rebuilt ({e}); using the existing one")


def glossary_pairs(glossary):
    """Deterministic fixes from glossary lines like:  - ACME — ... (NOT "ACNE", "AKMEE")"""
    pairs = []
    for line in glossary.splitlines():
        m = re.match(r'\s*-\s*([^—\-:]+?)\s+[—-]', line)
        if not m:
            continue
        right = m.group(1).strip()
        wrongs = []
        for paren in re.findall(r'\((?:NOT|not)\s+([^)]*)\)', line):
            wrongs += re.findall(r'"([^"]+)"', paren)
        wrongs += re.findall(r'transcripts say "([^"]+)"', line)
        # Skip ordinary lowercase phrases ("point of interest"): too likely to be right as said.
        pairs += [(w, right) for w in wrongs
                  if w.lower() != right.lower() and not (w.islower() and " " in w)]
    return pairs


def apply_pairs(text, pairs, word_bounded=True):
    applied = []
    for wrong, right in pairs:
        pat = re.escape(wrong)
        if word_bounded:
            pat = r'(?<![\w])' + pat + r'(?![\w])'
        new, n = re.subn(pat, lambda _m: right, text,
                         flags=re.IGNORECASE if word_bounded else 0)
        if n:
            text = new
            applied.append((wrong, right, n))
    return text, applied


def clean_transcript(model, host, transcript, glossary, use_model=True):
    """Returns (clean_text, report_lines, fixes) where fixes = [(wrong, right, word_bounded)]."""
    gpairs = glossary_pairs(glossary)
    text, applied = apply_pairs(transcript, gpairs)
    fixes = [(w, r, True) for w, r, _ in applied]
    report = [f'glossary: "{w}" -> "{r}" ({n}x)' for w, r, n in applied]
    if not use_model:
        return text, report, fixes
    user = []
    if glossary:
        user.append("GLOSSARY:\n" + glossary.strip())
    prof = os.path.join(CONFIG_DIR, "profile.md")
    if os.path.exists(prof):
        user.append("PROFILE:\n" + open(prof, encoding="utf-8").read().strip())
    user.append("TRANSCRIPT:\n" + text)
    try:
        raw = ask(model, host, CLEAN_PROMPT, "\n\n".join(user), 4096)
        m = re.search(r"\[.*\]", raw, re.S)
        found = json.loads(m.group(0)) if m else []
    except Exception as e:
        log(f"transcript clean-up: model step failed ({e}); glossary fixes only")
        return text, report, fixes
    pairs = []
    for f in found:
        frm, to, kind = str(f.get("from", "")), str(f.get("to", "")), f.get("kind", "")
        if kind == "code":
            to = CODE_MARK
        if len(frm) < 3 or frm == to or frm not in text:
            continue
        pairs.append((frm, to))
    pairs.sort(key=lambda p: -len(p[0]))   # longest first: "Askren A" before "Askren"
    text, applied = apply_pairs(text, pairs, word_bounded=False)
    for w, r, n in applied:
        fixes.append((w, r, False))
        report.append(f'model: "{"<code>" if r == CODE_MARK else w}" -> "{r}" ({n}x)')
    return text, report, fixes


def clean_transcript_files(path, model, host, glossary, use_model):
    """Clean <base>.transcript.txt (and .md with the same fixes) in place, keeping
    <base>.transcript.raw.*. Returns (clean_txt or None, report_lines)."""
    base = None
    for suf in (".transcript.md", ".transcript.txt"):
        if path.endswith(suf):
            base = path[:-len(suf)]
    if base is None:
        return None, []
    txt, md = f"{base}.transcript.txt", f"{base}.transcript.md"
    src_file = txt if os.path.exists(txt) else md
    for f in (txt, md):
        raw = f.replace(".transcript.", ".transcript.raw.")
        if os.path.exists(f) and not os.path.exists(raw):
            shutil.copyfile(f, raw)
    log("cleaning transcript (glossary" + (" + names/codes" if use_model else "") + ")")
    src = open(src_file.replace(".transcript.", ".transcript.raw."), encoding="utf-8").read()
    clean, report, fixes = clean_transcript(model, host, src, glossary, use_model)
    with open(src_file, "w", encoding="utf-8") as fh:
        fh.write(clean)
    if src_file == txt and os.path.exists(md):
        other = open(md.replace(".transcript.", ".transcript.raw."), encoding="utf-8").read()
        for w, r, bounded in fixes:
            other, _ = apply_pairs(other, [(w, r)], word_bounded=bounded)
        with open(md, "w", encoding="utf-8") as fh:
            fh.write(other)
    log(f"transcript clean-up: {len(report)} fix(es); raw copy kept as .transcript.raw.*")
    return clean, report


# ---- Fact-check pass ----------------------------------------------------------
# Chain-of-verification: (1) list the draft's checkable claims, (2) answer each
# claim from the transcript WITHOUT seeing the draft, so the draft's mistakes
# aren't repeated, (3) revise the draft using only those verdicts.

VERIFY_PLAN = """\
You are given draft meeting notes. List every checkable claim in them, one per
line, numbered. A claim is one atomic fact a reader might rely on:
- who was present, and each person's name, rank and role
- every number, date, time, room, deadline, count, document name
- who said, proposed, promised or owns what (each due-out row is a claim:
  its owner, its action, its date, and whether it was assigned or inferred)
- every quoted phrase, and how certain the speaker was
- each bold topic takeaway
Do NOT list the title, date or recording length in the heading lines (they come
from the file, not the transcript). When a claim carries a level of certainty
(possibly, may, will, firmly, "almost certain"), keep that wording in the claim.
Write each claim as a short neutral statement that can be checked against a
transcript. Do not judge them. Output only the numbered list."""

VERIFY_ANSWER = """\
You are checking claims against a meeting transcript made by speech-to-text
software (speaker labels are unreliable; names, acronyms and numbers may be
mis-heard; a GLOSSARY of known corrections may be given). For each numbered
claim, answer on one line:
N. SUPPORTED (~MM:SS)
N. WRONG (~MM:SS): what the transcript actually says
N. OVERSTATED (~MM:SS): the transcript is more hedged/limited; say how
N. UNDERSTATED (~MM:SS): the speaker was more certain or firmer than the claim
   says (e.g. claim "may", speaker "almost certain" or "I don't want it"); quote them
N. NOT FOUND: nothing in the transcript supports it
Use the transcript's own timestamps. An inferred due-out that is labelled
inferred is SUPPORTED if the transcript gives grounds for it. A name corrected
by the glossary or the profile counts as SUPPORTED. A person is present if they
speak or are spoken to in the transcript, or appear in ATTENDEES; that is
SUPPORTED. Any claim that MY NOTES state is SUPPORTED (notes), marked ✎ or
not, even if the transcript doesn't cover it. Output only the list."""

VERIFY_REVISE = """\
You are given draft meeting notes and a fact-check of their claims against the
transcript. Produce the corrected notes:
- Fix every WRONG claim to what the transcript says.
- Soften every OVERSTATED claim to the transcript's level of certainty.
- Strengthen every UNDERSTATED claim to the speaker's level of certainty,
  quoting their short phrase.
- Remove every NOT FOUND claim, or, if it is an inferred due-out, keep it and
  make sure it is listed as inferred.
- Never remove or weaken a point that comes from MY NOTES (the recorder's own
  notes); those are the recorder's own record.
- Never delete the heading lines, the "In the room" or "Documents" lines, a
  section or a table; fix or remove only the individual name, phrase or row.
- Change nothing else: same structure, headings, wording, order and length.
Reply with the full corrected notes in markdown only - no preamble, no list of
changes."""


def ask(model, host, system, user, num_predict=8192):
    if is_claude(model):
        return call_claude(model, system, user)
    return call_ollama(host, model, system, user, ctx_for(user, num_predict), num_predict)


def verify_notes(model, host, transcript, glossary, draft):
    """Returns (revised_notes, check_report). On any failure, the draft unchanged."""
    if not is_claude(model) and len(transcript.split()) > MAPREDUCE_THRESHOLD:
        log("fact-check skipped: transcript too long for the local model's context")
        return draft, ""
    try:
        log("fact-check 1/3: listing claims")
        claims = ask(model, host, VERIFY_PLAN, "DRAFT NOTES:\n" + draft, 4096)
        log("fact-check 2/3: checking claims against the transcript")
        user = []
        if glossary:
            user.append("GLOSSARY:\n" + glossary.strip())
        prof = os.path.join(CONFIG_DIR, "profile.md")
        if os.path.exists(prof):
            user.append("PROFILE (known people and spellings):\n"
                        + open(prof, encoding="utf-8").read().strip())
        if intake_block():
            user.append(intake_block())
        user += ["TRANSCRIPT:\n" + transcript, "CLAIMS:\n" + claims]
        verdicts = ask(model, host, VERIFY_ANSWER, "\n\n".join(user), 4096)
        problems = [l for l in verdicts.splitlines()
                    if re.search(r"\b(WRONG|OVERSTATED|UNDERSTATED|NOT FOUND)\b", l)]
        report = ("# Fact-check\n\n## Claims\n\n" + claims + "\n\n## Verdicts\n\n"
                  + verdicts + "\n")
        if not problems:
            log("fact-check 3/3: all claims supported; no changes")
            return draft, report
        log(f"fact-check 3/3: fixing {len(problems)} claim(s)")
        revised = ask(model, host, VERIFY_REVISE,
                      (intake_block() + "\n\n" if INTAKE["my_notes"] else "")
                      + "DRAFT NOTES:\n" + draft + "\n\nCLAIMS:\n" + claims
                      + "\n\nFACT-CHECK (problems only, by claim number):\n"
                      + "\n".join(problems), 8192)
        if not revised or len(revised) < 0.6 * len(draft):
            log("fact-check revision looked truncated; keeping the draft")
            return draft, report
        return revised, report
    except Exception as e:
        log(f"fact-check failed ({e}); keeping the draft")
        return draft, ""


# ---- Ollama ------------------------------------------------------------------

def call_ollama(host, model, system, user, num_ctx, num_predict):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": num_ctx, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.loads(resp.read())
    return data.get("message", {}).get("content", "").strip()


def unload_model(host, model):
    """Free the model's memory as soon as the notes are done (Ollama otherwise
    keeps it loaded for 5 minutes — ~13 GB for gpt-oss:20b)."""
    try:
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        urllib.request.urlopen(urllib.request.Request(
            host.rstrip("/") + "/api/generate", data=body,
            headers={"Content-Type": "application/json"}), timeout=30).read()
    except Exception:
        pass


def ensure_model(host, model):
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=10) as r:
            tags = json.loads(r.read())
        names = [m.get("name", "") for m in tags.get("models", [])]
        if not any(n == model or n.startswith(model.split(":")[0]) for n in names):
            log(f"WARNING: model '{model}' not found in Ollama. Available: {names}")
        return True
    except Exception as e:
        log(f"Ollama not reachable at {host}: {e}")
        return False


def ctx_for(text, output_tokens):
    est = len(text) // 4 + output_tokens + 600
    for ctx in (8192, 16384, 32768, 65536, 131072):
        if est <= ctx:
            return ctx
    return 131072


# ---- Chunking ----------------------------------------------------------------

def chunk_transcript(text, max_words):
    """Split into ~max_words chunks on line boundaries (keeps speaker turns)."""
    lines = [ln for ln in text.split("\n")]
    chunks, cur, n = [], [], 0
    for ln in lines:
        w = len(ln.split())
        if n + w > max_words and cur:
            chunks.append("\n".join(cur).strip())
            cur, n = [], 0
        cur.append(ln); n += w
    if cur and "\n".join(cur).strip():
        chunks.append("\n".join(cur).strip())
    return [c for c in chunks if c]


# ---- Generation --------------------------------------------------------------

def gen_single(host, model, transcript, glossary, title, date):
    user = []
    if glossary:
        user.append("GLOSSARY (apply these corrections):\n" + glossary.strip())
    user.append(f"MEETING TITLE: {title}\nMEETING DATE: {date}")
    if intake_block():
        user.append(intake_block())
    user.append("TRANSCRIPT:\n" + transcript)
    sysmsg = with_context(SYSTEM_SINGLE.replace("{TITLE}", title).replace("{DATE}", date))
    ctx = ctx_for(transcript, 8192)
    log(f"single-pass notes (num_ctx={ctx})")
    return call_ollama(host, model, sysmsg, "\n\n".join(user), ctx, 8192)


def gen_mapreduce(host, model, transcript, glossary, title, date, chunk_words):
    chunks = chunk_transcript(transcript, chunk_words)
    log(f"map-reduce: {len(chunks)} sections (~{chunk_words} words each)")
    section_notes = []
    for i, ch in enumerate(chunks, 1):
        user = []
        if glossary:
            user.append("GLOSSARY (apply these corrections):\n" + glossary.strip())
        user.append(f"MEETING: {title} ({date}) — SECTION {i} of {len(chunks)}")
        if INTAKE["attendees"]:
            user.append("ATTENDEES:\n" + INTAKE["attendees"])
        user.append("SECTION TRANSCRIPT:\n" + ch)
        ctx = ctx_for(ch, 4096)
        log(f"  section {i}/{len(chunks)} (num_ctx={ctx})")
        note = call_ollama(host, model, SYSTEM_SECTION, "\n\n".join(user), ctx, 4096)
        if note:
            section_notes.append(f"## Section {i} notes\n\n{note}")

    if not section_notes:
        return ""

    combined = "\n\n---\n\n".join(section_notes)
    sysmsg = with_context(SYSTEM_REDUCE.replace("{TITLE}", title).replace("{DATE}", date))
    user = []
    if glossary:
        user.append("GLOSSARY (apply these corrections):\n" + glossary.strip())
    user.append(f"MEETING TITLE: {title}\nMEETING DATE: {date}")
    if intake_block():
        user.append(intake_block())
    user.append("SECTION NOTES (in order):\n\n" + combined)

    # If the combined notes are very large, reduce in groups first.
    if len(combined) // 4 > 28000:
        log("section notes large; doing a two-level reduce")
        groups, g, gw = [], [], 0
        for sn in section_notes:
            w = len(sn) // 4
            if gw + w > 18000 and g:
                groups.append("\n\n---\n\n".join(g)); g, gw = [], 0
            g.append(sn); gw += w
        if g:
            groups.append("\n\n---\n\n".join(g))
        partials = []
        for j, grp in enumerate(groups, 1):
            log(f"  intermediate reduce {j}/{len(groups)}")
            pu = f"MEETING: {title} ({date})\n\nSECTION NOTES (in order):\n\n{grp}"
            partials.append(call_ollama(host, model, SYSTEM_REDUCE.replace("{TITLE}", title).replace("{DATE}", date), pu, 32768, 8192))
        combined = "\n\n---\n\n".join(p for p in partials if p)
        user = []
        if glossary:
            user.append("GLOSSARY (apply these corrections):\n" + glossary.strip())
        user.append(f"MEETING TITLE: {title}\nMEETING DATE: {date}")
        user.append("CONSOLIDATED NOTES (in order):\n\n" + combined)

    ctx = ctx_for(combined, 8192)
    log(f"reduce/synthesis pass (num_ctx={ctx})")
    return call_ollama(host, model, sysmsg, "\n\n".join(user), ctx, 8192)


# ---- I/O ---------------------------------------------------------------------

def read_transcript(path):
    if path.endswith(".transcript.md"):
        txt = path[:-len(".transcript.md")] + ".transcript.txt"
        if os.path.exists(txt):
            path = txt
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def default_out(transcript_path):
    base = transcript_path
    for suf in (".transcript.md", ".transcript.txt"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    else:
        base = os.path.splitext(base)[0]
    return base + ".notes.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--title")
    ap.add_argument("--date")
    ap.add_argument("--model", default=os.environ.get("TM_NOTES_MODEL", DEFAULT_MODEL))
    ap.add_argument("--glossary", default=GLOSSARY)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    ap.add_argument("--single", action="store_true", help="force a single pass")
    ap.add_argument("--sensitive", action="store_true",
                    help="involves students or other protected information: never send to Claude; use a local model")
    ap.add_argument("--verify-only", metavar="NOTES",
                    help="fact-check existing notes against the transcript; writes --out")
    ap.add_argument("--attendees", help='who was in the room, e.g. "Pat Lee, Sam Ortiz, Dr. Rivera"')
    ap.add_argument("--my-notes", help="your own key points: a text/markdown file, or the text itself")
    ap.add_argument("--previous", help="notes of the previous meeting in this series "
                    "(default: the latest notes with the same meeting name in the same folder)")
    ap.add_argument("--no-previous", action="store_true", help="don't look for a previous meeting")
    ap.add_argument("--tracker", nargs="?", const=TRACKER_DATA, metavar="TABLETOP_DATA_JS",
                    help="also carry forward the open items on the program due-out tracker "
                         "(TM_TRACKER_DATA), so 'Since last time' reports on them by their exact wording")
    ap.add_argument("--brief", metavar="MEETING_NAME",
                    help="print the open items from the last meeting with this name, then exit")
    ap.add_argument("--type", choices=["auto", "general", "handover", "aar", "committee", "debrief"],
                    default="auto", help="meeting type (default: let the notes step decide)")
    ap.add_argument("--no-clean", action="store_true",
                    help="leave the transcript as transcribed (no glossary/name fixes, no code removal)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the fact-check pass (faster, less safe)")
    ap.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    args = ap.parse_args()

    if args.brief:
        folder = os.path.expanduser("~/Documents/Transcribed Meetings")
        prev = find_previous(folder, meeting_key(args.brief))
        if not prev:
            print(f"No earlier notes named “{args.brief}” in {folder}.")
            return
        items = open_items(open(prev, encoding="utf-8").read())
        print(f"Open items from {os.path.basename(prev)}:\n\n" + (items or "(none recorded)"))
        return
    if not args.transcript:
        ap.error("give a transcript (or --brief NAME)")

    if not os.path.exists(args.transcript):
        log(f"transcript not found: {args.transcript}")
        sys.exit(2)

    transcript = read_transcript(args.transcript)
    glossary = ""
    refresh_glossary(args.glossary)
    if args.glossary and os.path.exists(args.glossary):
        glossary = open(args.glossary, encoding="utf-8").read()

    title = args.title or os.path.basename(default_out(args.transcript)).replace(".notes.md", "")
    date = args.date or datetime.date.today().isoformat()
    out = args.out or default_out(args.transcript)

    # Tell the folder watcher (tm-watch) this meeting is being handled.
    hold = (out[:-len(".notes.md")] if out.endswith(".notes.md") else os.path.splitext(out)[0]) + ".hold"
    made_hold = not os.path.exists(hold)
    if made_hold:
        try:
            open(hold, "w").close()
            import atexit
            atexit.register(lambda: os.path.exists(hold) and os.remove(hold))
        except OSError:
            pass

    model = args.model
    if args.sensitive and is_claude(model):
        log(f"sensitive meeting: not sending to Claude; using local model {LOCAL_FALLBACK}")
        model = LOCAL_FALLBACK
    if is_claude(model) and not claude_bin():
        log(f"claude CLI not found; falling back to local model {LOCAL_FALLBACK}")
        model = LOCAL_FALLBACK
    args.model = model

    if not is_claude(args.model) and not ensure_model(args.host, args.model):
        log("Cannot generate notes: local LLM (Ollama) unavailable. "
            "Start it with `ollama serve` and ensure the model is pulled.")
        sys.exit(3)

    if args.type != "auto":
        INTAKE["type"] = args.type
    if args.attendees:
        INTAKE["attendees"] = args.attendees.strip()
    if not args.no_previous and not args.verify_only:
        prev = args.previous or find_previous(
            os.path.dirname(os.path.abspath(out)), meeting_key(title),
            exclude=[out, default_out(args.transcript)])
        if prev and os.path.exists(prev):
            items = open_items(open(prev, encoding="utf-8").read())
            if items:
                INTAKE["previous"] = f"(from: {os.path.basename(prev)})\n" + items
                log(f"carrying forward open items from: {os.path.basename(prev)}")
    if args.tracker:
        block = tracker_items(args.tracker)
        if block:
            INTAKE["previous"] = (INTAKE["previous"] + "\n\n" if INTAKE["previous"] else "") + block
            log("carrying forward the open items on the due-out tracker")
    if args.my_notes:
        mn = args.my_notes
        INTAKE["my_notes"] = (open(mn, encoding="utf-8").read() if os.path.isfile(mn) else mn).strip()

    if args.verify_only:
        draft = open(args.verify_only, encoding="utf-8").read()
        notes, rep = verify_notes(args.model, args.host, transcript, glossary, draft)
        if not is_claude(args.model):
            unload_model(args.host, args.model)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(notes.rstrip() + "\n")
        with open(os.path.splitext(out)[0] + ".check.md", "w", encoding="utf-8") as fh:
            fh.write(rep)
        print(out)
        return

    if not args.no_clean:
        cleaned, _rep = clean_transcript_files(args.transcript, args.model, args.host,
                                               glossary, use_model=is_claude(args.model))
        if cleaned is not None:
            transcript = cleaned

    words = len(transcript.split())
    spoken = len(re.sub(r"^#.*$|\*\*[^*]+\*\*|\[[\d:]+\]|^[^\n\[]*\[[\d:]+\]:",
                        " ", transcript, flags=re.M).split())
    if spoken < MIN_SPOKEN_WORDS:
        log(f"notes skipped: only {spoken} words were spoken — too little for minutes")
        sys.exit(6)
    mode = ("single-pass" if (is_claude(args.model) or args.single or words <= MAPREDUCE_THRESHOLD)
            else "map-reduce")
    log(f"generating notes with {args.model}: {words} words, {mode}")
    notes = ""
    if is_claude(args.model):
        try:
            notes = gen_claude(args.model, transcript, glossary, title, date)
        except Exception as e:
            log(f"Claude notes failed ({e}); falling back to local model {LOCAL_FALLBACK}")
            args.model = LOCAL_FALLBACK
            if not ensure_model(args.host, args.model):
                log("local model unavailable too; no notes written")
                sys.exit(3)
    try:
        if notes:
            pass
        elif args.single or words <= MAPREDUCE_THRESHOLD:
            notes = gen_single(args.host, args.model, transcript, glossary, title, date)
        else:
            notes = gen_mapreduce(args.host, args.model, transcript, glossary,
                                  title, date, args.chunk_words)
    except Exception as e:
        log(f"notes generation failed: {e}")
        if not is_claude(args.model):
            unload_model(args.host, args.model)
        sys.exit(4)
    check_report = ""
    if notes and not args.no_verify and not is_claude(args.model) and not os.environ.get("TM_VERIFY_LOCAL"):
        log("fact-check skipped: the offline model can't do it reliably yet (set TM_VERIFY_LOCAL=1 to force)")
    elif notes and not args.no_verify:
        notes, check_report = verify_notes(args.model, args.host, transcript, glossary, notes)
    if not is_claude(args.model):
        unload_model(args.host, args.model)

    if not notes:
        log("model returned empty notes")
        sys.exit(5)

    if CODE_MARK in transcript:
        stamps = re.findall(r"\[(\d+:\d+(?::\d+)?)\][^\n]*" + re.escape(CODE_MARK), transcript)
        where = ", ".join("~" + t for t in stamps) or "one point"
        notes = re.sub(r"\n*\*\*Transcript hygiene:\*\*[^\n]*", "", notes).rstrip()
        notes += (f"\n\n**Transcript hygiene:** a spoken code at {where} was removed from "
                  "the transcript. The audio still contains it; don't share the audio as is.")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(notes.rstrip() + "\n")
    log(f"notes written: {out}")
    if check_report:
        check_out = (out[:-len(".notes.md")] if out.endswith(".notes.md")
                     else os.path.splitext(out)[0]) + ".check.md"
        with open(check_out, "w", encoding="utf-8") as fh:
            fh.write(check_report)
        log(f"fact-check report: {check_out}")
    # Word copy next to the markdown (never fails the run).
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import notes_docx
        docx_out = (out[:-3] if out.endswith(".md") else out) + ".docx"
        notes_docx.write_md(notes, docx_out)
        log(f"word copy: {docx_out}")
    except Exception as e:
        log(f"word copy failed (notes .md is fine): {e}")
    print(out)


if __name__ == "__main__":
    main()
