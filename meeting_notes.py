#!/usr/bin/env python3
"""Turn a meeting transcript into high-quality markdown meeting notes, fully
offline, using a local LLM via Ollama (default: gpt-oss:20b).

Aims for rich, complete minutes — not a thin abstract: a narrative Discussion
of who said what, explicit Agreements & Decisions, Action Items grouped by
person, Open Questions, and Notable Quotes. Long meetings are processed with a
map-reduce strategy (detailed per-section notes, then a synthesis pass) so
detail in the middle is preserved and there is effectively no length ceiling.

Applies terminology corrections from a glossary so speech-to-text errors
(acronyms, names, jargon) are fixed automatically.

Usage:
    meeting-notes <transcript.(md|txt)> [--out notes.md] [--title T]
                  [--date YYYY-MM-DD] [--model gpt-oss:20b]
                  [--glossary path] [--host http://localhost:11434]
                  [--single]          # force one pass (no map-reduce)
                  [--chunk-words N]    # section size for long meetings (default 3000)
"""
import argparse, datetime, json, os, sys, urllib.request, urllib.error

DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_HOST = "http://localhost:11434"
GLOSSARY = os.path.expanduser("~/.config/transcribe-meeting/glossary.md")
LOGFILE = os.path.expanduser("~/Library/Logs/transcribe-meeting.log")

# A meeting longer than this (in words) is processed section-by-section.
MAPREDUCE_THRESHOLD = 3500
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

QUALITY_BAR = """\
QUALITY BAR — these notes must be thorough, specific, and faithful:
- Capture WHO said WHAT. Attribute points, positions, concerns, and proposals
  to the person (or speaker label) who made them.
- Narrate the actual conversation and how it developed — not a vague abstract.
- Record EVERY agreement/decision and EVERY action item. Miss nothing material.
- Preserve specifics: names, numbers, dates, systems, documents, deadlines.
- Be faithful. Never invent content, decisions, names, or action items that
  aren't supported by the transcript. If something is unclear, write
  "(unclear)" rather than guessing.
- Do NOT manufacture generic "follow up" / "clarify" / meta action items that
  no one actually proposed. Only record action items that were genuinely raised.
- Apply the GLOSSARY corrections exactly; fix phonetic misspellings silently.
- Prefer completeness over brevity. Long, detailed, well-organized notes are
  the goal."""

FINAL_FORMAT = """\
OUTPUT — produce markdown in exactly this structure:

# Meeting Notes — {TITLE}

**Date:** {DATE}
**Attendees:** {names or speaker labels, with roles if stated}
**Purpose:** {1–2 sentences, or "(not specified)"}

## Summary
A substantial overview (one or two paragraphs) of what the meeting covered and
what came out of it.

## Discussion
The heart of the notes. Organize by topic with `###` subheadings. Under each,
narrate what was discussed and **who said what** — positions, reasoning,
disagreements, context. Use prose and bullets as needed. This section should be
detailed and complete.

## Agreements & Decisions
- Each decision/agreement actually reached, noting who agreed or owns it. If
  none, write "- None recorded."

## Action Items
Group by person. Number items per owner:
**{Person or Speaker}**
- [ ] {action} — {due date / context if stated}
Use **Unassigned** for actions with no clear owner. If none, "- None recorded."

## Open Questions / Parking Lot
- Anything raised but unresolved, or deferred for later. Omit if none.

## Notable Quotes
- "{short verbatim quote}" — {speaker}, on {topic}. Include only genuinely
  meaningful lines. Omit the section if there are none."""

SYSTEM_SINGLE = ("You are an expert minute-taker producing detailed, faithful "
                 "meeting minutes from a speaker-labeled transcript.\n\n"
                 + QUALITY_BAR + "\n\n" + FINAL_FORMAT)

SYSTEM_SECTION = ("You are an expert minute-taker. You are given ONE SECTION of "
                  "a longer meeting transcript. Produce detailed notes for just "
                  "this section.\n\n" + QUALITY_BAR + "\n\n"
                  "OUTPUT — markdown with these parts (omit a part only if truly "
                  "empty):\n"
                  "### Topics & discussion\nNarrate what was discussed and who "
                  "said what, attributed, with specifics.\n"
                  "### Agreements / decisions in this section\n- ...\n"
                  "### Action items in this section\n- {owner}: {action}\n"
                  "### Open questions\n- ...\n"
                  "### Notable quotes\n- \"...\" — {speaker}\n\n"
                  "Do not write an overall summary or attendee list — that is "
                  "done later. Just capture this section completely.")

SYSTEM_REDUCE = ("You are an expert minute-taker. You are given DETAILED NOTES "
                 "from consecutive sections of a SINGLE meeting, in order. Merge "
                 "them into one complete, coherent, non-repetitive minutes "
                 "document.\n\n" + QUALITY_BAR + "\n\n"
                 "Consolidate duplicates, order topics logically, and PRESERVE "
                 "all specifics, attributions, decisions, and action items from "
                 "the section notes. Do not drop detail to be concise.\n\n"
                 + FINAL_FORMAT)


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
    user.append("TRANSCRIPT:\n" + transcript)
    sysmsg = SYSTEM_SINGLE.replace("{TITLE}", title).replace("{DATE}", date)
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
        user.append("SECTION TRANSCRIPT:\n" + ch)
        ctx = ctx_for(ch, 4096)
        log(f"  section {i}/{len(chunks)} (num_ctx={ctx})")
        note = call_ollama(host, model, SYSTEM_SECTION, "\n\n".join(user), ctx, 4096)
        if note:
            section_notes.append(f"## Section {i} notes\n\n{note}")

    if not section_notes:
        return ""

    combined = "\n\n---\n\n".join(section_notes)
    sysmsg = SYSTEM_REDUCE.replace("{TITLE}", title).replace("{DATE}", date)
    user = []
    if glossary:
        user.append("GLOSSARY (apply these corrections):\n" + glossary.strip())
    user.append(f"MEETING TITLE: {title}\nMEETING DATE: {date}")
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
    ap.add_argument("transcript")
    ap.add_argument("--out")
    ap.add_argument("--title")
    ap.add_argument("--date")
    ap.add_argument("--model", default=os.environ.get("TM_NOTES_MODEL", DEFAULT_MODEL))
    ap.add_argument("--glossary", default=GLOSSARY)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", DEFAULT_HOST))
    ap.add_argument("--single", action="store_true", help="force a single pass")
    ap.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    args = ap.parse_args()

    if not os.path.exists(args.transcript):
        log(f"transcript not found: {args.transcript}")
        sys.exit(2)

    transcript = read_transcript(args.transcript)
    glossary = ""
    if args.glossary and os.path.exists(args.glossary):
        glossary = open(args.glossary, encoding="utf-8").read()

    title = args.title or os.path.basename(default_out(args.transcript)).replace(".notes.md", "")
    date = args.date or datetime.date.today().isoformat()
    out = args.out or default_out(args.transcript)

    if not ensure_model(args.host, args.model):
        log("Cannot generate notes: local LLM (Ollama) unavailable. "
            "Start it with `ollama serve` and ensure the model is pulled.")
        sys.exit(3)

    words = len(transcript.split())
    log(f"generating notes with {args.model}: {words} words, "
        f"{'single-pass' if (args.single or words <= MAPREDUCE_THRESHOLD) else 'map-reduce'}")
    try:
        if args.single or words <= MAPREDUCE_THRESHOLD:
            notes = gen_single(args.host, args.model, transcript, glossary, title, date)
        else:
            notes = gen_mapreduce(args.host, args.model, transcript, glossary,
                                  title, date, args.chunk_words)
    except Exception as e:
        log(f"notes generation failed: {e}")
        sys.exit(4)

    if not notes:
        log("model returned empty notes")
        sys.exit(5)

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(notes.rstrip() + "\n")
    log(f"notes written: {out}")
    print(out)


if __name__ == "__main__":
    main()
