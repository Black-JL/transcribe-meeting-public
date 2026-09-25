#!/bin/zsh
# One-command installer for Transcribe Meeting.
# Works on any Apple Silicon Mac (M1–M5, incl. Mac Studio). Fully local:
# parakeet-mlx runs transcription on the GPU; sherpa-onnx does speaker labels
# with models from GitHub (no Hugging Face account/token).
set -e
KIT="$(cd "$(dirname "$0")" && pwd)"
echo "==> Installing Transcribe Meeting from: $KIT"

# 1. Python with packages
PYBIN="${PYBIN:-$(command -v python3)}"
[[ -n "$PYBIN" ]] || { echo "No python3 found. Install Python 3 first."; exit 1; }
echo "==> Python: $PYBIN"

# 2. ffmpeg (audio decode)
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then brew install ffmpeg
  else echo "WARNING: ffmpeg missing and no Homebrew. Install ffmpeg manually."; fi
fi

# 3. Python deps
echo "==> Installing Python packages (parakeet-mlx, sherpa-onnx, soundfile)"
"$PYBIN" -m pip install --upgrade parakeet-mlx sherpa-onnx soundfile

# 4. Pipeline + notes generator + CLI wrappers
mkdir -p "$HOME/.claude/tools" "$HOME/.local/bin" "$HOME/.config/transcribe-meeting"
cp "$KIT/transcribe_meeting.py" "$HOME/.claude/tools/transcribe_meeting.py"
cp "$KIT/meeting_notes.py" "$HOME/.claude/tools/meeting_notes.py"
cp "$KIT/tm-watch" "$HOME/.local/bin/tm-watch"; chmod +x "$HOME/.local/bin/tm-watch"
[[ -f "$HOME/.config/transcribe-meeting/glossary.md" ]] || \
  cp "$KIT/config/glossary.example.md" "$HOME/.config/transcribe-meeting/glossary.md"
[[ -f "$HOME/.config/transcribe-meeting/destinations.json" ]] || \
  cp "$KIT/config/destinations.example.json" "$HOME/.config/transcribe-meeting/destinations.json"
cat > "$HOME/.local/bin/transcribe-meeting" <<EOF
#!/bin/zsh
exec "$PYBIN" "\$HOME/.claude/tools/transcribe_meeting.py" "\$@"
EOF
cat > "$HOME/.local/bin/meeting-notes" <<EOF
#!/bin/zsh
exec "$PYBIN" "\$HOME/.claude/tools/meeting_notes.py" "\$@"
EOF
chmod +x "$HOME/.local/bin/transcribe-meeting" "$HOME/.local/bin/meeting-notes"
echo "==> CLIs installed: ~/.local/bin/{transcribe-meeting,meeting-notes}"

# 4b. Local LLM for meeting notes (Ollama + gpt-oss:20b)
NOTES_MODEL="${TM_NOTES_MODEL:-gpt-oss:20b}"
if command -v ollama >/dev/null 2>&1; then
  if ! ollama list 2>/dev/null | grep -q "$NOTES_MODEL"; then
    echo "==> Pulling local notes model $NOTES_MODEL (~13 GB, one time)"
    ollama pull "$NOTES_MODEL" || echo "    (pull failed — notes skipped until a model is available)"
  fi
else
  echo "    NOTE: Ollama not found — transcription works, but meeting notes need it."
  echo "    Install from https://ollama.com then: ollama pull $NOTES_MODEL"
fi

# 5. Diarization models from GitHub (no Hugging Face)
MODELS="$HOME/.config/transcribe-meeting/models"; mkdir -p "$MODELS"; cd "$MODELS"
if [[ ! -f "sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]]; then
  echo "==> Downloading segmentation model"
  curl -fL -o seg.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar xjf seg.tar.bz2 && rm -f seg.tar.bz2
fi
if [[ ! -f "nemo_en_titanet_small.onnx" ]]; then
  echo "==> Downloading speaker-embedding model"
  curl -fL -o nemo_en_titanet_small.onnx "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx"
fi

# 6. Build the native app
echo "==> Building the app"
"$KIT/build-app.sh"

# 7. Optional: install the auto-transcribe watch-folder agent
if [[ "$1" == "--watch" ]]; then
  echo "==> Installing watch-folder auto-transcribe agent"
  "$KIT/install-watcher.sh"
fi

echo ""
echo "==> Done."
echo "    App:  ~/Applications/Transcribe Meeting.app   (double-click)"
echo "    CLI:  transcribe-meeting /path/to/recording.m4a   (transcript + meeting notes)"
echo "    Each meeting yields: recording, <name>.transcript.md, and <name>.notes.md"
echo "    Files live in: ~/Documents/Transcribed Meetings/"
echo "    First run downloads the parakeet model (~1–2 GB) once, then it's cached."
