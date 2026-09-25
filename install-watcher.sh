#!/bin/zsh
# Install (or remove with --uninstall) the auto-transcribe watch-folder agent.
set -e
KIT="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.jaredblack.transcribemeeting.watch"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DIR="$HOME/Documents/Transcribed Meetings"

if [[ "$1" == "--uninstall" ]]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Watch-folder agent removed."
  exit 0
fi

cp "$KIT/tm-watch" "$HOME/.local/bin/tm-watch"; chmod +x "$HOME/.local/bin/tm-watch"
mkdir -p "$DIR" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$HOME/.local/bin/tm-watch</string></array>
  <key>WatchPaths</key>
  <array><string>$DIR</string></array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
PL

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null || launchctl load "$PLIST"
echo "Watch-folder agent installed."
echo "Drop any audio/video into: $DIR  → a transcript appears automatically."
echo "Remove it with: $KIT/install-watcher.sh --uninstall"
