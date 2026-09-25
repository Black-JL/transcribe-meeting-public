#!/bin/zsh
# Build the native "Transcribe Meeting.app" from TranscribeMeetingApp.swift.
set -e
KIT="$(cd "$(dirname "$0")" && pwd)"
APP="$HOME/Applications/Transcribe Meeting.app"
BUNDLE_ID="com.jaredblack.transcribemeeting"

echo "==> compiling Swift app"
BIN="/tmp/TranscribeMeeting.bin"
swiftc -O -parse-as-library "$KIT/TranscribeMeetingApp.swift" -o "$BIN"

echo "==> assembling bundle"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/TranscribeMeeting"
chmod +x "$APP/Contents/MacOS/TranscribeMeeting"
[[ -f "$KIT/AppIcon.icns" ]] && cp "$KIT/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Transcribe Meeting</string>
  <key>CFBundleDisplayName</key><string>Transcribe Meeting</string>
  <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
  <key>CFBundleExecutable</key><string>TranscribeMeeting</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Transcribe Meeting records your meetings to create transcripts on your Mac.</string>
</dict>
</plist>
PLIST

echo "==> ad-hoc signing (no hardened runtime, so local mic works without notarization)"
codesign --force --deep -s - "$APP"
codesign -dv "$APP" 2>&1 | head -3 || true

# refresh icon + launch services
touch "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true
echo "==> done: $APP"
