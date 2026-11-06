// Transcribe Meeting — a native macOS app.
// Record a meeting from the mic (AVFoundation → proper TCC prompt) or pick a
// file, then: transcribe + diarize locally, and summarize into meeting notes
// with a local LLM (Ollama). The meeting NOTES are the headline deliverable.
import SwiftUI
import AVFoundation
import AppKit
import Vision
import CoreAudio

// MARK: - Model

struct Meeting: Identifiable {
    let id = UUID()
    let title: String
    let date: Date
    let notesURL: URL?       // <stem>.notes.md   (preferred)
    let transcriptURL: URL?  // <stem>.transcript.md
    var primaryURL: URL { notesURL ?? transcriptURL! }
}

struct Destination: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let pathTemplate: String   // may contain {year}
    let isDefault: Bool
    func resolved(for date: Date) -> URL {
        let df = DateFormatter(); df.dateFormat = "yyyy"
        let p = pathTemplate.replacingOccurrences(of: "{year}", with: df.string(from: date))
        return URL(fileURLWithPath: (p as NSString).expandingTildeInPath)
    }
}

// MARK: - Engine

final class Engine: NSObject, ObservableObject {
    static weak var shared: Engine?
    override init() { super.init(); Engine.shared = self }

    @Published var isRecording = false
    @Published var online = UserDefaults.standard.bool(forKey: "onlineMeeting") {
        didSet { UserDefaults.standard.set(online, forKey: "onlineMeeting") }
    }
    @Published var busy = false
    @Published var status = "Ready — record a meeting or transcribe a file."
    @Published var stage = ""
    @Published var elapsed = 0
    @Published var notesPath = ""       // primary result (notes, or transcript if notes failed)
    @Published var history: [Meeting] = []
    @Published var meetingName = ""
    @Published var destinations: [Destination] = []
    @Published var selectedDestination: Destination?
    @Published var actionMsg = ""
    @Published var notesModels: [String] = []
    // Notes writer: "claude" (online, default) or an Ollama model (offline).
    // notesModelV2: one-time switch of existing installs to Claude (2026-10-21).
    @Published var selectedModel: String = Engine.initialModel() {
        didSet { UserDefaults.standard.set(selectedModel, forKey: "notesModel") }
    }
    // Sensitive meetings (e.g. about students or patients): notes never go to Claude.
    @Published var sensitive = false
    private var lastRunSensitive = false   // the switch as it was when the notes were made
    // Intake: asked after Stop (or after choosing a file), before notes are written.
    @Published var myNotes = ""          // typed live while recording, or after
    @Published var attendees = ""
    @Published var speakers = 0          // 0 = let the software count
    @Published var meetingType = "auto"
    @Published var showIntake = false
    @Published var ocrBusy = false
    private var pending: (url: URL, title: String, system: URL?, outDir: URL?)?
    @Published var pendingIsImport = false
    @Published var recorderMounted = false
    @Published var importQueue: [URL] = []
    @Published var inputDevice = ""

    static func initialModel() -> String {
        let d = UserDefaults.standard
        if !d.bool(forKey: "notesModelV2") {
            d.set(true, forKey: "notesModelV2")
            d.set("claude", forKey: "notesModel")
            return "claude"
        }
        return d.string(forKey: "notesModel") ?? "claude"
    }

    static func modelLabel(_ m: String) -> String {
        m == "claude" ? "Claude (online)" : "\(m) (offline)"
    }

    private var recorder: AVAudioRecorder?
    private let sysRec = SystemAudioRecorder()
    private var currentURL: URL?
    private var systemURL: URL?
    private var proc: Process?
    private var timer: Timer?
    // Watchdog clock: systemUptime does not advance while the Mac sleeps, so a
    // closed lid no longer looks like a stalled job.
    private var lastActivity = ProcessInfo.processInfo.systemUptime
    private var sleepBlock: NSObjectProtocol?

    var meetingsDir: URL {
        let d = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/Transcribed Meetings")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    private var tool: String {
        FileManager.default.homeDirectoryForCurrentUser.path + "/.local/bin/transcribe-meeting"
    }

    // MARK: Recording

    func toggleRecord() { isRecording ? stopRecording() : startRecording() }

    func startRecording() {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                guard granted else {
                    self.status = "Microphone access was denied. Enable “Transcribe Meeting” in System Settings ▸ Privacy & Security ▸ Microphone, then try again."
                    return
                }
                self.beginRecording()
            }
        }
    }

    private func sanitized(_ s: String) -> String {
        let bad = CharacterSet(charactersIn: "/:\\")
        return s.components(separatedBy: bad).joined(separator: "-")
            .trimmingCharacters(in: .whitespaces)
    }

    private func beginRecording() {
        let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd HH.mm"
        let stamp = df.string(from: Date())
        let name = sanitized(meetingName)
        let stem = name.isEmpty ? "Meeting \(stamp)" : "\(name) — \(stamp)"
        // AAC .m4a: ~15 MB/hour instead of ~115 MB/hour for WAV; same transcript.
        let url = meetingsDir.appendingPathComponent("\(stem).m4a")
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 32000,
        ]
        do {
            let r = try AVAudioRecorder(url: url, settings: settings)
            guard r.record() else { status = "Could not start recording."; return }
            if online {
                let sysURL = meetingsDir.appendingPathComponent("\(stem).system.m4a")
                do { try sysRec.start(to: sysURL); systemURL = sysURL } catch {
                    r.stop()
                    status = "Couldn’t record the call audio: \(error.localizedDescription) Allow Transcribe Meeting under System Settings ▸ Privacy & Security ▸ Screen & System Audio Recording."
                    return
                }
            } else { systemURL = nil }
            recorder = r; currentURL = url
            isRecording = true
            status = online ? "● Recording you and the call audio…  click Stop when the meeting ends."
                            : "● Recording…  click Stop when the meeting ends."
            startTimer()
        } catch {
            status = "Could not start recording: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        recorder?.stop(); recorder = nil; isRecording = false
        if systemURL != nil { sysRec.stop() }
        stopTimer()
        guard let url = currentURL else { return }
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int ?? 0
        if size < 4000 {
            status = "The recording was silent. Check System Settings ▸ Sound ▸ Input (use your real mic, not muted/virtual), then try again."
            return
        }
        pending = (url, meetingName.isEmpty ? url.deletingPathExtension().lastPathComponent : meetingName,
                   systemURL, nil)
        pendingIsImport = false
        // Hold marker: the folder watcher leaves this file alone while you add notes.
        FileManager.default.createFile(atPath: url.deletingPathExtension().path + ".hold", contents: nil)
        showIntake = true
    }

    /// Called from the intake panel.
    func finishIntake(skip: Bool, notNow: Bool = false) {
        showIntake = false
        guard let p = pending else { return }
        pending = nil
        let title = (pendingIsImport && !meetingName.isEmpty) ? meetingName : p.title
        if notNow {
            // ".later": the folder watcher never auto-processes it; you decide.
            let stem = (p.outDir ?? p.url.deletingLastPathComponent())
                .appendingPathComponent(p.url.deletingPathExtension().lastPathComponent).path
            FileManager.default.createFile(atPath: stem + ".later", contents: nil)
            try? FileManager.default.removeItem(atPath: p.url.deletingPathExtension().path + ".hold")
            myNotes = ""; attendees = ""; speakers = 0; meetingType = "auto"
            if pendingIsImport { meetingName = "" }
            status = "Left for later: \(p.url.lastPathComponent). Use Transcribe a File… when ready."
            nextImport()
            return
        }
        var extra: [String] = []
        if let o = p.outDir { extra += ["--out", o.path] }
        if !skip {
            let names = attendees.trimmingCharacters(in: .whitespacesAndNewlines)
            if !names.isEmpty { extra += ["--attendees", names] }
            if speakers > 0 { extra += ["--speakers", String(speakers)] }
            else if !names.isEmpty {
                // No count picked: use the number of attendee names typed.
                let n = names.split(whereSeparator: { $0 == "," || $0 == ";" || $0 == "\n" })
                    .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }.count
                if n > 1 { extra += ["--speakers", String(n)] }
            }
            if meetingType != "auto" { extra += ["--type", meetingType] }
            let notes = myNotes.trimmingCharacters(in: .whitespacesAndNewlines)
            if !notes.isEmpty {
                let stem = p.url.deletingPathExtension().lastPathComponent
                let f = meetingsDir.appendingPathComponent("\(stem).my-notes.md")
                try? notes.write(to: f, atomically: true, encoding: .utf8)
                extra += ["--my-notes", f.path]
            }
        }
        myNotes = ""; attendees = ""; speakers = 0; meetingType = "auto"
        if pendingIsImport { meetingName = "" }
        runPipeline(p.url, title: title, system: p.system, extra: extra)
    }

    /// Read a photo of handwritten or printed notes on this Mac (Apple Vision).
    func addNotesPhoto() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.message = "Choose a photo of your notes"
        guard panel.runModal() == .OK else { return }
        let urls = panel.urls
        ocrBusy = true
        DispatchQueue.global().async {
            let text = urls.map { Engine.recognizeText($0) }.filter { !$0.isEmpty }.joined(separator: "\n")
            DispatchQueue.main.async {
                self.ocrBusy = false
                if text.isEmpty { self.actionMsg = "Couldn’t read any text in that photo."; return }
                self.myNotes += (self.myNotes.isEmpty || self.myNotes.hasSuffix("\n") ? "" : "\n") + text
            }
        }
    }

    static func recognizeText(_ url: URL) -> String {
        guard let img = NSImage(contentsOf: url),
              let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return "" }
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = true
        try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
        return (req.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    }

    /// Start macOS dictation in whichever text box has focus.
    func dictate() {
        NSApp.sendAction(Selector(("startDictation:")), to: nil, from: nil)
    }

    /// Quit cleanly: stop recording, stop any running pipeline.
    func shutdown() {
        if isRecording { recorder?.stop(); if systemURL != nil { sysRec.stop() } }
        proc?.terminate()
    }

    // MARK: Philips VoiceTracer (or any recorder with a VOICE folder)

    func recorderVoiceDirs() -> [URL] {
        let fm = FileManager.default
        let vols = (try? fm.contentsOfDirectory(atPath: "/Volumes")) ?? []
        var out: [URL] = []
        for v in vols {
            let voice = URL(fileURLWithPath: "/Volumes/\(v)/VOICE")
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: voice.path, isDirectory: &isDir), isDir.boolValue {
                for sub in ((try? fm.contentsOfDirectory(atPath: voice.path)) ?? []).sorted() {
                    out.append(voice.appendingPathComponent(sub))
                }
            }
        }
        return out
    }

    func checkRecorder() { recorderMounted = !recorderVoiceDirs().isEmpty }

    /// Copy new recordings off the recorder, verify each copy, never delete from it.
    func importFromRecorder() {
        let fm = FileManager.default
        let dest = meetingsDir.appendingPathComponent("Recorder")
        try? fm.createDirectory(at: dest, withIntermediateDirectories: true)
        let audio: Set<String> = ["mp3", "wav"]
        var copied: [URL] = [], failed = 0, already = 0
        for dir in recorderVoiceDirs() {
            let folder = dir.lastPathComponent
            for name in ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).sorted() {
                let src = dir.appendingPathComponent(name)
                guard audio.contains(src.pathExtension.lowercased()) else { continue }
                let attrs = try? fm.attributesOfItem(atPath: src.path)
                let size = (attrs?[.size] as? Int) ?? 0
                if size < 32_000 { continue }                       // under a second or two
                let when = (attrs?[.modificationDate] as? Date) ?? Date()
                let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd HH.mm"
                let base = "Recorder \(df.string(from: when)) (\(folder) \(src.deletingPathExtension().lastPathComponent))"
                let target = dest.appendingPathComponent(base + "." + src.pathExtension.lowercased())
                if fm.fileExists(atPath: target.path) {
                    if fm.contentsEqual(atPath: src.path, andPath: target.path) { already += 1; continue }
                }
                do {
                    if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
                    try fm.copyItem(at: src, to: target)
                    if fm.contentsEqual(atPath: src.path, andPath: target.path) { copied.append(target) }
                    else { failed += 1; try? fm.removeItem(at: target) }
                } catch { failed += 1 }
            }
        }
        var msg = "Copied \(copied.count) new recording\(copied.count == 1 ? "" : "s") from the recorder (each copy checked; nothing deleted from the recorder)."
        if already > 0 { msg += " \(already) already copied before." }
        if failed > 0 { msg += " \(failed) FAILED to copy — leave them on the recorder." }
        status = msg; actionMsg = msg
        importQueue = copied
        nextImport()
    }

    /// Open the notes panel for the next imported recording, if any.
    func nextImport() {
        guard !busy, !showIntake, !importQueue.isEmpty else { return }
        let url = importQueue.removeFirst()
        pending = (url, url.deletingPathExtension().lastPathComponent, nil, meetingsDir)
        pendingIsImport = true
        meetingName = ""
        showIntake = true
    }

    /// Name of the microphone macOS will record from (the system default input).
    func refreshInputDevice() {
        var dev = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var addr = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultInputDevice,
                                              mScope: kAudioObjectPropertyScopeGlobal,
                                              mElement: kAudioObjectPropertyElementMain)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &dev) == noErr
        else { inputDevice = ""; return }
        var name: Unmanaged<CFString>?
        var nsize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var naddr = AudioObjectPropertyAddress(mSelector: kAudioObjectPropertyName,
                                               mScope: kAudioObjectPropertyScopeGlobal,
                                               mElement: kAudioObjectPropertyElementMain)
        if AudioObjectGetPropertyData(dev, &naddr, 0, nil, &nsize, &name) == noErr, let n = name {
            inputDevice = n.takeRetainedValue() as String
        }
    }

    func openSoundInput() {
        if let u = URL(string: "x-apple.systempreferences:com.apple.Sound-Settings.extension?input") {
            NSWorkspace.shared.open(u)
        }
    }

    // MARK: File import

    func pickAndTranscribe() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Choose an audio or video file to transcribe"
        if panel.runModal() == .OK, let url = panel.url {
            pending = (url, url.deletingPathExtension().lastPathComponent, nil, nil)
            pendingIsImport = false
            FileManager.default.createFile(atPath: url.deletingPathExtension().path + ".hold", contents: nil)
            showIntake = true
        }
    }

    // MARK: Pipeline (transcribe → notes). Robust: drains stderr, watchdog.

    private func runPipeline(_ url: URL, title: String, system: URL? = nil, extra: [String] = []) {
        busy = true; elapsed = 0; stage = "Starting…"; actionMsg = ""
        lastRunSensitive = sensitive
        status = "Processing “\(title)” … Keep the lid open or stay plugged in until it finishes."
        startTimer()
        lastActivity = ProcessInfo.processInfo.systemUptime
        // Keep the Mac from idle-sleeping while the job runs. (Closing the lid on
        // battery still forces sleep; the watchdog clock above tolerates that.)
        sleepBlock = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled],
            reason: "Transcribing a meeting")

        DispatchQueue.global().async {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: self.tool)
            p.arguments = [url.path, "--title", title, "--notes-model", self.selectedModel]
                + (system.map { ["--system", $0.path] } ?? [])
                + (self.sensitive ? ["--sensitive"] : [])
                + extra
            let outPipe = Pipe(); let errPipe = Pipe()
            p.standardOutput = outPipe; p.standardError = errPipe

            // Drain stderr live (large: Voz/ffmpeg/notes progress) so the
            // child never blocks on a full pipe buffer. Map lines to stages.
            errPipe.fileHandleForReading.readabilityHandler = { fh in
                let d = fh.availableData
                if d.isEmpty { return }
                self.lastActivity = ProcessInfo.processInfo.systemUptime
                if let s = String(data: d, encoding: .utf8) {
                    for line in s.split(separator: "\n") {
                        if let friendly = self.friendlyStage(String(line)) {
                            DispatchQueue.main.async { self.stage = friendly }
                        }
                    }
                }
            }

            self.proc = p
            do { try p.run() } catch {
                errPipe.fileHandleForReading.readabilityHandler = nil
                DispatchQueue.main.async {
                    self.busy = false; self.stopTimer()
                    self.status = "Couldn’t start the transcriber: \(error.localizedDescription)"
                }
                return
            }
            // stdout is just the final path (tiny) — safe to read to EOF.
            let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            errPipe.fileHandleForReading.readabilityHandler = nil
            let out = String(data: outData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let last = out.split(separator: "\n").last.map(String.init) ?? ""
            let code = p.terminationStatus

            DispatchQueue.main.async {
                self.busy = false; self.stopTimer(); self.proc = nil
                if code == 0 && !last.isEmpty && FileManager.default.fileExists(atPath: last) {
                    self.notesPath = last
                    let isNotes = last.hasSuffix(".notes.md")
                    self.status = isNotes ? "Done ✓  Meeting notes are ready."
                                          : "Done ✓  Transcript ready (notes step was skipped)."
                    self.stage = ""
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: last)])
                    self.loadHistory()
                } else {
                    self.status = "Processing didn’t finish. See ~/Library/Logs/transcribe-meeting.log"
                    self.stage = ""
                }
                self.nextImport()
            }
        }
    }

    private func friendlyStage(_ line: String) -> String? {
        let l = line.lowercased()
        if l.contains("converting audio") { return "Step 1 of 4 — preparing audio…" }
        if l.contains("transcribing with voz") { return "Step 2 of 4 — transcribing on the Neural Engine…" }
        if l.contains("transcribed") && l.contains("words") { return "Step 2 of 4 — transcribed; tidying up…" }
        if l.contains("diarizing") { return "Step 3 of 4 — identifying who spoke…" }
        if l.contains("generating meeting notes") { return "Step 4 of 4 — writing meeting notes (local AI)…" }
        return nil
    }

    // MARK: Timer / watchdog

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            if self.isRecording || self.busy { self.elapsed += 1 }
            if self.busy, ProcessInfo.processInfo.systemUptime - self.lastActivity > 1200 {
                self.proc?.terminate()
                self.status = "Processing stalled and was stopped. See the log."
                self.busy = false; self.stopTimer()
            }
        }
    }
    private func stopTimer() {
        timer?.invalidate(); timer = nil
        if let b = sleepBlock { ProcessInfo.processInfo.endActivity(b); sleepBlock = nil }
    }

    // MARK: History + helpers

    func loadHistory() {
        let fm = FileManager.default
        let items = (try? fm.contentsOfDirectory(at: meetingsDir,
            includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        struct Pair { var notes: URL?; var transcript: URL?; var date: Date }
        var byStem: [String: Pair] = [:]
        for url in items {
            let name = url.lastPathComponent
            let stem: String
            if name.hasSuffix(".notes.md") { stem = String(name.dropLast(".notes.md".count)) }
            else if name.hasSuffix(".transcript.md") { stem = String(name.dropLast(".transcript.md".count)) }
            else { continue }
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate ?? Date.distantPast
            var p = byStem[stem] ?? Pair(notes: nil, transcript: nil, date: date)
            if name.hasSuffix(".notes.md") { p.notes = url } else { p.transcript = url }
            if date > p.date { p.date = date }
            byStem[stem] = p
        }
        let meetings: [Meeting] = byStem.map { (stem, p) in
            var title = stem
            if let primary = p.notes ?? p.transcript,
               let first = (try? String(contentsOf: primary, encoding: .utf8))?
                   .split(separator: "\n").first, first.hasPrefix("# ") {
                title = String(first.dropFirst(2))
            }
            return Meeting(title: title, date: p.date, notesURL: p.notes, transcriptURL: p.transcript)
        }.sorted { $0.date > $1.date }
        history = Array(meetings.prefix(20))
    }

    // MARK: Filing destinations (the "what next" action menu)

    func loadDestinations() {
        let cfg = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/transcribe-meeting/destinations.json")
        guard let data = try? Data(contentsOf: cfg),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let arr = obj["destinations"] as? [[String: Any]] else {
            destinations = []; selectedDestination = nil; return
        }
        destinations = arr.compactMap { d in
            guard let name = d["name"] as? String, let path = d["path"] as? String else { return nil }
            return Destination(name: name, pathTemplate: path, isDefault: (d["default"] as? Bool) ?? false)
        }
        selectedDestination = destinations.first(where: { $0.isDefault }) ?? destinations.first
    }

    /// Query Ollama for installed models so the user can pick / A-B test.
    func loadModels() {
        guard let url = URL(string: "http://localhost:11434/api/tags") else { return }
        URLSession.shared.dataTask(with: url) { data, _, _ in
            var names: [String] = []
            if let data = data,
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let arr = obj["models"] as? [[String: Any]] {
                names = arr.compactMap { $0["name"] as? String }.sorted()
            }
            // Show only the offline models listed in notes-models.txt (if present).
            let listURL = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".config/transcribe-meeting/notes-models.txt")
            if let txt = try? String(contentsOf: listURL, encoding: .utf8) {
                let wanted = txt.split(separator: "\n")
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty && !$0.hasPrefix("#") }
                names = names.filter { wanted.contains($0) }
            }
            DispatchQueue.main.async {
                names = ["claude"] + names
                if !names.contains(self.selectedModel) { self.selectedModel = "claude" }
                self.notesModels = names
            }
        }.resume()
    }

    private func slug(_ s: String) -> String {
        let allowed = CharacterSet(charactersIn:
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        let parts = s.unicodeScalars.map { allowed.contains($0) ? Character($0) : " " }
        return String(parts).split(separator: " ").joined(separator: "-").lowercased()
    }

    /// File a copy of the current notes into the chosen destination, using the
    /// meeting-log naming convention. Returns nothing; sets actionMsg.
    func fileToDestination() {
        guard !notesPath.isEmpty, let dest = selectedDestination else { return }
        // Box meeting folders are searched by the knowledge base. Student meetings stay out
        // until someone reviews and de-identifies them (FERPA).
        if (sensitive || lastRunSensitive) && dest.pathTemplate.contains("Box-Box") {
            actionMsg = "Not filed: this meeting involves students, and your knowledge base searches the Box folders. Review and de-identify it first."
            return
        }
        let notes = URL(fileURLWithPath: notesPath)
        let fm = FileManager.default
        let date = (try? notes.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate ?? Date()
        let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd"
        // derive a topic slug from the meeting name / file stem
        var stem = notes.lastPathComponent
        if stem.hasSuffix(".notes.md") { stem = String(stem.dropLast(".notes.md".count)) }
        let topic = slug(meetingName.isEmpty ? stem : meetingName)
        let base = "meeting-log-\(df.string(from: date))" + (topic.isEmpty ? "" : "-\(topic)")
        let folder = dest.resolved(for: date)
        do {
            try fm.createDirectory(at: folder, withIntermediateDirectories: true)
            var target = folder.appendingPathComponent(base + ".md")
            var n = 2
            while fm.fileExists(atPath: target.path) {
                target = folder.appendingPathComponent("\(base)-\(n).md"); n += 1
            }
            try fm.copyItem(at: notes, to: target)
            actionMsg = "Filed to \(dest.name)"
            NSWorkspace.shared.activateFileViewerSelecting([target])
        } catch {
            actionMsg = "Couldn’t file: \(error.localizedDescription)"
        }
    }

    /// Email placeholder (recipients/send deferred): copy notes + open a draft.
    func prepareEmail() {
        guard !notesPath.isEmpty else { return }
        let notes = URL(fileURLWithPath: notesPath)
        let body = (try? String(contentsOf: notes, encoding: .utf8)) ?? ""
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(body, forType: .string)
        var stem = notes.lastPathComponent
        if stem.hasSuffix(".notes.md") { stem = String(stem.dropLast(".notes.md".count)) }
        let subject = (meetingName.isEmpty ? stem : meetingName)
            .addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "Meeting Notes"
        if let url = URL(string: "mailto:?subject=\(subject)") { NSWorkspace.shared.open(url) }
        actionMsg = "Notes copied — paste into the draft. (One-click send coming later.)"
    }

    func openMeetingsFolder() { NSWorkspace.shared.open(meetingsDir) }
    func reveal(_ url: URL) { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    func open(_ url: URL) { NSWorkspace.shared.open(url) }
    func copyText(_ url: URL) {
        var src = url
        if url.lastPathComponent.hasSuffix(".transcript.md") {
            let txt = url.deletingPathExtension().deletingPathExtension()
                .appendingPathExtension("transcript").appendingPathExtension("txt")
            if FileManager.default.fileExists(atPath: txt.path) { src = txt }
        }
        let body = (try? String(contentsOf: src, encoding: .utf8)) ?? ""
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(body, forType: .string)
    }
    func docxFor(_ notes: URL) -> URL? {
        guard notes.lastPathComponent.hasSuffix(".notes.md") else { return nil }
        let d = notes.deletingPathExtension().appendingPathExtension("docx")
        return FileManager.default.fileExists(atPath: d.path) ? d : nil
    }

    func transcriptFor(_ notes: URL) -> URL? {
        guard notes.lastPathComponent.hasSuffix(".notes.md") else { return nil }
        let stem = String(notes.lastPathComponent.dropLast(".notes.md".count))
        let t = notes.deletingLastPathComponent().appendingPathComponent(stem + ".transcript.md")
        return FileManager.default.fileExists(atPath: t.path) ? t : nil
    }

    func elapsedString() -> String {
        let m = elapsed / 60, s = elapsed % 60
        return String(format: "%d:%02d", m, s)
    }
}

// MARK: - Views

struct IntakeSheet: View {
    @ObservedObject var engine: Engine

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Before I write the notes").font(.title3).fontWeight(.semibold)
            Text("All optional. Anything you add makes the notes more accurate.")
                .font(.caption).foregroundStyle(.secondary)
            if engine.pendingIsImport {
                Text("Meeting name").fontWeight(.medium)
                TextField("e.g. Weekly team sync", text: $engine.meetingName)
                    .textFieldStyle(.roundedBorder)
            }
            Toggle(isOn: $engine.sensitive) {
                Text("Involves students — keep everything offline (notes by the Mac's own AI)").fontWeight(.medium)
            }.toggleStyle(.checkbox)

            Text("Who was in the room").fontWeight(.medium)
            TextField("e.g. Pat Lee, Sam Ortiz, Dr. Rivera", text: $engine.attendees)
                .textFieldStyle(.roundedBorder)
            HStack {
                Text("People who spoke")
                Picker("", selection: $engine.speakers) {
                    Text("Let it count").tag(0)
                    ForEach(1..<13) { n in Text("\(n)").tag(n) }
                }.labelsHidden().frame(width: 130)
                Spacer()
                Text("Type")
                Picker("", selection: $engine.meetingType) {
                    Text("Let it decide").tag("auto")
                    Text("Handover").tag("handover")
                    Text("AAR").tag("aar")
                    Text("Committee / board").tag("committee")
                    Text("Debrief (just me)").tag("debrief")
                    Text("General").tag("general")
                }.labelsHidden().frame(width: 150)
            }.font(.callout)

            Text("My key notes").fontWeight(.medium)
            Text("The 4–6 points that mattered most. Type, paste, add a photo of handwritten notes, or dictate.")
                .font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $engine.myNotes)
                .font(.body).frame(minHeight: 140)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            HStack {
                Button { engine.addNotesPhoto() } label: { Label("Add Photo of Notes…", systemImage: "camera") }
                    .disabled(engine.ocrBusy)
                Button { engine.dictate() } label: { Label("Dictate", systemImage: "mic") }
                    .help("Click in the notes box first, then Dictate (or press the dictation key)")
                if engine.ocrBusy { ProgressView().controlSize(.small); Text("Reading photo…").font(.caption) }
                Spacer()
            }.controlSize(.small)

            HStack {
                Button("Not Now") { engine.finishIntake(skip: true, notNow: true) }
                    .help("Leave this recording unprocessed for now")
                Button("Skip Questions") { engine.finishIntake(skip: true) }
                Spacer()
                Button("Write Notes") { engine.finishIntake(skip: false) }
                    .keyboardShortcut(.defaultAction).buttonStyle(.borderedProminent)
            }
        }
        .padding(20).frame(width: 480)
    }
}

struct ContentView: View {
    @StateObject var engine = Engine()

    var body: some View {
        VStack(spacing: 0) {
            header
            ScrollView {
                VStack(spacing: 16) {
                    actionCard
                    if engine.busy || engine.isRecording { statusCard }
                    if !engine.notesPath.isEmpty && !engine.busy { resultCard }
                    historyCard
                }
                .padding(18)
            }
        }
        .frame(minWidth: 500, minHeight: 580)
        .sheet(isPresented: $engine.showIntake) { IntakeSheet(engine: engine) }
        .onAppear {
            engine.loadHistory(); engine.loadDestinations(); engine.loadModels()
            engine.checkRecorder(); engine.refreshInputDevice()
        }
        .onReceive(NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.didMountNotification)) { _ in engine.checkRecorder() }
        .onReceive(NSWorkspace.shared.notificationCenter.publisher(for: NSWorkspace.didUnmountNotification)) { _ in engine.checkRecorder() }
        .onReceive(NSApplication.shared.publisher(for: \.isActive)) { _ in engine.refreshInputDevice() }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Image(systemName: "waveform.and.mic")
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(.white)
            VStack(alignment: .leading, spacing: 1) {
                Text("Transcribe Meeting").font(.system(size: 20, weight: .bold)).foregroundStyle(.white)
                Text("Record → transcribe → meeting notes · 100% local").font(.caption).foregroundStyle(.white.opacity(0.9))
            }
            Spacer()
        }
        .padding(.horizontal, 20).padding(.vertical, 16)
        .background(LinearGradient(colors: [Color(red: 0.36, green: 0.36, blue: 0.94),
                                            Color(red: 0.09, green: 0.78, blue: 0.71)],
                                   startPoint: .topLeading, endPoint: .bottomTrailing))
    }

    private var actionCard: some View {
        VStack(spacing: 12) {
            TextField("Meeting name (optional) — e.g. “Weekly team sync”", text: $engine.meetingName)
                .textFieldStyle(.roundedBorder)
                .disabled(engine.isRecording || engine.busy)
            Picker("", selection: $engine.online) {
                Text("In the room").tag(false)
                Text("Online (video call)").tag(true)
            }
            .pickerStyle(.segmented).labelsHidden()
            .disabled(engine.isRecording || engine.busy)
            if engine.online {
                Text("Wear headphones so the call doesn’t echo into your mic.")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Button(action: { engine.toggleRecord() }) {
                Label(engine.isRecording ? "Stop Recording" : "Record New Meeting",
                      systemImage: engine.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(maxWidth: .infinity).padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent).tint(engine.isRecording ? .red : .accentColor)
            .disabled(engine.busy)
            if engine.recorderMounted {
                Button(action: { engine.importFromRecorder() }) {
                    Label("Import from Recorder", systemImage: "externaldrive.badge.plus")
                        .frame(maxWidth: .infinity).padding(.vertical, 5)
                }
                .buttonStyle(.bordered).disabled(engine.busy || engine.isRecording)
            }
            Button(action: { engine.pickAndTranscribe() }) {
                Label("Transcribe a File…", systemImage: "doc.text.magnifyingglass")
                    .frame(maxWidth: .infinity).padding(.vertical, 5)
            }
            .buttonStyle(.bordered).disabled(engine.busy || engine.isRecording)

            HStack(spacing: 6) {
                Image(systemName: "mic").foregroundStyle(.secondary)
                Text(engine.inputDevice.isEmpty ? "Microphone: unknown" : "Microphone: \(engine.inputDevice)")
                    .foregroundStyle(.secondary).lineLimit(1)
                Button("Change…") { engine.openSoundInput() }.buttonStyle(.link)
                Spacer()
            }
            .controlSize(.small).font(.caption)
            HStack(spacing: 6) {
                Image(systemName: "brain").foregroundStyle(.secondary)
                Text("Notes AI").foregroundStyle(.secondary)
                Picker("", selection: $engine.selectedModel) {
                    ForEach(engine.notesModels, id: \.self) { Text(Engine.modelLabel($0)).tag($0) }
                }.labelsHidden()
                Button { engine.loadModels() } label: { Image(systemName: "arrow.clockwise") }
                    .help("Refresh the list of offline models")
            }
            .controlSize(.small).font(.caption)
            .disabled(engine.busy || engine.isRecording)
            Toggle(isOn: $engine.sensitive) {
                Text("Involves students — keep notes offline")
            }
            .toggleStyle(.checkbox).controlSize(.small).font(.caption)
            .frame(maxWidth: .infinity, alignment: .leading)
            .disabled(engine.busy || engine.isRecording)
            if engine.sensitive && engine.selectedModel == "claude" {
                Text("Notes will be written by gpt-oss:20b on this Mac, not Claude.")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(16).background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private var statusCard: some View {
        VStack(spacing: 8) {
            HStack {
                if engine.isRecording {
                    Circle().fill(.red).frame(width: 10, height: 10)
                    Text("Recording").fontWeight(.semibold)
                } else {
                    ProgressView().controlSize(.small)
                    Text(engine.stage.isEmpty ? "Working…" : engine.stage).fontWeight(.medium)
                }
                Spacer()
                Text(engine.elapsedString()).monospacedDigit().foregroundStyle(.secondary)
            }
            Text(engine.status).font(.callout).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
            if engine.isRecording {
                Text("My notes — jot the 4–6 points that matter; the AI expands them from the recording.")
                    .font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                TextEditor(text: $engine.myNotes)
                    .font(.body).frame(minHeight: 90)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            }
        }
        .padding(14).background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private var resultCard: some View {
        let primary = URL(fileURLWithPath: engine.notesPath)
        let isNotes = engine.notesPath.hasSuffix(".notes.md")
        let transcript = engine.transcriptFor(primary)
        return VStack(alignment: .leading, spacing: 10) {
            Label(isNotes ? "Meeting notes ready" : "Transcript ready",
                  systemImage: "checkmark.seal.fill").foregroundStyle(.green).fontWeight(.semibold)
            Text(engine.notesPath).font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.secondary).textSelection(.enabled).lineLimit(2)
            HStack {
                Button { engine.open(primary) } label: { Label(isNotes ? "Open Notes" : "Open", systemImage: "doc.richtext") }
                if isNotes, let docx = engine.docxFor(primary) {
                    Button { engine.open(docx) } label: { Label("Open in Word", systemImage: "doc.text") }
                }
                if let t = transcript {
                    Button { engine.open(t) } label: { Label("Open Transcript", systemImage: "doc.plaintext") }
                }
                Button { engine.copyText(primary) } label: { Label("Copy", systemImage: "doc.on.clipboard") }
                Button { engine.reveal(primary) } label: { Label("Reveal", systemImage: "folder") }
            }.controlSize(.small)

            if isNotes {
                Divider().padding(.vertical, 2)
                Text("What next?").font(.subheadline).fontWeight(.semibold)
                HStack(spacing: 8) {
                    if !engine.destinations.isEmpty {
                        Picker("", selection: $engine.selectedDestination) {
                            ForEach(engine.destinations) { d in
                                Text(d.name).tag(Optional(d))
                            }
                        }.labelsHidden().frame(maxWidth: 280)
                        Button { engine.fileToDestination() } label: {
                            Label("File a copy", systemImage: "tray.and.arrow.down")
                        }.buttonStyle(.borderedProminent)
                    }
                    Button { engine.prepareEmail() } label: {
                        Label("Email…", systemImage: "envelope")
                    }
                }.controlSize(.small)
                if !engine.actionMsg.isEmpty {
                    Text(engine.actionMsg).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .padding(14).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }

    private var historyCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Recent Meetings").font(.headline)
                Spacer()
                Button { engine.openMeetingsFolder() } label: {
                    Label("Open Folder", systemImage: "folder")
                }.controlSize(.small)
            }
            if engine.history.isEmpty {
                Text("No meetings yet. Recordings, transcripts, and notes are all saved in your Meetings folder.")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                ForEach(engine.history) { m in
                    HStack(spacing: 8) {
                        Image(systemName: m.notesURL != nil ? "doc.richtext" : "doc.plaintext")
                            .foregroundStyle(m.notesURL != nil ? Color.accentColor : Color.secondary)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(m.title).lineLimit(1)
                            Text(m.date.formatted(date: .abbreviated, time: .shortened))
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if let n = m.notesURL {
                            Button { engine.open(n) } label: { Image(systemName: "doc.richtext") }
                                .help("Open meeting notes")
                        }
                        if let t = m.transcriptURL {
                            Button { engine.open(t) } label: { Image(systemName: "doc.plaintext") }
                                .help("Open transcript")
                        }
                        Button { engine.copyText(m.primaryURL) } label: { Image(systemName: "doc.on.clipboard") }
                            .help("Copy text")
                        Button { engine.reveal(m.primaryURL) } label: { Image(systemName: "magnifyingglass") }
                            .help("Reveal in Finder")
                    }
                    .buttonStyle(.borderless).controlSize(.small)
                    .padding(.vertical, 4)
                    Divider()
                }
            }
        }
        .padding(14).frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) { Engine.shared?.shutdown() }
}

@main
struct TranscribeMeetingApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    init() { NSApplication.shared.setActivationPolicy(.regular) }
    var body: some Scene {
        WindowGroup { ContentView() }
            .windowResizability(.contentMinSize)
    }
}
