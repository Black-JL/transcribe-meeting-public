// Transcribe Meeting — a native macOS app.
// Record a meeting from the mic (AVFoundation → proper TCC prompt) or pick a
// file, then: transcribe + diarize locally, and summarize into meeting notes
// with a local LLM (Ollama). The meeting NOTES are the headline deliverable.
import SwiftUI
import AVFoundation
import AppKit

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
    @Published var isRecording = false
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
    @Published var selectedModel: String = UserDefaults.standard.string(forKey: "notesModel") ?? "gpt-oss:20b" {
        didSet { UserDefaults.standard.set(selectedModel, forKey: "notesModel") }
    }

    private var recorder: AVAudioRecorder?
    private var currentURL: URL?
    private var proc: Process?
    private var timer: Timer?
    private var lastActivity = Date()

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
        let url = meetingsDir.appendingPathComponent("\(stem).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
        ]
        do {
            let r = try AVAudioRecorder(url: url, settings: settings)
            guard r.record() else { status = "Could not start recording."; return }
            recorder = r; currentURL = url
            isRecording = true
            status = "● Recording…  click Stop when the meeting ends."
            startTimer()
        } catch {
            status = "Could not start recording: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        recorder?.stop(); recorder = nil; isRecording = false
        stopTimer()
        guard let url = currentURL else { return }
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int ?? 0
        if size < 4000 {
            status = "The recording was silent. Check System Settings ▸ Sound ▸ Input (use your real mic, not muted/virtual), then try again."
            return
        }
        runPipeline(url, title: meetingName.isEmpty ? url.deletingPathExtension().lastPathComponent : meetingName)
    }

    // MARK: File import

    func pickAndTranscribe() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Choose an audio or video file to transcribe"
        if panel.runModal() == .OK, let url = panel.url {
            runPipeline(url, title: url.deletingPathExtension().lastPathComponent)
        }
    }

    // MARK: Pipeline (transcribe → notes). Robust: drains stderr, watchdog.

    private func runPipeline(_ url: URL, title: String) {
        busy = true; elapsed = 0; stage = "Starting…"; actionMsg = ""
        status = "Processing “\(title)” …"
        startTimer()
        lastActivity = Date()

        DispatchQueue.global().async {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: self.tool)
            p.arguments = [url.path, "--title", title, "--notes-model", self.selectedModel]
            let outPipe = Pipe(); let errPipe = Pipe()
            p.standardOutput = outPipe; p.standardError = errPipe

            // Drain stderr live (large: parakeet/ffmpeg/notes progress) so the
            // child never blocks on a full pipe buffer. Map lines to stages.
            errPipe.fileHandleForReading.readabilityHandler = { fh in
                let d = fh.availableData
                if d.isEmpty { return }
                self.lastActivity = Date()
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
            }
        }
    }

    private func friendlyStage(_ line: String) -> String? {
        let l = line.lowercased()
        if l.contains("converting audio") { return "Step 1 of 4 — preparing audio…" }
        if l.contains("transcribing with parakeet") { return "Step 2 of 4 — transcribing on the GPU (the long part)…" }
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
            if self.busy, Date().timeIntervalSince(self.lastActivity) > 1200 {
                self.proc?.terminate()
                self.status = "Processing stalled and was stopped. See the log."
                self.busy = false; self.stopTimer()
            }
        }
    }
    private func stopTimer() { timer?.invalidate(); timer = nil }

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
            DispatchQueue.main.async {
                if names.isEmpty { names = [self.selectedModel] }
                if !names.contains(self.selectedModel) { names.insert(self.selectedModel, at: 0) }
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
        .onAppear { engine.loadHistory(); engine.loadDestinations(); engine.loadModels() }
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
            Button(action: { engine.toggleRecord() }) {
                Label(engine.isRecording ? "Stop Recording" : "Record New Meeting",
                      systemImage: engine.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(maxWidth: .infinity).padding(.vertical, 10)
            }
            .buttonStyle(.borderedProminent).tint(engine.isRecording ? .red : .accentColor)
            .disabled(engine.busy)
            Button(action: { engine.pickAndTranscribe() }) {
                Label("Transcribe a File…", systemImage: "doc.text.magnifyingglass")
                    .frame(maxWidth: .infinity).padding(.vertical, 5)
            }
            .buttonStyle(.bordered).disabled(engine.busy || engine.isRecording)

            HStack(spacing: 6) {
                Image(systemName: "brain").foregroundStyle(.secondary)
                Text("Notes AI").foregroundStyle(.secondary)
                Picker("", selection: $engine.selectedModel) {
                    ForEach(engine.notesModels, id: \.self) { Text($0).tag($0) }
                }.labelsHidden()
                Button { engine.loadModels() } label: { Image(systemName: "arrow.clockwise") }
                    .help("Refresh model list from Ollama")
            }
            .controlSize(.small).font(.caption)
            .disabled(engine.busy || engine.isRecording)
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

@main
struct TranscribeMeetingApp: App {
    init() { NSApplication.shared.setActivationPolicy(.regular) }
    var body: some Scene {
        WindowGroup { ContentView() }
            .windowResizability(.contentMinSize)
    }
}
