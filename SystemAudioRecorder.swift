// SystemAudioRecorder (shared with Session Notes) — records what the Mac is playing (the other side of a
// Zoom / Teams / browser video call) using a Core Audio process tap
// (macOS 14.2+). Works with any video app; needs the one-time "System Audio
// Recording" permission (NSAudioCaptureUsageDescription), not Screen Recording.
import AVFoundation
import CoreAudio

final class SystemAudioRecorder {
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var file: AVAudioFile?
    private let queue = DispatchQueue(label: "SessionNotes.systemAudio")
    private(set) var framesWritten: AVAudioFramePosition = 0

    enum TapError: LocalizedError {
        case osStatus(String, OSStatus)
        var errorDescription: String? {
            switch self {
            case .osStatus(let what, let st): return "\(what) failed (\(st))."
            }
        }
    }

    private func check(_ st: OSStatus, _ what: String) throws {
        if st != noErr { throw TapError.osStatus(what, st) }
    }

    private func defaultOutputUID() throws -> String {
        var dev = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultSystemOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        try check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &dev),
                  "Finding the speaker device")
        var uid: CFString = "" as CFString
        size = UInt32(MemoryLayout<CFString>.size)
        addr.mSelector = kAudioDevicePropertyDeviceUID
        try withUnsafeMutablePointer(to: &uid) { p in
            try check(AudioObjectGetPropertyData(dev, &addr, 0, nil, &size, p), "Reading the speaker ID")
        }
        return uid as String
    }

    func start(to url: URL) throws {
        // 1. A private tap on everything the Mac plays, left audible.
        let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        desc.uuid = UUID()
        desc.muteBehavior = .unmuted
        desc.isPrivate = true
        desc.name = "Session Notes"
        try check(AudioHardwareCreateProcessTap(desc, &tapID), "Creating the system-audio tap")

        // 2. A private aggregate device that exposes the tap as an input.
        let outUID = try defaultOutputUID()
        let agg: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Session Notes Tap",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outUID]],
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                               kAudioSubTapUIDKey: desc.uuid.uuidString]],
        ]
        try check(AudioHardwareCreateAggregateDevice(agg as CFDictionary, &aggregateID),
                  "Creating the recording device")

        // 3. The tap's audio format.
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var addr = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
                                              mScope: kAudioObjectPropertyScopeGlobal,
                                              mElement: kAudioObjectPropertyElementMain)
        try check(AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &asbd), "Reading the audio format")
        guard let format = AVAudioFormat(streamDescription: &asbd) else {
            throw TapError.osStatus("Understanding the audio format", -1)
        }

        // 4. Compressed file (AAC), mono is plenty for speech.
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVEncoderBitRateKey: 64000,
        ]
        file = try AVAudioFile(forWriting: url, settings: settings,
                               commonFormat: format.commonFormat, interleaved: format.isInterleaved)

        // 5. Pull audio and write it.
        try check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, queue) {
            [weak self] _, inInput, _, _, _ in
            guard let self, let file = self.file,
                  let buf = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: inInput,
                                             deallocator: nil) else { return }
            do { try file.write(from: buf); self.framesWritten += AVAudioFramePosition(buf.frameLength) }
            catch { }
        }, "Starting the system-audio stream")
        try check(AudioDeviceStart(aggregateID, procID), "Starting system-audio recording")
    }

    func stop() {
        if aggregateID != kAudioObjectUnknown {
            if let procID {
                AudioDeviceStop(aggregateID, procID)
                AudioDeviceDestroyIOProcID(aggregateID, procID)
            }
            AudioHardwareDestroyAggregateDevice(aggregateID)
        }
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
        queue.sync { file = nil }            // closes and finalizes the .m4a
        procID = nil; aggregateID = kAudioObjectUnknown; tapID = kAudioObjectUnknown
    }

    deinit { stop() }
}
