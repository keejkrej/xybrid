import Foundation
import React
#if os(iOS)
import UIKit
#endif

@objc(XybridReactNativeAdapter)
final class XybridReactNativeAdapter: NSObject {
    @objc var progressEmitter: ((NSDictionary) -> Void)?
    @objc var errorEmitter: ((NSDictionary) -> Void)?

    private let lock = NSLock()
    private var initialized = false
    private var nextHandle: Double = 1
    private var models: [Double: XybridModel] = [:]

    @objc(initializeWithOptions:resolve:reject:)
    func initialize(options: NSDictionary?, resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock) {
        lock.lock()
        defer { lock.unlock() }

        if !initialized {
            setBinding(binding: "react-native")
            initSdkCacheDir(cacheDir: cacheDir(from: options))
            registerPlatformObservers()
            initialized = true
        }
        resolve(nil)
    }

    @objc(setApiKey:)
    func setApiKey(_ apiKey: String) {
        setenv("XYBRID_API_KEY", apiKey, 1)
    }

    @objc(loadModelWithSource:requestId:resolve:reject:)
    func loadModel(source: NSDictionary, requestId: String, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
        Task {
            let modelId = source["value"] as? String
            emitProgress(requestId: requestId, modelId: modelId, progress: 0.0, phase: "starting")
            do {
                let loader = try loader(from: source)
                emitProgress(requestId: requestId, modelId: modelId, progress: 0.1, phase: "loading")
                let model = try await loader.load()
                let handle = store(model)
                emitProgress(requestId: requestId, modelId: modelId, progress: 1.0, phase: "ready")
                resolve(["handle": handle])
            } catch {
                let message = String(describing: error)
                errorEmitter?(["requestId": requestId, "message": message])
                reject("xybrid_load_error", message, error)
            }
        }
    }

    @objc(runModelWithHandle:envelope:config:resolve:reject:)
    func runModel(handle: Double, envelope: NSDictionary, config: NSDictionary?, resolve: @escaping RCTPromiseResolveBlock, reject: @escaping RCTPromiseRejectBlock) {
        Task {
            do {
                guard let model = model(for: handle) else {
                    throw AdapterError.unknownHandle(handle)
                }
                let result = try await model.run(envelope: try envelopeFromDictionary(envelope), config: generationConfig(from: config))
                var payload: [String: Any] = [
                    "success": result.success,
                    "latencyMs": result.latencyMs,
                ]
                if let text = result.text {
                    payload["text"] = text
                }
                if let audio = result.audioBytes {
                    payload["audioBase64"] = audio.base64EncodedString()
                }
                if let embedding = result.embedding {
                    payload["embedding"] = embedding
                }
                resolve(payload)
            } catch {
                reject("xybrid_run_error", String(describing: error), error)
            }
        }
    }

    @objc(voicesWithHandle:resolve:reject:)
    func voices(handle: Double, resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock) {
        guard let model = model(for: handle) else {
            reject("xybrid_unknown_handle", "Unknown or disposed model handle: \(handle)", AdapterError.unknownHandle(handle))
            return
        }
        let voices = model.voices()?.map { voice in
            var payload: [String: Any] = [
                "id": voice.id,
                "name": voice.name,
            ]
            payload["gender"] = voice.gender
            payload["language"] = voice.language
            payload["style"] = voice.style
            return payload
        } ?? []
        resolve(voices)
    }

    @objc(defaultVoiceIdWithHandle:resolve:reject:)
    func defaultVoiceId(handle: Double, resolve: RCTPromiseResolveBlock, reject: RCTPromiseRejectBlock) {
        guard let model = model(for: handle) else {
            reject("xybrid_unknown_handle", "Unknown or disposed model handle: \(handle)", AdapterError.unknownHandle(handle))
            return
        }
        resolve(model.defaultVoiceId() ?? NSNull())
    }

    @objc(disposeModelWithHandle:)
    func disposeModel(handle: Double) {
        lock.lock()
        models.removeValue(forKey: handle)
        lock.unlock()
    }

    @objc(isModelCachedWithModelId:resolve:reject:)
    func isModelCached(modelId: String, resolve: RCTPromiseResolveBlock, reject _: RCTPromiseRejectBlock) {
        let path = defaultCacheDirectory().appendingPathComponent(modelId)
        resolve(FileManager.default.fileExists(atPath: path.path))
    }

    private func store(_ model: XybridModel) -> Double {
        lock.lock()
        defer { lock.unlock() }
        let handle = nextHandle
        nextHandle += 1
        models[handle] = model
        return handle
    }

    private func model(for handle: Double) -> XybridModel? {
        lock.lock()
        defer { lock.unlock() }
        return models[handle]
    }

    private func cacheDir(from options: NSDictionary?) -> String {
        if let cacheDir = options?["cacheDir"] as? String {
            return cacheDir
        }
        return defaultCacheDirectory().path
    }

    private func defaultCacheDirectory() -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("xybrid/models", isDirectory: true)
    }

    private func loader(from source: NSDictionary) throws -> XybridModelLoader {
        guard let value = source["value"] as? String else {
            throw AdapterError.invalidSource
        }
        switch source["kind"] as? String {
        case "registry", nil:
            return XybridModelLoader.fromRegistry(modelId: value)
        case "bundle":
            return XybridModelLoader.fromBundle(path: value)
        case "directory":
            return try XybridModelLoader.fromDirectory(path: value)
        case "huggingface":
            return XybridModelLoader.fromHuggingface(repo: value)
        default:
            throw AdapterError.invalidSource
        }
    }

    private func envelopeFromDictionary(_ envelope: NSDictionary) throws -> XybridEnvelope {
        switch envelope["kind"] as? String {
        case "text":
            return .text(
                text: envelope["text"] as? String ?? "",
                voiceId: envelope["voiceId"] as? String,
                speed: envelope["speed"] as? Double
            )
        case "audio":
            let base64 = envelope["audioBase64"] as? String ?? ""
            guard let data = Data(base64Encoded: base64) else {
                throw AdapterError.invalidEnvelope
            }
            return .audio(
                bytes: data,
                sampleRate: UInt32((envelope["sampleRate"] as? NSNumber)?.uint32Value ?? 16000),
                channels: UInt32((envelope["channels"] as? NSNumber)?.uint32Value ?? 1)
            )
        case "embedding":
            return .embedding(data: (envelope["embedding"] as? [NSNumber])?.map { $0.floatValue } ?? [])
        default:
            throw AdapterError.invalidEnvelope
        }
    }

    private func generationConfig(from config: NSDictionary?) -> XybridGenerationConfig? {
        guard let config else { return nil }
        return XybridGenerationConfig(
            maxTokens: (config["maxTokens"] as? NSNumber)?.uint32Value,
            temperature: (config["temperature"] as? NSNumber)?.floatValue,
            topP: (config["topP"] as? NSNumber)?.floatValue,
            minP: (config["minP"] as? NSNumber)?.floatValue,
            topK: (config["topK"] as? NSNumber)?.uint32Value,
            repetitionPenalty: (config["repetitionPenalty"] as? NSNumber)?.floatValue,
            stopSequences: config["stopSequences"] as? [String]
        )
    }

    private func emitProgress(requestId: String, modelId: String?, progress: Double, phase: String) {
        var event: [String: Any] = [
            "requestId": requestId,
            "progress": progress,
            "phase": phase,
        ]
        event["modelId"] = modelId
        progressEmitter?(event as NSDictionary)
    }

    private func registerPlatformObservers() {
        #if os(iOS)
        let device = UIDevice.current
        device.isBatteryMonitoringEnabled = true
        pushBatteryLevel(device.batteryLevel)
        NotificationCenter.default.addObserver(
            forName: UIDevice.batteryLevelDidChangeNotification,
            object: nil,
            queue: nil
        ) { _ in
            self.pushBatteryLevel(UIDevice.current.batteryLevel)
        }
        #endif
    }

    #if os(iOS)
    private func pushBatteryLevel(_ level: Float) {
        guard level.isFinite, level >= 0 else {
            clearBatteryLevel()
            return
        }
        let pct = max(0, min(100, Int((level * 100).rounded())))
        setBatteryLevel(percent: UInt8(pct))
    }
    #endif
}

private enum AdapterError: Error {
    case invalidSource
    case invalidEnvelope
    case unknownHandle(Double)
}
