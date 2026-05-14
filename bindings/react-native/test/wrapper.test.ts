import { beforeEach, describe, expect, it, vi } from "vitest";
import { __setXybridMock, createEventEmitter } from "./react-native-mock.js";
import type { LoadErrorEvent, LoadProgressEvent } from "../src/specs/NativeXybrid.js";

const progress = createEventEmitter<LoadProgressEvent>();
const errors = createEventEmitter<LoadErrorEvent>();

beforeEach(() => {
  progress.emit = progress.emit.bind(progress);
  errors.emit = errors.emit.bind(errors);
});

describe("@xybrid/react-native wrapper", () => {
  it("maps model shorthand, text envelopes, and native results", async () => {
    const native = {
      initialize: vi.fn().mockResolvedValue(undefined),
      setApiKey: vi.fn(),
      loadModel: vi.fn().mockResolvedValue({ handle: 7 }),
      runModel: vi.fn().mockResolvedValue({
        success: true,
        text: "ok",
        latencyMs: 12,
      }),
      voices: vi.fn().mockResolvedValue([]),
      defaultVoiceId: vi.fn().mockResolvedValue(null),
      disposeModel: vi.fn(),
      isModelCached: vi.fn().mockResolvedValue(false),
      xybridLoadProgress: progress,
      xybridLoadError: errors,
    };
    __setXybridMock(native);

    const { Envelope, Xybrid } = await import("../src/index.js");
    await Xybrid.init();
    const model = await Xybrid.model("kokoro-82m").load();
    const result = await model.run(Envelope.text("Hello", { voiceId: "af_heart", speed: 1 }));

    expect(native.initialize).toHaveBeenCalledWith(undefined);
    expect(native.loadModel).toHaveBeenCalledWith({ kind: "registry", value: "kokoro-82m" }, expect.any(String));
    expect(native.runModel).toHaveBeenCalledWith(
      7,
      { kind: "text", text: "Hello", voiceId: "af_heart", speed: 1 },
      undefined,
    );
    expect(result).toEqual({ success: true, text: "ok", audioBytes: undefined, embedding: undefined, latencyMs: 12 });
  });

  it("filters progress events by request id", async () => {
    let capturedRequestId = "";
    const seen: LoadProgressEvent[] = [];
    const native = {
      initialize: vi.fn().mockResolvedValue(undefined),
      setApiKey: vi.fn(),
      loadModel: vi.fn((_source, requestId: string) => {
        capturedRequestId = requestId;
        progress.emit({ requestId: "other", progress: 0.9, phase: "ready" });
        progress.emit({ requestId, modelId: "kokoro-82m", progress: 0.5, phase: "loading" });
        return Promise.resolve({ handle: 3 });
      }),
      runModel: vi.fn(),
      voices: vi.fn(),
      defaultVoiceId: vi.fn(),
      disposeModel: vi.fn(),
      isModelCached: vi.fn(),
      xybridLoadProgress: progress,
      xybridLoadError: errors,
    };
    __setXybridMock(native);

    const { Xybrid } = await import("../src/index.js");
    await Xybrid.model("kokoro-82m").load({ onProgress: (event) => seen.push(event) });

    expect(capturedRequestId).toMatch(/^rn-/);
    expect(seen).toEqual([{ requestId: capturedRequestId, modelId: "kokoro-82m", progress: 0.5, phase: "loading" }]);
  });

  it("converts audio payloads through base64", async () => {
    const native = {
      initialize: vi.fn().mockResolvedValue(undefined),
      setApiKey: vi.fn(),
      loadModel: vi.fn().mockResolvedValue({ handle: 4 }),
      runModel: vi.fn().mockResolvedValue({
        success: true,
        audioBase64: "AQIDBA==",
        latencyMs: 5,
      }),
      voices: vi.fn(),
      defaultVoiceId: vi.fn(),
      disposeModel: vi.fn(),
      isModelCached: vi.fn(),
      xybridLoadProgress: progress,
      xybridLoadError: errors,
    };
    __setXybridMock(native);

    const { Envelope, Xybrid } = await import("../src/index.js");
    const model = await Xybrid.model({ kind: "bundle", value: "/model.xyb" }).load();
    const result = await model.run(Envelope.audio(new Uint8Array([1, 2, 3, 4])));

    expect(native.runModel.mock.calls[0]![1]).toMatchObject({ kind: "audio", audioBase64: "AQIDBA==" });
    expect(Array.from(result.audioBytes ?? [])).toEqual([1, 2, 3, 4]);
  });
});
