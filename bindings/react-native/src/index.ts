import NativeXybrid, {
  type GenerationConfig,
  type LoadErrorEvent,
  type LoadProgressEvent,
  type ModelSource,
  type NativeEnvelope,
  type VoiceInfo,
  type XybridResult as NativeXybridResult,
} from "./specs/NativeXybrid.js";
import { base64ToBytes, bytesToBase64 } from "./base64.js";
import { toXybridError, XybridError } from "./errors.js";

export type { GenerationConfig, LoadProgressEvent, ModelSource, VoiceInfo };
export { XybridError };

export type TextEnvelopeOptions = {
  voiceId?: string;
  speed?: number;
};

export type AudioEnvelopeOptions = {
  sampleRate?: number;
  channels?: number;
};

export class Envelope {
  private constructor(readonly native: NativeEnvelope) {}

  static text(text: string, options: TextEnvelopeOptions = {}): Envelope {
    return new Envelope({
      kind: "text",
      text,
      voiceId: options.voiceId,
      speed: options.speed,
    });
  }

  static audio(bytes: Uint8Array | ArrayBuffer, options: AudioEnvelopeOptions = {}): Envelope {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    return new Envelope({
      kind: "audio",
      audioBase64: bytesToBase64(data),
      sampleRate: options.sampleRate ?? 16000,
      channels: options.channels ?? 1,
    });
  }

  static embedding(data: Float32Array | readonly number[]): Envelope {
    const values = data instanceof Float32Array ? Array.from(data) : Array.from(data);
    return new Envelope({ kind: "embedding", embedding: values });
  }
}

export type XybridRunInput = Envelope | string | Uint8Array | ArrayBuffer | Float32Array | readonly number[];

export type XybridResult = {
  success: boolean;
  text?: string | undefined;
  audioBytes?: Uint8Array | undefined;
  embedding?: number[] | undefined;
  latencyMs: number;
};

export type LoadModelOptions = {
  onProgress?: (event: LoadProgressEvent) => void;
};

export class XybridModelRef {
  constructor(readonly source: ModelSource) {}

  async load(options: LoadModelOptions = {}): Promise<XybridModel> {
    const requestId = createRequestId();
    let progressSubscription: { remove(): void } | undefined;
    let errorSubscription: { remove(): void } | undefined;

    const loadError = new Promise<never>((_, reject) => {
      progressSubscription = NativeXybrid.xybridLoadProgress((event) => {
        if (event.requestId === requestId) {
          options.onProgress?.(event);
        }
      });
      errorSubscription = NativeXybrid.xybridLoadError((event: LoadErrorEvent) => {
        if (event.requestId === requestId) {
          reject(new XybridError("load_error", event.message));
        }
      });
    });
    const load = NativeXybrid.loadModel(this.source, requestId);

    try {
      const loaded = await Promise.race([load, loadError]);
      return new XybridModel(loaded.handle);
    } catch (error) {
      throw toXybridError(error);
    } finally {
      progressSubscription?.remove();
      errorSubscription?.remove();
    }
  }
}

export class XybridModel {
  constructor(readonly handle: number) {}

  async run(input: XybridRunInput, config?: GenerationConfig): Promise<XybridResult> {
    try {
      const native = await NativeXybrid.runModel(this.handle, envelopeFromInput(input).native, config);
      return resultFromNative(native);
    } catch (error) {
      throw toXybridError(error);
    }
  }

  async voices(): Promise<VoiceInfo[]> {
    try {
      return await NativeXybrid.voices(this.handle);
    } catch (error) {
      throw toXybridError(error);
    }
  }

  async defaultVoiceId(): Promise<string | null> {
    try {
      return await NativeXybrid.defaultVoiceId(this.handle);
    } catch (error) {
      throw toXybridError(error);
    }
  }

  dispose(): void {
    NativeXybrid.disposeModel(this.handle);
  }
}

export class Xybrid {
  static async init(options?: { cacheDir?: string }): Promise<void> {
    try {
      await NativeXybrid.initialize(options);
    } catch (error) {
      throw toXybridError(error);
    }
  }

  static setApiKey(apiKey: string): void {
    NativeXybrid.setApiKey(apiKey);
  }

  static model(source: string | ModelSource): XybridModelRef {
    return new XybridModelRef(typeof source === "string" ? { kind: "registry", value: source } : source);
  }

  static async isModelCached(modelId: string): Promise<boolean> {
    try {
      return await NativeXybrid.isModelCached(modelId);
    } catch (error) {
      throw toXybridError(error);
    }
  }
}

export function envelopeFromInput(input: XybridRunInput): Envelope {
  if (input instanceof Envelope) {
    return input;
  }
  if (typeof input === "string") {
    return Envelope.text(input);
  }
  if (input instanceof Uint8Array || input instanceof ArrayBuffer) {
    return Envelope.audio(input);
  }
  return Envelope.embedding(input);
}

function resultFromNative(native: NativeXybridResult): XybridResult {
  return {
    success: native.success,
    text: native.text,
    audioBytes: native.audioBase64 ? base64ToBytes(native.audioBase64) : undefined,
    embedding: native.embedding,
    latencyMs: native.latencyMs,
  };
}

function createRequestId(): string {
  const random = Math.random().toString(36).slice(2);
  return `rn-${Date.now().toString(36)}-${random}`;
}
