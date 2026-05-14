import { XybridError } from "./errors.js";
import { XybridTensor } from "./tensor.js";
import type { TensorDType } from "./types.js";

export type XybridOutputType = "text" | "audio" | "tensor" | "class" | "tokens";

export type XybridRunResultValue =
  | { readonly type: "text"; readonly text: string }
  | { readonly type: "audio"; readonly audio: Uint8Array; readonly sampleRate?: number; readonly samples?: Float32Array }
  | { readonly type: "tensor"; readonly outputs: Record<string, XybridTensor> }
  | { readonly type: "class"; readonly classId: number }
  | { readonly type: "tokens"; readonly tokenIds: readonly number[] };

export class XybridRunResult {
  readonly outputType: XybridOutputType;

  constructor(readonly value: XybridRunResultValue) {
    this.outputType = value.type;
  }

  text(): string | undefined {
    if (this.value.type === "text") {
      return this.value.text;
    }
    if (this.value.type === "class") {
      return `Class: ${this.value.classId}`;
    }
    return undefined;
  }

  audio(): Uint8Array | undefined {
    return this.value.type === "audio" ? this.value.audio : undefined;
  }

  tensor(): Record<string, XybridTensor> | undefined {
    return this.value.type === "tensor" ? this.value.outputs : undefined;
  }

  unwrapText(): string {
    const text = this.text();
    if (text === undefined) {
      throw new XybridError("runtime_error", "XybridRunResult is not text");
    }
    return text;
  }

  unwrapAudio(): Uint8Array {
    const audio = this.audio();
    if (audio === undefined) {
      throw new XybridError("runtime_error", "XybridRunResult is not audio");
    }
    return audio;
  }

  async audioBuffer(): Promise<AudioBuffer> {
    if (this.value.type !== "audio") {
      throw new XybridError("runtime_error", "XybridRunResult is not audio");
    }
    if (typeof AudioContext === "undefined") {
      throw new XybridError("runtime_error", "AudioContext is not available");
    }
    const context = new AudioContext();
    const copy = this.value.audio.buffer.slice(
      this.value.audio.byteOffset,
      this.value.audio.byteOffset + this.value.audio.byteLength,
    ) as ArrayBuffer;
    return context.decodeAudioData(copy);
  }
}

export function tensorFromFloat32(data: Float32Array, shape: readonly number[], dtype: TensorDType = "float32"): XybridTensor {
  return XybridTensor.fromData(data, dtype, { shape });
}
