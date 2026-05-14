import { XybridEnvelope } from "./envelope.js";
import { XybridError } from "./errors.js";
import { XybridTensor } from "./tensor.js";

export type PreprocessedData =
  | { readonly kind: "audio-bytes"; readonly bytes: Uint8Array }
  | { readonly kind: "audio-samples"; readonly samples: Float32Array; readonly sampleRate: number }
  | { readonly kind: "text"; readonly text: string }
  | { readonly kind: "tensor"; readonly tensor: XybridTensor }
  | {
      readonly kind: "token-ids";
      readonly ids: readonly number[];
      readonly attentionMask: readonly number[];
      readonly tokenTypeIds: readonly number[];
      readonly vocabFile: string;
      readonly originalText: string;
    }
  | { readonly kind: "phoneme-ids"; readonly ids: readonly bigint[]; readonly phonemes: string; readonly originalText: string };

export type RawOutputs =
  | { readonly kind: "tensor-map"; readonly outputs: Record<string, XybridTensor> }
  | { readonly kind: "token-ids"; readonly ids: readonly number[] }
  | { readonly kind: "text"; readonly text: string }
  | { readonly kind: "class-id"; readonly classId: number }
  | { readonly kind: "audio-bytes"; readonly bytes: Uint8Array; readonly sampleRate?: number; readonly samples?: Float32Array };

export function preprocessedFromEnvelope(envelope: XybridEnvelope): PreprocessedData {
  switch (envelope.kind) {
    case "Text":
      return { kind: "text", text: envelope.text() ?? "" };
    case "Audio":
      return { kind: "audio-bytes", bytes: envelope.audio() ?? new Uint8Array() };
    case "Embedding": {
      const values = envelope.embedding() ?? new Float32Array();
      return { kind: "tensor", tensor: XybridTensor.fromFloat32(values, { shape: [values.length] }) };
    }
    default:
      throw new XybridError("runtime_error", "Unsupported envelope kind");
  }
}
