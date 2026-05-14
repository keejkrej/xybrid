import * as ort from "onnxruntime-web";
import { ArtifactResolver } from "./artifact-resolver.js";
import { concatFloat32, trimSamples, wavFromFloat32 } from "./audio.js";
import { XybridError } from "./errors.js";
import type { PreprocessedData, RawOutputs } from "./intermediate.js";
import { runPreprocessing } from "./preprocessing.js";
import { XybridTensor } from "./tensor.js";
import type { TensorMetadata, XybridRunOptions, XybridWebManifest } from "./types.js";
import { loadVoiceEmbedding } from "./voice.js";

export async function runTts(
  session: ort.InferenceSession,
  manifest: XybridWebManifest,
  artifacts: ArtifactResolver,
  preprocessed: PreprocessedData,
  options: XybridRunOptions = {},
): Promise<RawOutputs> {
  if (preprocessed.kind !== "phoneme-ids") {
    throw new XybridError("runtime_error", "TTS execution requires phoneme IDs");
  }
  const maxChars = manifest.max_chunk_chars ?? 350;
  const chunks = chunkText(preprocessed.originalText, maxChars);
  if (chunks.length <= 1) {
    return runOneTts(session, manifest, artifacts, preprocessed.ids, options);
  }

  const audioChunks: Float32Array[] = [];
  for (const chunk of chunks) {
    const chunkData = await runPreprocessing(manifest, { kind: "text", text: chunk }, artifacts);
    if (chunkData.kind !== "phoneme-ids") {
      throw new XybridError("runtime_error", "TTS chunk preprocessing did not produce phoneme IDs");
    }
    const raw = await runOneTts(session, manifest, artifacts, chunkData.ids, options);
    const samples = raw.kind === "tensor-map" ? firstFloat32(raw.outputs) : raw.kind === "audio-bytes" ? raw.samples : undefined;
    if (samples !== undefined) {
      audioChunks.push(trimSamples(samples, manifest.trim_trailing_samples ?? 0));
    }
  }
  const samples = concatFloat32(audioChunks);
  return { kind: "tensor-map", outputs: { waveform: XybridTensor.fromFloat32(samples, { shape: [samples.length] }) } };
}

async function runOneTts(
  session: ort.InferenceSession,
  manifest: XybridWebManifest,
  artifacts: ArtifactResolver,
  ids: readonly bigint[],
  options: XybridRunOptions,
): Promise<RawOutputs> {
  const tokenCount = ids.length >= 2 && ids[0] === 0n && ids[ids.length - 1] === 0n ? ids.length - 2 : ids.length;
  const embedding = loadVoiceEmbedding(manifest.voices, artifacts, {
    ...(options.voice === undefined ? {} : { voice: options.voice }),
    tokenCount,
  });
  const speed = options.speed ?? 1;
  const feeds: Record<string, ort.Tensor> = {};
  for (const input of manifest.inputs) {
    const kind = classifyTtsInput(input);
    if (kind === "tokens") {
      feeds[input.name] = new ort.Tensor("int64", BigInt64Array.from(ids), [1, ids.length]);
    } else if (kind === "voice") {
      feeds[input.name] = new ort.Tensor("float32", embedding, [1, embedding.length]);
    } else if (kind === "speed") {
      feeds[input.name] = new ort.Tensor("float32", new Float32Array([speed]), [1]);
    }
  }
  if (Object.keys(feeds).length !== manifest.inputs.length) {
    const found = manifest.inputs.map((input) => `${input.name} ${input.dtype} [${input.shape.join(", ")}]`).join("; ");
    throw new XybridError(
      "runtime_error",
      `TTS model inputs must be int64 [1,N], float32 [1,256], and float32 [1]. Found: ${found}`,
    );
  }
  const results = await session.run(feeds);
  return { kind: "tensor-map", outputs: Object.fromEntries(Object.entries(results).map(([name, tensor]) => [name, fromOrtTensor(tensor)])) };
}

export function encodeTtsAudio(raw: RawOutputs, sampleRate: number): RawOutputs {
  const samples = raw.kind === "tensor-map" ? firstFloat32(raw.outputs) : raw.kind === "audio-bytes" ? raw.samples : undefined;
  if (samples === undefined) {
    throw new XybridError("runtime_error", "TTSAudioEncode requires waveform output");
  }
  return { kind: "audio-bytes", bytes: wavFromFloat32(samples, sampleRate), sampleRate, samples };
}

function classifyTtsInput(input: TensorMetadata): "tokens" | "voice" | "speed" | "unknown" {
  if (input.dtype === "int64" && input.shape.length === 2 && concreteOrDynamic(input.shape[0], 1)) {
    return "tokens";
  }
  if (input.dtype === "float32" && input.shape.length === 2 && concreteOrDynamic(input.shape[0], 1)) {
    return "voice";
  }
  if (input.dtype === "float32" && input.shape.length === 1) {
    return "speed";
  }
  return "unknown";
}

function concreteOrDynamic(value: number | string | null | undefined, expected: number): boolean {
  return value === expected || typeof value === "string" || value === null || value === undefined;
}

function firstFloat32(outputs: Record<string, XybridTensor>): Float32Array | undefined {
  const waveform = outputs.waveform ?? Object.values(outputs)[0];
  if (waveform?.data instanceof Float32Array) {
    return waveform.data;
  }
  return undefined;
}

function fromOrtTensor(tensor: ort.Tensor): XybridTensor {
  if (tensor.data instanceof Float32Array) {
    return XybridTensor.fromFloat32(tensor.data, { shape: [...tensor.dims] });
  }
  if (tensor.data instanceof BigInt64Array) {
    return XybridTensor.fromInt64(tensor.data, { shape: [...tensor.dims] });
  }
  if (tensor.data instanceof Int32Array) {
    return XybridTensor.fromInt32(tensor.data, { shape: [...tensor.dims] });
  }
  if (tensor.data instanceof Uint8Array) {
    return XybridTensor.fromUint8(tensor.data, { shape: [...tensor.dims] });
  }
  throw new XybridError("tensor_invalid", `Unsupported ORT tensor type ${tensor.type}`);
}

function chunkText(text: string, maxChars: number): string[] {
  if (text.length <= maxChars) {
    return [text];
  }
  const sentences = text.match(/[^.!?\u2026]+[.!?\u2026]*/g) ?? [text];
  const chunks: string[] = [];
  let current = "";
  for (const sentence of sentences) {
    const candidate = current ? `${current} ${sentence.trim()}` : sentence.trim();
    if (candidate.length > maxChars && current) {
      chunks.push(current);
      current = sentence.trim();
    } else {
      current = candidate;
    }
  }
  if (current) {
    chunks.push(current);
  }
  return chunks;
}
