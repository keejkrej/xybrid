import { ArtifactResolver } from "./artifact-resolver.js";
import { postprocessTtsAudio, trimTrailingNearSilence } from "./audio.js";
import { XybridError } from "./errors.js";
import type { RawOutputs } from "./intermediate.js";
import { XybridRunResult } from "./result.js";
import { XybridTensor } from "./tensor.js";
import { encodeTtsAudio } from "./tts.js";
import type { PostprocessingStep, XybridWebManifest } from "./types.js";

export async function runPostprocessing(
  manifest: XybridWebManifest,
  input: RawOutputs,
  artifacts: ArtifactResolver,
): Promise<RawOutputs> {
  let data = input;
  for (const step of manifest.postprocessing) {
    data = await applyPostprocessingStep(step, data, artifacts);
  }
  return data;
}

export async function applyPostprocessingStep(
  step: PostprocessingStep,
  data: RawOutputs,
  artifacts: ArtifactResolver,
): Promise<RawOutputs> {
  switch (step.type) {
    case "Argmax":
      return argmaxStep(data);
    case "Softmax":
      return softmaxStep(data, step.dim ?? undefined);
    case "TopK":
      return topKStep(data, step.k);
    case "Threshold":
      return thresholdStep(data, step.threshold, step.return_indices ?? false);
    case "MeanPool":
      return meanPoolStep(data, step.dim ?? 1);
    case "CTCDecode":
      return ctcDecodeStep(data, artifacts.text(step.vocab_file), step.blank_index ?? 0);
    case "BPEDecode":
      return bpeDecodeStep(data, artifacts.text(step.vocab_file));
    case "WhisperDecode":
      return whisperDecodeStep(data, artifacts.text(step.tokenizer_file));
    case "TTSAudioEncode": {
      const encoded = encodeTtsAudio(data, step.sample_rate);
      if (encoded.kind !== "audio-bytes" || encoded.samples === undefined) {
        return encoded;
      }
      let samples = step.apply_postprocessing ?? true ? postprocessTtsAudio(encoded.samples) : encoded.samples;
      if (step.trim_trailing_silence ?? false) {
        samples = trimTrailingNearSilence(samples, step.sample_rate);
      }
      return encodeTtsAudio({ kind: "audio-bytes", bytes: encoded.bytes, sampleRate: step.sample_rate, samples }, step.sample_rate);
    }
    case "TemperatureSample":
    case "Denormalize":
      return data;
    case "CodecDecode":
      throw new XybridError("unsupported_step", "CodecDecode is not supported by the browser generic dispatcher yet");
    default:
      return assertNever(step);
  }
}

export function resultFromRawOutputs(data: RawOutputs): XybridRunResult {
  switch (data.kind) {
    case "text":
      return new XybridRunResult({ type: "text", text: data.text });
    case "class-id":
      return new XybridRunResult({ type: "class", classId: data.classId });
    case "audio-bytes":
      return new XybridRunResult({
        type: "audio",
        audio: data.bytes,
        ...(data.sampleRate === undefined ? {} : { sampleRate: data.sampleRate }),
        ...(data.samples === undefined ? {} : { samples: data.samples }),
      });
    case "token-ids":
      return new XybridRunResult({ type: "tokens", tokenIds: data.ids });
    case "tensor-map":
      return new XybridRunResult({ type: "tensor", outputs: data.outputs });
  }
}

function firstTensor(data: RawOutputs, op: string): XybridTensor {
  if (data.kind !== "tensor-map") {
    throw new XybridError("runtime_error", `${op} requires tensor map`);
  }
  const tensor = Object.values(data.outputs)[0];
  if (tensor === undefined) {
    throw new XybridError("runtime_error", `${op} requires at least one tensor`);
  }
  return tensor;
}

function argmaxStep(data: RawOutputs): RawOutputs {
  const values = floatValues(firstTensor(data, "Argmax"));
  let best = 0;
  for (let i = 1; i < values.length; i += 1) {
    if ((values[i] ?? Number.NEGATIVE_INFINITY) > (values[best] ?? Number.NEGATIVE_INFINITY)) {
      best = i;
    }
  }
  return { kind: "class-id", classId: best };
}

function softmaxStep(data: RawOutputs, dim: number | undefined): RawOutputs {
  if (data.kind !== "tensor-map") {
    throw new XybridError("runtime_error", "Softmax requires tensor map");
  }
  const outputs: Record<string, XybridTensor> = {};
  for (const [name, tensor] of Object.entries(data.outputs)) {
    const values = new Float32Array(floatValues(tensor));
    const shape = tensor.shape;
    const targetDim = dim ?? shape.length - 1;
    if (shape.length === 1) {
      softmax1d(values, 0, values.length);
    } else if (shape.length === 2 && targetDim === 1) {
      const classes = shape[1] ?? values.length;
      for (let batch = 0; batch < (shape[0] ?? 1); batch += 1) {
        softmax1d(values, batch * classes, classes);
      }
    } else {
      throw new XybridError("runtime_error", `Softmax only supports 1D or 2D tensors, got [${shape.join(", ")}]`);
    }
    outputs[name] = XybridTensor.fromFloat32(values, { shape: tensor.shape, layout: tensor.layout });
  }
  return { kind: "tensor-map", outputs };
}

function topKStep(data: RawOutputs, k: number): RawOutputs {
  const tensor = firstTensor(data, "TopK");
  const values = floatValues(tensor);
  const pairs = [...values].map((score, index) => ({ index, score })).sort((a, b) => b.score - a.score).slice(0, k);
  const flat = new Float32Array(pairs.length * 2);
  for (let i = 0; i < pairs.length; i += 1) {
    flat[i * 2] = pairs[i]?.index ?? 0;
    flat[i * 2 + 1] = pairs[i]?.score ?? 0;
  }
  return { kind: "tensor-map", outputs: { topk: XybridTensor.fromFloat32(flat, { shape: [flat.length] }) } };
}

function thresholdStep(data: RawOutputs, threshold: number, returnIndices: boolean): RawOutputs {
  const tensor = firstTensor(data, "Threshold");
  const values = floatValues(tensor);
  if (returnIndices) {
    const indices = [...values].flatMap((value, index) => (value > threshold ? [index] : []));
    return { kind: "tensor-map", outputs: { threshold_indices: XybridTensor.fromFloat32(indices, { shape: [indices.length] }) } };
  }
  const mask = Float32Array.from(values, (value) => (value > threshold ? 1 : 0));
  return { kind: "tensor-map", outputs: { threshold_mask: XybridTensor.fromFloat32(mask, { shape: tensor.shape }) } };
}

function meanPoolStep(data: RawOutputs, dim: number): RawOutputs {
  const tensor = firstTensor(data, "MeanPool");
  const values = floatValues(tensor);
  const [batch, seqLen, hidden] = tensor.shape;
  if (tensor.shape.length !== 3 || batch === undefined || seqLen === undefined || hidden === undefined || dim !== 1) {
    throw new XybridError("runtime_error", "MeanPool expects [batch, seq_len, hidden] and dim=1");
  }
  const pooled = new Float32Array(batch * hidden);
  for (let b = 0; b < batch; b += 1) {
    for (let h = 0; h < hidden; h += 1) {
      let sum = 0;
      for (let s = 0; s < seqLen; s += 1) {
        sum += values[(b * seqLen + s) * hidden + h] ?? 0;
      }
      pooled[b * hidden + h] = sum / seqLen;
    }
  }
  return { kind: "tensor-map", outputs: { sentence_embedding: XybridTensor.fromFloat32(pooled, { shape: [batch, hidden] }) } };
}

function ctcDecodeStep(data: RawOutputs, vocabContent: string, blankIndex: number): RawOutputs {
  const tensor = firstTensor(data, "CTCDecode");
  const values = floatValues(tensor);
  const [batch, time, vocabSize] = tensor.shape;
  if (tensor.shape.length !== 3 || batch === undefined || time === undefined || vocabSize === undefined) {
    throw new XybridError("runtime_error", "CTCDecode expects [batch, time, vocab]");
  }
  const tokenIds: number[] = [];
  let prev: number | undefined;
  for (let t = 0; t < time; t += 1) {
    let best = 0;
    for (let v = 1; v < vocabSize; v += 1) {
      if ((values[t * vocabSize + v] ?? Number.NEGATIVE_INFINITY) > (values[t * vocabSize + best] ?? Number.NEGATIVE_INFINITY)) {
        best = v;
      }
    }
    if (best !== blankIndex && best !== prev) {
      tokenIds.push(best);
    }
    prev = best;
  }
  return { kind: "text", text: decodeVocab(tokenIds, vocabContent) };
}

function bpeDecodeStep(data: RawOutputs, vocabContent: string): RawOutputs {
  if (data.kind !== "token-ids") {
    throw new XybridError("runtime_error", "BPEDecode requires token IDs");
  }
  return { kind: "text", text: decodeVocab(data.ids, vocabContent) };
}

function whisperDecodeStep(data: RawOutputs, _tokenizerJson: string): RawOutputs {
  if (data.kind !== "token-ids") {
    throw new XybridError("runtime_error", "WhisperDecode requires token IDs");
  }
  return { kind: "text", text: data.ids.filter((id) => id < 50257).join(" ") };
}

function decodeVocab(ids: readonly number[], content: string): string {
  const trimmed = content.trim();
  const vocab = trimmed.startsWith("{") ? reverseJsonVocab(trimmed) : content.split(/\r?\n/).map((line) => line.trim());
  return ids
    .map((id) => vocab[id] ?? "")
    .filter((token) => token && !/^<.*>$/.test(token))
    .map((token) => (token === "|" ? " " : token))
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

function reverseJsonVocab(content: string): string[] {
  const parsed = JSON.parse(content) as Record<string, number>;
  const out: string[] = [];
  for (const [token, id] of Object.entries(parsed)) {
    out[id] = token;
  }
  return out;
}

function floatValues(tensor: XybridTensor): Float32Array {
  if (!(tensor.data instanceof Float32Array)) {
    throw new XybridError("tensor_invalid", "Postprocessing tensor operation requires float32 tensor");
  }
  return tensor.data;
}

function softmax1d(values: Float32Array, start: number, length: number): void {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = start; i < start + length; i += 1) {
    max = Math.max(max, values[i] ?? Number.NEGATIVE_INFINITY);
  }
  let sum = 0;
  for (let i = start; i < start + length; i += 1) {
    const next = Math.exp((values[i] ?? 0) - max);
    values[i] = next;
    sum += next;
  }
  for (let i = start; i < start + length; i += 1) {
    values[i] = (values[i] ?? 0) / sum;
  }
}

function assertNever(value: never): never {
  throw new XybridError("unsupported_step", `Unsupported postprocessing step ${(value as { type?: string }).type ?? "unknown"}`);
}
