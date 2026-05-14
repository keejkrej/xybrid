import { ArtifactResolver } from "./artifact-resolver.js";
import { decodeWav } from "./audio.js";
import { XybridError } from "./errors.js";
import type { PreprocessedData } from "./intermediate.js";
import { phonemizeText } from "./phonemize.js";
import { XybridTensor } from "./tensor.js";
import type { PreprocessingStep, XybridWebManifest } from "./types.js";

export async function runPreprocessing(
  manifest: XybridWebManifest,
  input: PreprocessedData,
  artifacts: ArtifactResolver,
): Promise<PreprocessedData> {
  let data = input;
  for (const step of manifest.preprocessing) {
    data = applyPreprocessingStep(step, data, artifacts);
  }
  return data;
}

export function applyPreprocessingStep(
  step: PreprocessingStep,
  data: PreprocessedData,
  artifacts: ArtifactResolver,
): PreprocessedData {
  switch (step.type) {
    case "AudioDecode": {
      if (data.kind !== "audio-bytes") {
        throw new XybridError("runtime_error", "AudioDecode requires audio bytes");
      }
      const decoded = decodeWav(data.bytes, step.sample_rate, step.channels);
      return { kind: "audio-samples", samples: decoded.samples, sampleRate: decoded.sampleRate };
    }
    case "MelSpectrogram":
      return melSpectrogramStep(data, step);
    case "Normalize":
      return normalizeStep(data, step.mean, step.std);
    case "Reshape":
      return reshapeStep(data, step.shape);
    case "Resize":
      return resizeStep(data, step.width, step.height);
    case "CenterCrop":
      return centerCropStep(data, step.width, step.height);
    case "Phonemize": {
      if (data.kind !== "text") {
        throw new XybridError("runtime_error", "Phonemize requires text input");
      }
      const result = phonemizeText(
        data.text,
        {
          tokensFile: step.tokens_file,
          backend: step.backend,
          language: step.language,
          addPadding: step.add_padding,
          normalizeText: step.normalize_text,
          silenceTokens: step.silence_tokens,
        },
        artifacts,
      );
      return { kind: "phoneme-ids", ids: result.ids, phonemes: result.phonemes, originalText: data.text };
    }
    case "Tokenize":
      return tokenizeStep(data, artifacts.text(step.vocab_file), step.max_length ?? undefined);
    case "PhonemeRaw":
      throw new XybridError("unsupported_step", "PhonemeRaw is not supported by the browser generic dispatcher yet");
    default:
      return assertNever(step);
  }
}

function normalizeStep(data: PreprocessedData, mean: readonly number[], std: readonly number[]): PreprocessedData {
  if (data.kind !== "tensor") {
    throw new XybridError("runtime_error", "Normalize requires tensor input");
  }
  const tensor = data.tensor;
  if (!(tensor.data instanceof Float32Array)) {
    throw new XybridError("tensor_invalid", "Normalize requires float32 tensor input");
  }
  const values = new Float32Array(tensor.data.length);
  const channels = inferChannels(tensor.shape, mean.length);
  const spatial = tensor.data.length / Math.max(1, channels);
  for (let i = 0; i < tensor.data.length; i += 1) {
    const channel = Math.floor(i / spatial) % channels;
    values[i] = ((tensor.data[i] ?? 0) - (mean[channel] ?? mean[0] ?? 0)) / (std[channel] ?? std[0] ?? 1);
  }
  return { kind: "tensor", tensor: XybridTensor.fromFloat32(values, { shape: tensor.shape, layout: tensor.layout }) };
}

function reshapeStep(data: PreprocessedData, shape: readonly number[]): PreprocessedData {
  const tensor = toTensor(data);
  return { kind: "tensor", tensor: XybridTensor.fromData(tensor.data, tensor.dtype, { shape, layout: tensor.layout }) };
}

function resizeStep(data: PreprocessedData, width: number, height: number): PreprocessedData {
  const tensor = toFloatTensor(data, "Resize");
  const shape = tensor.shape;
  const { batch, channels, sourceHeight, sourceWidth } = imageShape(shape);
  const output = new Float32Array(batch * channels * height * width);
  for (let b = 0; b < batch; b += 1) {
    for (let c = 0; c < channels; c += 1) {
      for (let y = 0; y < height; y += 1) {
        const srcY = Math.min(sourceHeight - 1, Math.round((y * sourceHeight) / height));
        for (let x = 0; x < width; x += 1) {
          const srcX = Math.min(sourceWidth - 1, Math.round((x * sourceWidth) / width));
          output[((b * channels + c) * height + y) * width + x] =
            tensor.data[((b * channels + c) * sourceHeight + srcY) * sourceWidth + srcX] ?? 0;
        }
      }
    }
  }
  return { kind: "tensor", tensor: XybridTensor.fromFloat32(output, { shape: shape.length === 4 ? [batch, channels, height, width] : [channels, height, width], layout: tensor.layout }) };
}

function centerCropStep(data: PreprocessedData, width: number, height: number): PreprocessedData {
  const tensor = toFloatTensor(data, "CenterCrop");
  const shape = tensor.shape;
  const { batch, channels, sourceHeight, sourceWidth } = imageShape(shape);
  if (width > sourceWidth || height > sourceHeight) {
    throw new XybridError("runtime_error", `Cannot crop ${width}x${height} from ${sourceWidth}x${sourceHeight}`);
  }
  const startX = Math.floor((sourceWidth - width) / 2);
  const startY = Math.floor((sourceHeight - height) / 2);
  const output = new Float32Array(batch * channels * height * width);
  for (let b = 0; b < batch; b += 1) {
    for (let c = 0; c < channels; c += 1) {
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          output[((b * channels + c) * height + y) * width + x] =
            tensor.data[((b * channels + c) * sourceHeight + startY + y) * sourceWidth + startX + x] ?? 0;
        }
      }
    }
  }
  return { kind: "tensor", tensor: XybridTensor.fromFloat32(output, { shape: shape.length === 4 ? [batch, channels, height, width] : [channels, height, width], layout: tensor.layout }) };
}

function melSpectrogramStep(data: PreprocessedData, step: Extract<PreprocessingStep, { type: "MelSpectrogram" }>): PreprocessedData {
  if (data.kind !== "audio-samples") {
    throw new XybridError("runtime_error", "MelSpectrogram requires decoded audio samples");
  }
  const nMels = step.preset?.startsWith("whisper") ? 80 : (step.n_mels ?? 80);
  const fftSize = step.preset?.startsWith("whisper") ? 400 : (step.fft_size ?? 400);
  const hop = step.preset?.startsWith("whisper") ? 160 : (step.hop_length ?? 160);
  const maxFrames = step.max_frames ?? 3000;
  const frames = Math.min(maxFrames, Math.max(1, Math.floor((data.samples.length - fftSize) / hop) + 1));
  const out = new Float32Array(nMels * frames);
  for (let frame = 0; frame < frames; frame += 1) {
    const offset = frame * hop;
    for (let mel = 0; mel < nMels; mel += 1) {
      let energy = 0;
      const start = offset + Math.floor((mel * fftSize) / nMels);
      const end = offset + Math.floor(((mel + 1) * fftSize) / nMels);
      for (let i = start; i < end && i < data.samples.length; i += 1) {
        const sample = data.samples[i] ?? 0;
        energy += sample * sample;
      }
      out[mel * frames + frame] = Math.log10(Math.max(1e-10, energy / Math.max(1, end - start)));
    }
  }
  return { kind: "tensor", tensor: XybridTensor.fromFloat32(out, { shape: [1, nMels, frames] }) };
}

function tokenizeStep(data: PreprocessedData, vocabContent: string, maxLength: number | undefined): PreprocessedData {
  if (data.kind !== "text") {
    throw new XybridError("runtime_error", "Tokenize requires text input");
  }
  const vocab = new Map<string, number>();
  for (const [index, line] of vocabContent.split(/\r?\n/).entries()) {
    const token = line.trim();
    if (token) {
      vocab.set(token, index);
    }
  }
  const unknown = vocab.get("[UNK]") ?? vocab.get("<unk>") ?? 0;
  const pieces = data.text.toLowerCase().split(/\s+/).filter(Boolean);
  const ids = pieces.map((piece) => vocab.get(piece) ?? unknown);
  const truncated = maxLength === undefined ? ids : ids.slice(0, maxLength);
  return {
    kind: "token-ids",
    ids: truncated,
    attentionMask: truncated.map(() => 1),
    tokenTypeIds: truncated.map(() => 0),
    vocabFile: "",
    originalText: data.text,
  };
}

function toTensor(data: PreprocessedData): XybridTensor {
  if (data.kind === "tensor") {
    return data.tensor;
  }
  if (data.kind === "audio-samples") {
    return XybridTensor.fromFloat32(data.samples, { shape: [1, data.samples.length] });
  }
  throw new XybridError("runtime_error", "Cannot convert preprocessed data to tensor");
}

function toFloatTensor(data: PreprocessedData, op: string): XybridTensor & { readonly data: Float32Array } {
  const tensor = toTensor(data);
  if (!(tensor.data instanceof Float32Array)) {
    throw new XybridError("tensor_invalid", `${op} requires float32 tensor input`);
  }
  return tensor as XybridTensor & { readonly data: Float32Array };
}

function inferChannels(shape: readonly number[], fallback: number): number {
  if (shape.length === 4) {
    return shape[1] ?? fallback;
  }
  if (shape.length === 3) {
    return shape[0] ?? fallback;
  }
  return fallback;
}

function imageShape(shape: readonly number[]): { batch: number; channels: number; sourceHeight: number; sourceWidth: number } {
  if (shape.length === 4) {
    return { batch: shape[0] ?? 1, channels: shape[1] ?? 1, sourceHeight: shape[2] ?? 1, sourceWidth: shape[3] ?? 1 };
  }
  if (shape.length === 3) {
    return { batch: 1, channels: shape[0] ?? 1, sourceHeight: shape[1] ?? 1, sourceWidth: shape[2] ?? 1 };
  }
  throw new XybridError("runtime_error", `Image step expects CHW or NCHW tensor, got [${shape.join(", ")}]`);
}

function assertNever(value: never): never {
  throw new XybridError("unsupported_step", `Unsupported preprocessing step ${(value as { type?: string }).type ?? "unknown"}`);
}
