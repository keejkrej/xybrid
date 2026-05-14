import { ArtifactResolver } from "./artifact-resolver.js";
import { XybridError } from "./errors.js";
import type { VoiceConfig, VoiceInfo } from "./types.js";

const DEFAULT_EMBEDDING_DIM = 256;

export function listVoices(config: VoiceConfig | undefined): VoiceInfo[] {
  return config === undefined ? [] : [...config.catalog];
}

export function defaultVoice(config: VoiceConfig | undefined): VoiceInfo | undefined {
  if (config === undefined) {
    return undefined;
  }
  if (config.default !== undefined) {
    return config.catalog.find((voice) => voice.id === config.default) ?? config.catalog[0];
  }
  return config.catalog[0];
}

export function loadVoiceEmbedding(
  config: VoiceConfig | undefined,
  artifacts: ArtifactResolver,
  options: { readonly voice?: string; readonly tokenCount?: number } = {},
): Float32Array {
  if (config === undefined) {
    throw new XybridError("runtime_error", "Model has no voice configuration");
  }
  const selected = selectVoice(config, options.voice);
  if (config.format.type === "Embedded") {
    if (config.format.loader !== "BinaryF32_256") {
      throw new XybridError("unsupported_step", `Voice loader '${config.format.loader}' is not supported in the browser SDK yet`);
    }
    const embeddingDim = config.format.embedding_dim ?? DEFAULT_EMBEDDING_DIM;
    return loadBinaryF32Embedding(artifacts.bytes(config.format.file), embeddingDim, selected.index);
  }
  if (config.format.type === "PerModel") {
    const file = `${config.format.voice_dir.replace(/\/?$/, "/")}${config.format.pattern.replace("{voice}", selected.id)}`;
    const embeddingDim = "embedding_dim" in config.format && typeof config.format.embedding_dim === "number"
      ? config.format.embedding_dim
      : DEFAULT_EMBEDDING_DIM;
    return loadTokenLengthEmbedding(artifacts.bytes(file), embeddingDim, options.tokenCount ?? 100);
  }
  throw new XybridError("unsupported_step", `Voice format '${config.format.type}' is not supported in the browser SDK yet`);
}

function loadBinaryF32Embedding(data: ArrayBuffer, embeddingDim: number, voiceIndex: number): Float32Array {
  const bytes = new Uint8Array(data);
  const voiceSize = embeddingDim * 4;
  const start = voiceIndex * voiceSize;
  const end = start + voiceSize;
  if (end > bytes.byteLength) {
    throw new XybridError("runtime_error", `Voice index ${voiceIndex} is out of range`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset + start, voiceSize);
  const embedding = new Float32Array(embeddingDim);
  for (let i = 0; i < embeddingDim; i += 1) {
    embedding[i] = view.getFloat32(i * 4, true);
  }
  return embedding;
}

function loadTokenLengthEmbedding(data: ArrayBuffer, embeddingDim: number, tokenCount: number): Float32Array {
  const bytes = new Uint8Array(data);
  const rows = Math.floor(bytes.byteLength / (embeddingDim * 4));
  if (rows <= 0) {
    throw new XybridError("runtime_error", "Voice file does not contain any embeddings");
  }
  const row = Math.max(0, Math.min(rows - 1, tokenCount));
  return loadBinaryF32Embedding(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), embeddingDim, row);
}

function selectVoice(config: VoiceConfig, requested: string | undefined): VoiceInfo {
  const selected =
    requested === undefined
      ? defaultVoice(config)
      : config.catalog.find((voice) => voice.id === requested || voice.name === requested);
  if (selected === undefined) {
    const available = config.catalog.map((voice) => voice.id).join(", ");
    throw new XybridError("runtime_error", `Voice '${requested}' not found. Available voices: ${available}`);
  }
  return selected;
}
