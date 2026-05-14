import { XybridError } from "./errors.js";
import type {
  ExecutionTemplate,
  PostprocessingStep,
  PreprocessingStep,
  TensorDType,
  TensorMetadata,
  VoiceConfig,
  VoiceFormat,
  VoiceInfo,
  XybridWebManifest,
  XybridWebManifestFile,
} from "./types.js";

const DTYPE_VALUES = new Set<TensorDType>([
  "float32",
  "float64",
  "int32",
  "int64",
  "uint8",
  "int8",
  "uint16",
  "int16",
  "bool",
]);

export function validateManifest(value: unknown): XybridWebManifest {
  if (!isRecord(value)) {
    throw invalid("Manifest must be a JSON object");
  }

  const modelId = optionalString(value, "model_id") ?? optionalString(value, "modelId") ?? failString("model_id");
  const version = readString(value, "version");
  const executionTemplate = readExecutionTemplate(
    value.execution_template ?? value.executionTemplate,
    value.execution_template === undefined ? "executionTemplate" : "execution_template",
  );

  const rawArtifacts = value.artifacts ?? value.files;
  const artifacts = readArtifacts(rawArtifacts, rawArtifacts === value.artifacts ? "artifacts" : "files");
  if (!artifacts.some((file) => file.path === executionTemplate.modelFile)) {
    throw invalid("execution_template.model_file must be present in artifacts/files");
  }

  return {
    modelId,
    version,
    model_id: modelId,
    executionTemplate,
    execution_template: executionTemplate,
    files: artifacts,
    artifacts,
    inputs: readOptionalArray(value, "inputs").map((item, index) => readTensorMetadata(item, `inputs[${index}]`)),
    outputs: readOptionalArray(value, "outputs").map((item, index) => readTensorMetadata(item, `outputs[${index}]`)),
    preprocessing: readOptionalArray(value, "preprocessing").map((item, index) =>
      readPreprocessingStep(item, `preprocessing[${index}]`),
    ),
    postprocessing: readOptionalArray(value, "postprocessing").map((item, index) =>
      readPostprocessingStep(item, `postprocessing[${index}]`),
    ),
    ...(value.voices === undefined ? {} : { voices: readVoiceConfig(value.voices) }),
    ...(optionalString(value, "description") === undefined ? {} : { description: optionalString(value, "description") }),
    ...(isRecord(value.metadata) ? { metadata: value.metadata } : {}),
    ...(optionalNumber(value, "max_chunk_chars") === undefined ? {} : { max_chunk_chars: optionalNumber(value, "max_chunk_chars") }),
    ...(optionalNumber(value, "trim_trailing_samples") === undefined
      ? {}
      : { trim_trailing_samples: optionalNumber(value, "trim_trailing_samples") }),
  } as XybridWebManifest;
}

export function resolveArtifactUrl(manifestUrl: string, artifactPath: string): string {
  const base =
    typeof globalThis.location === "object" && typeof globalThis.location.href === "string"
      ? globalThis.location.href
      : "http://localhost/";
  return new URL(artifactPath, new URL(manifestUrl, base)).href;
}

function readTensorMetadata(value: unknown, label: string): TensorMetadata {
  if (!isRecord(value)) {
    throw invalid(`${label} must be an object`);
  }
  const dtype = readString(value, "dtype");
  if (!DTYPE_VALUES.has(dtype as TensorDType)) {
    throw invalid(`${label}.dtype is not supported: ${dtype}`);
  }
  const shape = readArray(value, "shape").map((dim, index) => {
    if (typeof dim === "number" && Number.isInteger(dim) && dim >= 0) {
      return dim;
    }
    if (typeof dim === "string" && dim.length > 0) {
      return dim;
    }
    if (dim === null) {
      return null;
    }
    throw invalid(`${label}.shape[${index}] must be a non-negative integer, string, or null`);
  });
  const layout = optionalString(value, "layout");
  const description = optionalString(value, "description");
  return {
    name: readString(value, "name"),
    dtype: dtype as TensorDType,
    shape,
    ...(layout === undefined ? {} : { layout }),
    ...(description === undefined ? {} : { description }),
  };
}

function readExecutionTemplate(value: unknown, label: string): ExecutionTemplate {
  if (!isRecord(value)) {
    throw invalid(`Manifest ${label} must be an object`);
  }
  if (value.type !== "onnx" && value.type !== "Onnx") {
    throw invalid("Only ONNX execution templates are supported");
  }
  const modelFile = optionalString(value, "model_file") ?? optionalString(value, "modelFile") ?? failString(`${label}.model_file`);
  return { type: "onnx", modelFile };
}

function readArtifacts(value: unknown, label: string): XybridWebManifestFile[] {
  const array = readArrayValue(value, label);
  return array.map((file, index) => {
    if (typeof file === "string") {
      return { path: file };
    }
    if (!isRecord(file)) {
      throw invalid(`${label}[${index}] must be an object or string`);
    }
    const path = readString(file, "path");
    const sizeBytes = optionalNumber(file, "sizeBytes") ?? optionalNumber(file, "size_bytes");
    const sha256 = optionalString(file, "sha256");
    if (sha256 !== undefined && !/^[a-fA-F0-9]{64}$/.test(sha256)) {
      throw invalid(`${label}[${index}].sha256 must be a 64-character hex digest`);
    }
    return { path, ...(sizeBytes === undefined ? {} : { sizeBytes }), ...(sha256 === undefined ? {} : { sha256 }) };
  });
}

function readPreprocessingStep(value: unknown, label: string): PreprocessingStep {
  if (!isRecord(value)) {
    throw invalid(`${label} must be an object`);
  }
  const type = readString(value, "type");
  switch (type) {
    case "MelSpectrogram":
      return {
        type,
        ...(optionalString(value, "preset") === undefined ? {} : { preset: optionalString(value, "preset") }),
        ...(optionalNumber(value, "n_mels") === undefined ? {} : { n_mels: optionalNumber(value, "n_mels") }),
        ...(optionalNumber(value, "sample_rate") === undefined ? {} : { sample_rate: optionalNumber(value, "sample_rate") }),
        ...(optionalNumber(value, "fft_size") === undefined ? {} : { fft_size: optionalNumber(value, "fft_size") }),
        ...(optionalNumber(value, "hop_length") === undefined ? {} : { hop_length: optionalNumber(value, "hop_length") }),
        ...(optionalString(value, "mel_scale") === undefined ? {} : { mel_scale: optionalString(value, "mel_scale") }),
        ...(optionalNumberOrNull(value, "max_frames") === undefined ? {} : { max_frames: optionalNumberOrNull(value, "max_frames") }),
      } as PreprocessingStep;
    case "Tokenize":
      return {
        type,
        vocab_file: readString(value, "vocab_file"),
        tokenizer_type: readString(value, "tokenizer_type"),
        ...(optionalNumberOrNull(value, "max_length") === undefined ? {} : { max_length: optionalNumberOrNull(value, "max_length") }),
      } as PreprocessingStep;
    case "Normalize":
      return { type, mean: readNumberArray(value, "mean"), std: readNumberArray(value, "std") };
    case "Resize":
      return {
        type,
        width: readNumber(value, "width"),
        height: readNumber(value, "height"),
        ...(optionalString(value, "interpolation") === undefined ? {} : { interpolation: optionalString(value, "interpolation") }),
      } as PreprocessingStep;
    case "CenterCrop":
      return { type, width: readNumber(value, "width"), height: readNumber(value, "height") };
    case "AudioDecode":
      return { type, sample_rate: readNumber(value, "sample_rate"), channels: readNumber(value, "channels") };
    case "Reshape":
      return { type, shape: readNumberArray(value, "shape") };
    case "PhonemeRaw":
      return {
        type,
        ...(optionalString(value, "backend") === undefined ? {} : { backend: optionalString(value, "backend") as PreprocessingStep["type"] extends never ? never : never }),
        ...(optionalStringOrNull(value, "language") === undefined ? {} : { language: optionalStringOrNull(value, "language") }),
      } as PreprocessingStep;
    case "Phonemize":
      return {
        type,
        tokens_file: readString(value, "tokens_file"),
        ...(optionalString(value, "backend") === undefined ? {} : { backend: optionalString(value, "backend") }),
        ...(optionalStringOrNull(value, "dict_file") === undefined ? {} : { dict_file: optionalStringOrNull(value, "dict_file") }),
        ...(optionalStringOrNull(value, "language") === undefined ? {} : { language: optionalStringOrNull(value, "language") }),
        ...(optionalBoolean(value, "add_padding") === undefined ? {} : { add_padding: optionalBoolean(value, "add_padding") }),
        ...(optionalBoolean(value, "normalize_text") === undefined ? {} : { normalize_text: optionalBoolean(value, "normalize_text") }),
        ...(optionalNumberOrNull(value, "silence_tokens") === undefined ? {} : { silence_tokens: optionalNumberOrNull(value, "silence_tokens") }),
      } as PreprocessingStep;
    default:
      throw invalid(`${label}.type is not supported: ${type}`);
  }
}

function readPostprocessingStep(value: unknown, label: string): PostprocessingStep {
  if (!isRecord(value)) {
    throw invalid(`${label} must be an object`);
  }
  const type = readString(value, "type");
  switch (type) {
    case "BPEDecode":
      return { type, vocab_file: readString(value, "vocab_file") };
    case "Argmax":
      return { type, ...(optionalNumberOrNull(value, "dim") === undefined ? {} : { dim: optionalNumberOrNull(value, "dim") }) };
    case "Softmax":
      return { type, ...(optionalNumberOrNull(value, "dim") === undefined ? {} : { dim: optionalNumberOrNull(value, "dim") }) };
    case "TopK":
      return { type, k: readNumber(value, "k"), ...(optionalNumberOrNull(value, "dim") === undefined ? {} : { dim: optionalNumberOrNull(value, "dim") }) };
    case "Threshold":
      return {
        type,
        threshold: readFiniteNumber(value, "threshold"),
        ...(optionalBoolean(value, "return_indices") === undefined ? {} : { return_indices: optionalBoolean(value, "return_indices") }),
      };
    case "TemperatureSample":
      return {
        type,
        temperature: readFiniteNumber(value, "temperature"),
        ...(optionalNumberOrNull(value, "top_k") === undefined ? {} : { top_k: optionalNumberOrNull(value, "top_k") }),
        ...(optionalFiniteNumberOrNull(value, "top_p") === undefined ? {} : { top_p: optionalFiniteNumberOrNull(value, "top_p") }),
      };
    case "Denormalize":
      return { type, mean: readNumberArray(value, "mean"), std: readNumberArray(value, "std") };
    case "MeanPool":
      return { type, ...(optionalNumber(value, "dim") === undefined ? {} : { dim: optionalNumber(value, "dim") }) };
    case "CTCDecode":
      return {
        type,
        vocab_file: readString(value, "vocab_file"),
        ...(optionalNumber(value, "blank_index") === undefined ? {} : { blank_index: optionalNumber(value, "blank_index") }),
      };
    case "TTSAudioEncode":
      return {
        type,
        sample_rate: readNumber(value, "sample_rate"),
        ...(optionalBoolean(value, "apply_postprocessing") === undefined ? {} : { apply_postprocessing: optionalBoolean(value, "apply_postprocessing") }),
        ...(optionalBoolean(value, "trim_trailing_silence") === undefined
          ? {}
          : { trim_trailing_silence: optionalBoolean(value, "trim_trailing_silence") }),
      };
    case "WhisperDecode":
      return { type, tokenizer_file: readString(value, "tokenizer_file") };
    case "CodecDecode":
      return {
        type,
        decoder_model: readString(value, "decoder_model"),
        sample_rate: readNumber(value, "sample_rate"),
        token_pattern: readString(value, "token_pattern"),
        ...(optionalBoolean(value, "apply_postprocessing") === undefined ? {} : { apply_postprocessing: optionalBoolean(value, "apply_postprocessing") }),
      };
    default:
      throw invalid(`${label}.type is not supported: ${type}`);
  }
}

function readVoiceConfig(value: unknown): VoiceConfig {
  if (!isRecord(value)) {
    throw invalid("voices must be an object");
  }
  return {
    format: readVoiceFormat(value.format),
    ...(optionalString(value, "default") === undefined ? {} : { default: optionalString(value, "default") }),
    catalog: readArray(value, "catalog").map((item, index) => readVoiceInfo(item, `voices.catalog[${index}]`)),
    ...(optionalString(value, "selection_strategy") === undefined
      ? {}
      : { selection_strategy: optionalString(value, "selection_strategy") }),
  };
}

function readVoiceFormat(value: unknown): VoiceFormat {
  if (!isRecord(value)) {
    throw invalid("voices.format must be an object");
  }
  const type = readString(value, "type");
  switch (type) {
    case "Embedded":
      return {
        type,
        file: readString(value, "file"),
        loader: readString(value, "loader"),
        ...(optionalNumber(value, "embedding_dim") === undefined ? {} : { embedding_dim: optionalNumber(value, "embedding_dim") }),
      };
    case "PerModel":
      return {
        type,
        voice_dir: readString(value, "voice_dir"),
        pattern: readString(value, "pattern"),
        ...(optionalNumber(value, "embedding_dim") === undefined ? {} : { embedding_dim: optionalNumber(value, "embedding_dim") }),
      } as VoiceFormat;
    case "PrecomputedCodes":
      return { type, codes_dir: readString(value, "codes_dir"), transcript_dir: readString(value, "transcript_dir") };
    case "Cloning":
      return { type, encoder_model: readString(value, "encoder_model") };
    default:
      throw invalid(`voices.format.type is not supported: ${type}`);
  }
}

function readVoiceInfo(value: unknown, label: string): VoiceInfo {
  if (!isRecord(value)) {
    throw invalid(`${label} must be an object`);
  }
  return {
    id: readString(value, "id"),
    name: readString(value, "name"),
    index: readNumber(value, "index"),
    ...(optionalString(value, "gender") === undefined ? {} : { gender: optionalString(value, "gender") }),
    ...(optionalString(value, "language") === undefined ? {} : { language: optionalString(value, "language") }),
    ...(optionalString(value, "description") === undefined ? {} : { description: optionalString(value, "description") }),
  };
}

function readString(value: Record<string, unknown>, key: string): string {
  const field = value[key];
  if (typeof field !== "string" || field.length === 0) {
    throw invalid(`${key} must be a non-empty string`);
  }
  return field;
}

function failString(key: string): never {
  throw invalid(`${key} must be a non-empty string`);
}

function optionalString(value: Record<string, unknown>, key: string): string | undefined {
  const field = value[key];
  if (field === undefined) {
    return undefined;
  }
  if (typeof field !== "string" || field.length === 0) {
    throw invalid(`${key} must be a non-empty string when present`);
  }
  return field;
}

function optionalNumber(value: Record<string, unknown>, key: string): number | undefined {
  const field = value[key];
  if (field === undefined) {
    return undefined;
  }
  if (typeof field !== "number" || !Number.isInteger(field) || field < 0) {
    throw invalid(`${key} must be a non-negative integer when present`);
  }
  return field;
}

function optionalNumberOrNull(value: Record<string, unknown>, key: string): number | null | undefined {
  const field = value[key];
  if (field === undefined) {
    return undefined;
  }
  if (field === null) {
    return null;
  }
  return readNumber(value, key);
}

function optionalFiniteNumberOrNull(value: Record<string, unknown>, key: string): number | null | undefined {
  const field = value[key];
  if (field === undefined) {
    return undefined;
  }
  if (field === null) {
    return null;
  }
  return readFiniteNumber(value, key);
}

function optionalStringOrNull(value: Record<string, unknown>, key: string): string | null | undefined {
  const field = value[key];
  if (field === undefined) {
    return undefined;
  }
  if (field === null) {
    return null;
  }
  return optionalString(value, key);
}

function optionalBoolean(value: Record<string, unknown>, key: string): boolean | undefined {
  const field = value[key];
  if (field === undefined) {
    return undefined;
  }
  if (typeof field !== "boolean") {
    throw invalid(`${key} must be a boolean when present`);
  }
  return field;
}

function readNumber(value: Record<string, unknown>, key: string): number {
  const field = value[key];
  if (typeof field !== "number" || !Number.isInteger(field) || field < 0) {
    throw invalid(`${key} must be a non-negative integer`);
  }
  return field;
}

function readFiniteNumber(value: Record<string, unknown>, key: string): number {
  const field = value[key];
  if (typeof field !== "number" || !Number.isFinite(field)) {
    throw invalid(`${key} must be a finite number`);
  }
  return field;
}

function readNumberArray(value: Record<string, unknown>, key: string): number[] {
  return readArray(value, key).map((item, index) => {
    if (typeof item !== "number" || !Number.isFinite(item)) {
      throw invalid(`${key}[${index}] must be a finite number`);
    }
    return item;
  });
}

function readArray(value: Record<string, unknown>, key: string): unknown[] {
  return readArrayValue(value[key], key);
}

function readOptionalArray(value: Record<string, unknown>, key: string): unknown[] {
  const field = value[key];
  if (field === undefined) {
    return [];
  }
  return readArrayValue(field, key);
}

function readArrayValue(value: unknown, key: string): unknown[] {
  if (!Array.isArray(value)) {
    throw invalid(`${key} must be an array`);
  }
  return value;
}

function invalid(message: string): XybridError {
  return new XybridError("manifest_invalid", message);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
