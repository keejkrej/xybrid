import type { ArtifactCache } from "./cache.js";

export type BackendKind = "ort-wasm" | "ort-webgpu";

export type CacheMode = "persistent" | "memory" | "none";

export type TensorLayout = "NCHW" | "NHWC" | "NCW" | "NWC" | "CHW" | "HWC" | "C" | string;

export type TensorDType =
  | "float32"
  | "float64"
  | "int32"
  | "int64"
  | "uint8"
  | "int8"
  | "uint16"
  | "int16"
  | "bool";

export type TensorShape = readonly (number | string | null)[];

export interface TensorMetadata {
  readonly name: string;
  readonly dtype: TensorDType;
  readonly shape: TensorShape;
  readonly layout?: TensorLayout | undefined;
  readonly description?: string | undefined;
}

export type ExecutionTemplate = {
  readonly type: "onnx";
  readonly modelFile: string;
};

export type PreprocessingStep =
  | {
      readonly type: "MelSpectrogram";
      readonly preset?: string | undefined;
      readonly n_mels?: number | undefined;
      readonly sample_rate?: number | undefined;
      readonly fft_size?: number | undefined;
      readonly hop_length?: number | undefined;
      readonly mel_scale?: "Slaney" | "HTK" | string | undefined;
      readonly max_frames?: number | null | undefined;
    }
  | {
      readonly type: "Tokenize";
      readonly vocab_file: string;
      readonly tokenizer_type: "WordPiece" | "BPE" | "SentencePiece" | string;
      readonly max_length?: number | null | undefined;
    }
  | { readonly type: "Normalize"; readonly mean: readonly number[]; readonly std: readonly number[] }
  | {
      readonly type: "Resize";
      readonly width: number;
      readonly height: number;
      readonly interpolation?: "Nearest" | "Bilinear" | "Bicubic" | string | undefined;
    }
  | { readonly type: "CenterCrop"; readonly width: number; readonly height: number }
  | { readonly type: "AudioDecode"; readonly sample_rate: number; readonly channels: number }
  | { readonly type: "Reshape"; readonly shape: readonly number[] }
  | {
      readonly type: "PhonemeRaw";
      readonly backend?: PhonemizerBackendKind | undefined;
      readonly language?: string | null | undefined;
    }
  | {
      readonly type: "Phonemize";
      readonly tokens_file: string;
      readonly backend?: PhonemizerBackendKind | undefined;
      readonly dict_file?: string | null | undefined;
      readonly language?: string | null | undefined;
      readonly add_padding?: boolean | undefined;
      readonly normalize_text?: boolean | undefined;
      readonly silence_tokens?: number | null | undefined;
    };

export type PostprocessingStep =
  | { readonly type: "BPEDecode"; readonly vocab_file: string }
  | { readonly type: "Argmax"; readonly dim?: number | null | undefined }
  | { readonly type: "Softmax"; readonly dim?: number | null | undefined }
  | { readonly type: "TopK"; readonly k: number; readonly dim?: number | null | undefined }
  | { readonly type: "Threshold"; readonly threshold: number; readonly return_indices?: boolean | undefined }
  | { readonly type: "TemperatureSample"; readonly temperature: number; readonly top_k?: number | null | undefined; readonly top_p?: number | null | undefined }
  | { readonly type: "Denormalize"; readonly mean: readonly number[]; readonly std: readonly number[] }
  | { readonly type: "MeanPool"; readonly dim?: number | undefined }
  | { readonly type: "CTCDecode"; readonly vocab_file: string; readonly blank_index?: number | undefined }
  | { readonly type: "TTSAudioEncode"; readonly sample_rate: number; readonly apply_postprocessing?: boolean | undefined; readonly trim_trailing_silence?: boolean | undefined }
  | { readonly type: "WhisperDecode"; readonly tokenizer_file: string }
  | {
      readonly type: "CodecDecode";
      readonly decoder_model: string;
      readonly sample_rate: number;
      readonly token_pattern: string;
      readonly apply_postprocessing?: boolean | undefined;
    };

export type PhonemizerBackendKind = "CmuDictionary" | "EspeakNG" | "MisakiDictionary" | "OpenPhonemizer";

export interface VoiceInfo {
  readonly id: string;
  readonly name: string;
  readonly index: number;
  readonly gender?: string | undefined;
  readonly language?: string | undefined;
  readonly description?: string | undefined;
}

export interface VoiceConfig {
  readonly format: VoiceFormat;
  readonly default?: string | undefined;
  readonly catalog: readonly VoiceInfo[];
  readonly selection_strategy?: "FixedIndex" | "TokenLength" | string | undefined;
}

export type VoiceFormat =
  | {
      readonly type: "Embedded";
      readonly file: string;
      readonly loader: "BinaryF32_256" | "NumpyNpz" | string;
      readonly embedding_dim?: number | undefined;
    }
  | {
      readonly type: "PerModel";
      readonly voice_dir: string;
      readonly pattern: string;
      readonly embedding_dim?: number | undefined;
    }
  | {
      readonly type: "PrecomputedCodes";
      readonly codes_dir: string;
      readonly transcript_dir: string;
    }
  | {
      readonly type: "Cloning";
      readonly encoder_model: string;
    };

export interface XybridWebManifest {
  readonly modelId: string;
  readonly version: string;
  readonly executionTemplate: ExecutionTemplate;
  readonly model_id: string;
  readonly execution_template: ExecutionTemplate;
  readonly files: readonly XybridWebManifestFile[];
  readonly artifacts: readonly XybridWebManifestFile[];
  readonly inputs: readonly TensorMetadata[];
  readonly outputs: readonly TensorMetadata[];
  readonly preprocessing: readonly PreprocessingStep[];
  readonly postprocessing: readonly PostprocessingStep[];
  readonly voices?: VoiceConfig | undefined;
  readonly description?: string | undefined;
  readonly metadata?: Record<string, unknown> | undefined;
  readonly max_chunk_chars?: number | undefined;
  readonly trim_trailing_samples?: number | undefined;
}

export interface XybridWebManifestFile {
  readonly path: string;
  readonly sizeBytes?: number;
  readonly sha256?: string;
}

export type LoadProgressEvent =
  | {
      readonly type: "manifest";
      readonly manifestUrl: string;
    }
  | {
      readonly type: "cache-hit";
      readonly path: string;
      readonly loadedFiles: number;
      readonly totalFiles: number;
      readonly loadedBytes: number;
      readonly totalBytes?: number;
    }
  | {
      readonly type: "download-start";
      readonly path: string;
      readonly loadedFiles: number;
      readonly totalFiles: number;
      readonly loadedBytes: number;
      readonly totalBytes?: number;
    }
  | {
      readonly type: "download-progress";
      readonly path: string;
      readonly loadedFiles: number;
      readonly totalFiles: number;
      readonly loadedBytes: number;
      readonly totalBytes?: number;
      readonly fileLoadedBytes: number;
      readonly fileTotalBytes?: number;
    }
  | {
      readonly type: "verify";
      readonly path: string;
      readonly loadedFiles: number;
      readonly totalFiles: number;
      readonly loadedBytes: number;
      readonly totalBytes?: number;
    }
  | {
      readonly type: "ready";
      readonly loadedFiles: number;
      readonly totalFiles: number;
      readonly loadedBytes: number;
      readonly totalBytes?: number;
    };

export interface XybridCreateOptions {
  readonly backend?: BackendKind;
  readonly cache?: CacheMode;
  readonly wasmPaths?: string;
  readonly wasmNumThreads?: number;
  readonly fetch?: FetchLike;
  /**
   * Overrides SDK cache selection. Intended for tests and embedders with their
   * own storage policy.
   */
  readonly artifactCache?: ArtifactCache;
}

export interface XybridLoadOptions {
  readonly onProgress?: (event: LoadProgressEvent) => void;
  readonly signal?: AbortSignal;
}

export interface XybridRunTensorRequest {
  readonly inputs: Record<string, import("./tensor.js").XybridTensor>;
}

export interface XybridRunTensorResult {
  readonly outputs: Record<string, import("./tensor.js").XybridTensor>;
}

export interface XybridRunOptions {
  readonly signal?: AbortSignal;
  readonly voice?: string;
  readonly speed?: number;
}

export type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
