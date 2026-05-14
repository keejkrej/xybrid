import * as ort from "onnxruntime-web";
import { ArtifactResolver } from "./artifact-resolver.js";
import { loadArtifacts } from "./artifacts.js";
import { createArtifactCache, type ArtifactCache } from "./cache.js";
import { envelopeFromInput, type XybridRunInput } from "./envelope.js";
import { abortError, isAbortError, XybridError } from "./errors.js";
import { validateManifest } from "./manifest.js";
import { preprocessedFromEnvelope, type PreprocessedData, type RawOutputs } from "./intermediate.js";
import { XybridRunResult } from "./result.js";
import { resultFromRawOutputs, runPostprocessing } from "./postprocessing.js";
import { runPreprocessing } from "./preprocessing.js";
import { XybridTensor, type XybridTensorData } from "./tensor.js";
import { runTts } from "./tts.js";
import type {
  BackendKind,
  CacheMode,
  FetchLike,
  TensorDType,
  XybridCreateOptions,
  XybridLoadOptions,
  XybridRunOptions,
  XybridRunTensorRequest,
  XybridRunTensorResult,
  XybridWebManifest,
  VoiceInfo,
} from "./types.js";

export class Xybrid {
  readonly backend: BackendKind;
  readonly cacheMode: CacheMode;
  readonly cacheKind: ArtifactCache["kind"];

  private constructor(
    options: {
      readonly backend: BackendKind;
      readonly cacheMode: CacheMode;
      readonly cache: ArtifactCache;
      readonly fetch: FetchLike;
    },
  ) {
    this.backend = options.backend;
    this.cacheMode = options.cacheMode;
    this.cacheKind = options.cache.kind;
    this.cache = options.cache;
    this.fetch = options.fetch;
  }

  private readonly cache: ArtifactCache;
  private readonly fetch: FetchLike;

  static async create(options: XybridCreateOptions = {}): Promise<Xybrid> {
    const backend = options.backend ?? "ort-wasm";
    const cacheMode = options.cache ?? "persistent";
    if (options.wasmPaths !== undefined) {
      ort.env.wasm.wasmPaths = options.wasmPaths;
    }
    if (options.wasmNumThreads !== undefined) {
      ort.env.wasm.numThreads = options.wasmNumThreads;
    }
    const fetchLike = options.fetch ?? globalThis.fetch?.bind(globalThis);
    if (!fetchLike) {
      throw new XybridError("fetch_failed", "A fetch implementation is required");
    }
    return new Xybrid({
      backend,
      cacheMode,
      cache: options.artifactCache ?? (await createArtifactCache(cacheMode)),
      fetch: fetchLike,
    });
  }

  model(manifestUrl: string): XybridModelRef {
    return new XybridModelRef(manifestUrl, this.backend, this.cache, this.fetch);
  }
}

export class XybridModelRef {
  constructor(
    readonly manifestUrl: string,
    private readonly backend: BackendKind,
    private readonly cache: ArtifactCache,
    private readonly fetch: FetchLike,
  ) {}

  async load(options: XybridLoadOptions = {}): Promise<XybridModel> {
    options.onProgress?.({ type: "manifest", manifestUrl: this.manifestUrl });
    const manifest = await this.fetchManifest(options.signal);
    const artifacts = await loadArtifacts({
      manifest,
      manifestUrl: this.manifestUrl,
      cache: this.cache,
      fetch: this.fetch,
      ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
    const modelBytes = artifacts.get(manifest.executionTemplate.modelFile);
    if (modelBytes === undefined) {
      throw new XybridError("manifest_invalid", "Manifest modelFile was not loaded");
    }
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: executionProviders(this.backend),
    } as ort.InferenceSession.SessionOptions);
    const loadedBytes = [...artifacts.values()].reduce((sum, data) => sum + data.byteLength, 0);
    const totalBytes = manifest.files.every((file) => file.sizeBytes !== undefined)
      ? manifest.files.reduce((sum, file) => sum + (file.sizeBytes ?? 0), 0)
      : undefined;
    options.onProgress?.({
      type: "ready",
      loadedFiles: manifest.files.length,
      totalFiles: manifest.files.length,
      loadedBytes,
      ...(totalBytes === undefined ? {} : { totalBytes }),
    });
    return new XybridModel(manifest, session, artifacts);
  }

  private async fetchManifest(signal: AbortSignal | undefined): Promise<XybridWebManifest> {
    let response: Response;
    try {
      response = await this.fetch(this.manifestUrl, signal === undefined ? {} : { signal });
    } catch (error) {
      if (isAbortError(error) || signal?.aborted) {
        throw abortError(`Manifest fetch aborted for '${this.manifestUrl}'`);
      }
      throw new XybridError("fetch_failed", `Failed to fetch manifest '${this.manifestUrl}'`, { cause: error });
    }
    if (!response.ok) {
      throw new XybridError("fetch_failed", `Failed to fetch manifest '${this.manifestUrl}': HTTP ${response.status}`);
    }
    try {
      return validateManifest(await response.json());
    } catch (error) {
      if (error instanceof XybridError) {
        throw error;
      }
      throw new XybridError("manifest_invalid", "Manifest is not valid JSON", { cause: error });
    }
  }
}

export class XybridModel {
  readonly modelId: string;
  readonly version: string;
  readonly inputs: XybridWebManifest["inputs"];
  readonly outputs: XybridWebManifest["outputs"];

  constructor(
    readonly manifest: XybridWebManifest,
    private readonly session: ort.InferenceSession,
    private readonly artifacts: ReadonlyMap<string, ArrayBuffer>,
  ) {
    this.modelId = manifest.modelId;
    this.version = manifest.version;
    this.inputs = manifest.inputs;
    this.outputs = manifest.outputs;
  }

  async run(input: XybridRunInput, options: XybridRunOptions = {}): Promise<XybridRunResult> {
    if (options.signal?.aborted) {
      throw abortError();
    }
    const artifacts = new ArtifactResolver(this.artifacts);
    const envelope = envelopeFromInput(input);
    const preprocessed = await runPreprocessing(this.manifest, preprocessedFromEnvelope(envelope), artifacts);
    if (options.signal?.aborted) {
      throw abortError();
    }
    const raw = await this.executePreprocessed(preprocessed, artifacts, options);
    const final = await runPostprocessing(this.manifest, raw, artifacts);
    return resultFromRawOutputs(final);
  }

  async runTensor(request: XybridRunTensorRequest): Promise<XybridRunTensorResult> {
    const feeds: Record<string, ort.Tensor> = {};
    for (const input of this.manifest.inputs) {
      const tensor = request.inputs[input.name];
      if (tensor === undefined) {
        throw new XybridError("tensor_invalid", `Missing required input tensor '${input.name}'`);
      }
      tensor.validateAgainst(input);
      feeds[input.name] = toOrtTensor(tensor);
    }
    for (const inputName of Object.keys(request.inputs)) {
      if (!this.manifest.inputs.some((input) => input.name === inputName)) {
        throw new XybridError("tensor_invalid", `Unknown input tensor '${inputName}'`);
      }
    }

    let results: ort.InferenceSession.ReturnType;
    try {
      results = await this.session.run(feeds);
    } catch (error) {
      throw new XybridError("runtime_error", "ONNX Runtime inference failed", { cause: error });
    }

    const outputs: Record<string, XybridTensor> = {};
    for (const [name, tensor] of Object.entries(results)) {
      outputs[name] = fromOrtTensor(name, tensor);
    }
    return { outputs };
  }

  async unload(): Promise<void> {
    const releasable = this.session as ort.InferenceSession & { release?: () => Promise<void> | void };
    await releasable.release?.();
  }

  voices(): readonly VoiceInfo[] {
    return this.manifest.voices?.catalog ?? [];
  }

  hasVoices(): boolean {
    return this.manifest.voices !== undefined;
  }

  defaultVoice(): VoiceInfo | undefined {
    const voices = this.voices();
    if (this.manifest.voices?.default !== undefined) {
      return voices.find((voice) => voice.id === this.manifest.voices?.default) ?? voices[0];
    }
    return voices[0];
  }

  private async executePreprocessed(
    data: PreprocessedData,
    artifacts: ArtifactResolver,
    options: XybridRunOptions,
  ): Promise<RawOutputs> {
    if (data.kind === "phoneme-ids") {
      return runTts(this.session, this.manifest, artifacts, data, options);
    }
    const feeds = feedsFromPreprocessed(this.manifest, data);
    try {
      const results = await this.session.run(feeds);
      return { kind: "tensor-map", outputs: Object.fromEntries(Object.entries(results).map(([name, tensor]) => [name, fromOrtTensor(name, tensor)])) };
    } catch (error) {
      throw new XybridError("runtime_error", "ONNX Runtime inference failed", { cause: error });
    }
  }
}

function executionProviders(backend: BackendKind): string[] {
  if (backend === "ort-webgpu" && typeof navigator !== "undefined" && "gpu" in navigator) {
    return ["webgpu", "wasm"];
  }
  return ["wasm"];
}

function toOrtTensor(tensor: XybridTensor): ort.Tensor {
  return new ort.Tensor(tensor.dtype as ort.Tensor.Type, tensor.data, [...tensor.shape]);
}

function feedsFromPreprocessed(manifest: XybridWebManifest, data: PreprocessedData): Record<string, ort.Tensor> {
  if (data.kind === "tensor") {
    const input = manifest.inputs[0];
    if (input === undefined) {
      throw new XybridError("manifest_invalid", "Model manifest has no inputs");
    }
    return { [input.name]: toOrtTensor(data.tensor) };
  }
  if (data.kind === "audio-samples") {
    const input = manifest.inputs[0];
    if (input === undefined) {
      throw new XybridError("manifest_invalid", "Model manifest has no inputs");
    }
    return { [input.name]: new ort.Tensor("float32", data.samples, [1, data.samples.length]) };
  }
  if (data.kind === "token-ids") {
    const feeds: Record<string, ort.Tensor> = {};
    const ids = BigInt64Array.from(data.ids.map((id) => BigInt(id)));
    const mask = BigInt64Array.from(data.attentionMask.map((id) => BigInt(id)));
    const tokenTypes = BigInt64Array.from(data.tokenTypeIds.map((id) => BigInt(id)));
    for (const input of manifest.inputs) {
      const lower = input.name.toLowerCase();
      if (lower.includes("attention")) {
        feeds[input.name] = new ort.Tensor("int64", mask, [1, data.attentionMask.length]);
      } else if (lower.includes("token_type") || lower.includes("type_ids")) {
        feeds[input.name] = new ort.Tensor("int64", tokenTypes, [1, data.tokenTypeIds.length]);
      } else {
        feeds[input.name] = new ort.Tensor("int64", ids, [1, data.ids.length]);
      }
    }
    return feeds;
  }
  throw new XybridError("runtime_error", `Cannot execute preprocessed data kind '${data.kind}'`);
}

function fromOrtTensor(name: string, tensor: ort.Tensor): XybridTensor {
  return XybridTensor.fromData(
    tensor.data as XybridTensorData,
    normalizeOrtDType(name, tensor.type),
    { shape: [...tensor.dims] },
  );
}

function normalizeOrtDType(name: string, dtype: ort.Tensor.Type): TensorDType {
  switch (dtype) {
    case "float32":
    case "float64":
    case "int32":
    case "int64":
    case "uint8":
    case "int8":
    case "uint16":
    case "int16":
    case "bool":
      return dtype;
    default:
      throw new XybridError("tensor_invalid", `Output tensor '${name}' has unsupported dtype ${dtype}`);
  }
}
