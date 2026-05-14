import type { CodegenTypes, TurboModule } from "react-native";
import { TurboModuleRegistry } from "react-native";

export type ModelSource = {
  kind: "registry" | "bundle" | "directory" | "huggingface";
  value: string;
};

export type NativeEnvelope = {
  kind: "text" | "audio" | "embedding";
  text?: string | undefined;
  audioBase64?: string | undefined;
  sampleRate?: number | undefined;
  channels?: number | undefined;
  embedding?: number[] | undefined;
  voiceId?: string | undefined;
  speed?: number | undefined;
};

export type GenerationConfig = {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  minP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
};

export type XybridResult = {
  success: boolean;
  text?: string | undefined;
  audioBase64?: string | undefined;
  embedding?: number[] | undefined;
  latencyMs: number;
};

export type VoiceInfo = {
  id: string;
  name: string;
  gender?: string;
  language?: string;
  style?: string;
};

export type ModelHandle = {
  handle: number;
};

export type LoadProgressEvent = {
  requestId: string;
  modelId?: string;
  progress: number;
  phase: string;
};

export type LoadErrorEvent = {
  requestId: string;
  message: string;
};

export interface Spec extends TurboModule {
  initialize(options?: { cacheDir?: string }): Promise<void>;
  setApiKey(apiKey: string): void;
  loadModel(source: ModelSource, requestId: string): Promise<ModelHandle>;
  runModel(handle: number, envelope: NativeEnvelope, config?: GenerationConfig): Promise<XybridResult>;
  voices(handle: number): Promise<VoiceInfo[]>;
  defaultVoiceId(handle: number): Promise<string | null>;
  disposeModel(handle: number): void;
  isModelCached(modelId: string): Promise<boolean>;

  readonly xybridLoadProgress: CodegenTypes.EventEmitter<LoadProgressEvent>;
  readonly xybridLoadError: CodegenTypes.EventEmitter<LoadErrorEvent>;
}

export default TurboModuleRegistry.getEnforcing<Spec>("XybridReactNative");
