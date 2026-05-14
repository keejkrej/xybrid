import { type GenerationConfig, type LoadProgressEvent, type ModelSource, type NativeEnvelope, type VoiceInfo } from "./specs/NativeXybrid.js";
import { XybridError } from "./errors.js";
export type { GenerationConfig, LoadProgressEvent, ModelSource, VoiceInfo };
export { XybridError };
export type TextEnvelopeOptions = {
    voiceId?: string;
    speed?: number;
};
export type AudioEnvelopeOptions = {
    sampleRate?: number;
    channels?: number;
};
export declare class Envelope {
    readonly native: NativeEnvelope;
    private constructor();
    static text(text: string, options?: TextEnvelopeOptions): Envelope;
    static audio(bytes: Uint8Array | ArrayBuffer, options?: AudioEnvelopeOptions): Envelope;
    static embedding(data: Float32Array | readonly number[]): Envelope;
}
export type XybridRunInput = Envelope | string | Uint8Array | ArrayBuffer | Float32Array | readonly number[];
export type XybridResult = {
    success: boolean;
    text?: string | undefined;
    audioBytes?: Uint8Array | undefined;
    embedding?: number[] | undefined;
    latencyMs: number;
};
export type LoadModelOptions = {
    onProgress?: (event: LoadProgressEvent) => void;
};
export declare class XybridModelRef {
    readonly source: ModelSource;
    constructor(source: ModelSource);
    load(options?: LoadModelOptions): Promise<XybridModel>;
}
export declare class XybridModel {
    readonly handle: number;
    constructor(handle: number);
    run(input: XybridRunInput, config?: GenerationConfig): Promise<XybridResult>;
    voices(): Promise<VoiceInfo[]>;
    defaultVoiceId(): Promise<string | null>;
    dispose(): void;
}
export declare class Xybrid {
    static init(options?: {
        cacheDir?: string;
    }): Promise<void>;
    static setApiKey(apiKey: string): void;
    static model(source: string | ModelSource): XybridModelRef;
    static isModelCached(modelId: string): Promise<boolean>;
}
export declare function envelopeFromInput(input: XybridRunInput): Envelope;
//# sourceMappingURL=index.d.ts.map