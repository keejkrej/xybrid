import NativeXybrid from "./specs/NativeXybrid.js";
import { base64ToBytes, bytesToBase64 } from "./base64.js";
import { toXybridError, XybridError } from "./errors.js";
export { XybridError };
export class Envelope {
    native;
    constructor(native) {
        this.native = native;
    }
    static text(text, options = {}) {
        return new Envelope({
            kind: "text",
            text,
            voiceId: options.voiceId,
            speed: options.speed,
        });
    }
    static audio(bytes, options = {}) {
        const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
        return new Envelope({
            kind: "audio",
            audioBase64: bytesToBase64(data),
            sampleRate: options.sampleRate ?? 16000,
            channels: options.channels ?? 1,
        });
    }
    static embedding(data) {
        const values = data instanceof Float32Array ? Array.from(data) : Array.from(data);
        return new Envelope({ kind: "embedding", embedding: values });
    }
}
export class XybridModelRef {
    source;
    constructor(source) {
        this.source = source;
    }
    async load(options = {}) {
        const requestId = createRequestId();
        let progressSubscription;
        let errorSubscription;
        const loadError = new Promise((_, reject) => {
            progressSubscription = NativeXybrid.xybridLoadProgress((event) => {
                if (event.requestId === requestId) {
                    options.onProgress?.(event);
                }
            });
            errorSubscription = NativeXybrid.xybridLoadError((event) => {
                if (event.requestId === requestId) {
                    reject(new XybridError("load_error", event.message));
                }
            });
        });
        const load = NativeXybrid.loadModel(this.source, requestId);
        try {
            const loaded = await Promise.race([load, loadError]);
            return new XybridModel(loaded.handle);
        }
        catch (error) {
            throw toXybridError(error);
        }
        finally {
            progressSubscription?.remove();
            errorSubscription?.remove();
        }
    }
}
export class XybridModel {
    handle;
    constructor(handle) {
        this.handle = handle;
    }
    async run(input, config) {
        try {
            const native = await NativeXybrid.runModel(this.handle, envelopeFromInput(input).native, config);
            return resultFromNative(native);
        }
        catch (error) {
            throw toXybridError(error);
        }
    }
    async voices() {
        try {
            return await NativeXybrid.voices(this.handle);
        }
        catch (error) {
            throw toXybridError(error);
        }
    }
    async defaultVoiceId() {
        try {
            return await NativeXybrid.defaultVoiceId(this.handle);
        }
        catch (error) {
            throw toXybridError(error);
        }
    }
    dispose() {
        NativeXybrid.disposeModel(this.handle);
    }
}
export class Xybrid {
    static async init(options) {
        try {
            await NativeXybrid.initialize(options);
        }
        catch (error) {
            throw toXybridError(error);
        }
    }
    static setApiKey(apiKey) {
        NativeXybrid.setApiKey(apiKey);
    }
    static model(source) {
        return new XybridModelRef(typeof source === "string" ? { kind: "registry", value: source } : source);
    }
    static async isModelCached(modelId) {
        try {
            return await NativeXybrid.isModelCached(modelId);
        }
        catch (error) {
            throw toXybridError(error);
        }
    }
}
export function envelopeFromInput(input) {
    if (input instanceof Envelope) {
        return input;
    }
    if (typeof input === "string") {
        return Envelope.text(input);
    }
    if (input instanceof Uint8Array || input instanceof ArrayBuffer) {
        return Envelope.audio(input);
    }
    return Envelope.embedding(input);
}
function resultFromNative(native) {
    return {
        success: native.success,
        text: native.text,
        audioBytes: native.audioBase64 ? base64ToBytes(native.audioBase64) : undefined,
        embedding: native.embedding,
        latencyMs: native.latencyMs,
    };
}
function createRequestId() {
    const random = Math.random().toString(36).slice(2);
    return `rn-${Date.now().toString(36)}-${random}`;
}
//# sourceMappingURL=index.js.map