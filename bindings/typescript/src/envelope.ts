export type XybridEnvelopeKind = "Audio" | "Text" | "Embedding";

export class XybridEnvelope {
  private constructor(
    readonly kind: XybridEnvelopeKind,
    readonly value: Uint8Array | string | Float32Array,
    readonly metadata: Readonly<Record<string, string>> = {},
  ) {}

  static text(text: string, metadata: Readonly<Record<string, string>> = {}): XybridEnvelope {
    return new XybridEnvelope("Text", text, metadata);
  }

  static audio(bytes: Uint8Array | ArrayBuffer, metadata: Readonly<Record<string, string>> = {}): XybridEnvelope {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    return new XybridEnvelope("Audio", data, metadata);
  }

  static embedding(values: Float32Array | readonly number[], metadata: Readonly<Record<string, string>> = {}): XybridEnvelope {
    const data = values instanceof Float32Array ? values : new Float32Array(values);
    return new XybridEnvelope("Embedding", data, metadata);
  }

  text(): string | undefined {
    return this.kind === "Text" ? (this.value as string) : undefined;
  }

  audio(): Uint8Array | undefined {
    return this.kind === "Audio" ? (this.value as Uint8Array) : undefined;
  }

  embedding(): Float32Array | undefined {
    return this.kind === "Embedding" ? (this.value as Float32Array) : undefined;
  }
}

export type XybridRunInput = XybridEnvelope | string | Uint8Array | ArrayBuffer | Float32Array | readonly number[];

export function envelopeFromInput(input: XybridRunInput): XybridEnvelope {
  if (input instanceof XybridEnvelope) {
    return input;
  }
  if (typeof input === "string") {
    return XybridEnvelope.text(input);
  }
  if (input instanceof Uint8Array || input instanceof ArrayBuffer) {
    return XybridEnvelope.audio(input);
  }
  return XybridEnvelope.embedding(input);
}
