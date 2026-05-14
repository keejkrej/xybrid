import { describe, expect, it } from "vitest";
import { ArtifactResolver } from "../src/artifact-resolver.js";
import { applyPostprocessingStep } from "../src/postprocessing.js";
import { applyPreprocessingStep } from "../src/preprocessing.js";
import { XybridTensor } from "../src/tensor.js";
import { loadVoiceEmbedding } from "../src/voice.js";

describe("preprocessing steps", () => {
  it("normalizes and reshapes tensors", () => {
    const normalized = applyPreprocessingStep(
      { type: "Normalize", mean: [1], std: [2] },
      { kind: "tensor", tensor: XybridTensor.fromFloat32([1, 3, 5, 7], { shape: [1, 1, 2, 2] }) },
      new ArtifactResolver(new Map()),
    );

    expect(normalized.kind).toBe("tensor");
    if (normalized.kind === "tensor") {
      expect([...(normalized.tensor.data as Float32Array)]).toEqual([0, 1, 2, 3]);
    }

    const reshaped = applyPreprocessingStep(
      { type: "Reshape", shape: [2, 2] },
      normalized,
      new ArtifactResolver(new Map()),
    );
    expect(reshaped.kind === "tensor" ? reshaped.tensor.shape : []).toEqual([2, 2]);
  });

  it("phonemizes text with bundled tokens and dictionaries", () => {
    const artifacts = new ArtifactResolver(
      new Map<string, ArrayBuffer>([
        ["tokens.txt", text("h 1\ni 2\n  3\n0 4\n")],
        ["misaki/us_gold.json", text(JSON.stringify({ hi: "hi" }))],
        ["misaki/us_silver.json", text("{}")],
      ]),
    );

    const result = applyPreprocessingStep(
      { type: "Phonemize", tokens_file: "tokens.txt", normalize_text: true },
      { kind: "text", text: "Hi" },
      artifacts,
    );

    expect(result.kind).toBe("phoneme-ids");
    if (result.kind === "phoneme-ids") {
      expect(result.ids).toEqual([0n, 1n, 2n, 0n]);
      expect(result.phonemes).toBe("hi");
    }
  });
});

describe("postprocessing steps", () => {
  it("applies softmax and top-k", async () => {
    const softmax = await applyPostprocessingStep(
      { type: "Softmax" },
      { kind: "tensor-map", outputs: { logits: XybridTensor.fromFloat32([1, 2], { shape: [2] }) } },
      new ArtifactResolver(new Map()),
    );

    expect(softmax.kind).toBe("tensor-map");
    if (softmax.kind === "tensor-map") {
      const values = softmax.outputs.logits?.data as Float32Array;
      expect(values[1]).toBeGreaterThan(values[0] ?? 0);
    }

    const topk = await applyPostprocessingStep({ type: "TopK", k: 1 }, softmax, new ArtifactResolver(new Map()));
    expect(topk.kind).toBe("tensor-map");
    if (topk.kind === "tensor-map") {
      expect([...(topk.outputs.topk?.data as Float32Array)]).toEqual([1, expect.any(Number)]);
    }
  });
});

describe("voice loading", () => {
  it("loads BinaryF32_256 voice embeddings", () => {
    const bytes = new Uint8Array(256 * 4 * 2);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < 512; i += 1) {
      view.setFloat32(i * 4, i, true);
    }
    const embedding = loadVoiceEmbedding(
      {
        format: { type: "Embedded", file: "voices.bin", loader: "BinaryF32_256" },
        default: "b",
        catalog: [
          { id: "a", name: "A", index: 0 },
          { id: "b", name: "B", index: 1 },
        ],
      },
      new ArtifactResolver(new Map([["voices.bin", bytes.buffer]])),
    );

    expect(embedding[0]).toBe(256);
    expect(embedding.length).toBe(256);
  });
});

function text(value: string): ArrayBuffer {
  return new TextEncoder().encode(value).buffer as ArrayBuffer;
}
