import { describe, expect, it } from "vitest";
import { resolveArtifactUrl, validateManifest } from "../src/manifest.js";
import { XybridError } from "../src/errors.js";

describe("manifest validation", () => {
  it("accepts the browser ONNX manifest shape", () => {
    const manifest = validateManifest({
      model_id: "xybrid/tiny",
      version: "1.0.0",
      execution_template: { type: "onnx", model_file: "model.onnx" },
      artifacts: [{ path: "model.onnx", sizeBytes: 4 }],
      files: ["model.onnx"],
      inputs: [{ name: "image", dtype: "float32", shape: [1, 1, 2, 2], layout: "NCHW" }],
      outputs: [{ name: "mask", dtype: "float32", shape: [1, 1, 2, 2], layout: "NCHW" }],
      preprocessing: [{ type: "Normalize", mean: [0], std: [1] }],
      postprocessing: [{ type: "Softmax" }],
    });

    expect(manifest.modelId).toBe("xybrid/tiny");
    expect(manifest.model_id).toBe("xybrid/tiny");
    expect(manifest.executionTemplate.modelFile).toBe("model.onnx");
    expect(manifest.preprocessing[0]?.type).toBe("Normalize");
  });

  it("accepts deprecated camelCase manifest aliases", () => {
    const manifest = validateManifest({
      modelId: "xybrid/tiny",
      version: "1.0.0",
      executionTemplate: { type: "onnx", modelFile: "model.onnx" },
      files: [{ path: "model.onnx", sizeBytes: 4 }],
      inputs: [],
      outputs: [],
    });

    expect(manifest.model_id).toBe("xybrid/tiny");
    expect(manifest.artifacts[0]?.path).toBe("model.onnx");
  });

  it("rejects manifests whose model file is not listed as an artifact", () => {
    expect(() =>
      validateManifest({
        modelId: "xybrid/tiny",
        version: "1.0.0",
        executionTemplate: { type: "onnx", modelFile: "model.onnx" },
        files: [{ path: "weights.bin" }],
        inputs: [],
        outputs: [],
      }),
    ).toThrow(XybridError);
  });
});

describe("artifact URL resolution", () => {
  it("resolves artifact paths relative to the manifest URL", () => {
    expect(resolveArtifactUrl("https://cdn.example.com/models/unet/xybrid.web.json", "artifacts/model.onnx")).toBe(
      "https://cdn.example.com/models/unet/artifacts/model.onnx",
    );
  });

  it("resolves artifact paths when the manifest URL is root-relative", () => {
    expect(resolveArtifactUrl("/models/unet/xybrid.web.json", "artifacts/model.onnx")).toBe(
      "http://localhost/models/unet/artifacts/model.onnx",
    );
  });
});
