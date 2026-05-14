import { describe, expect, it } from "vitest";
import { XybridError } from "../src/errors.js";
import { XybridTensor } from "../src/tensor.js";

describe("XybridTensor", () => {
  it("validates shape element counts at construction", () => {
    expect(() => XybridTensor.fromFloat32([1, 2, 3], { shape: [1, 2] })).toThrow(XybridError);
  });

  it("validates dtype and concrete manifest dimensions", () => {
    const tensor = XybridTensor.fromFloat32([1, 2, 3, 4], { shape: [1, 1, 2, 2], layout: "NCHW" });

    expect(() =>
      tensor.validateAgainst({ name: "image", dtype: "float32", shape: [1, "C", 2, 2], layout: "NCHW" }),
    ).not.toThrow();
    expect(() =>
      tensor.validateAgainst({ name: "image", dtype: "int32", shape: [1, 1, 2, 2], layout: "NCHW" }),
    ).toThrow(XybridError);
    expect(() =>
      tensor.validateAgainst({ name: "image", dtype: "float32", shape: [1, 1, 4, 4], layout: "NCHW" }),
    ).toThrow(XybridError);
  });
});
