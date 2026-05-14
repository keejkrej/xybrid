import { XybridError } from "./errors.js";
import type { TensorDType, TensorLayout, TensorMetadata, TensorShape } from "./types.js";

export type XybridTensorData =
  | Float32Array
  | Float64Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Int8Array
  | Uint16Array
  | Int16Array;

export interface XybridTensorOptions {
  readonly shape: readonly number[];
  readonly layout?: TensorLayout | undefined;
}

export class XybridTensor {
  readonly data: XybridTensorData;
  readonly dtype: TensorDType;
  readonly shape: readonly number[];
  readonly layout: TensorLayout | undefined;

  private constructor(data: XybridTensorData, dtype: TensorDType, options: XybridTensorOptions) {
    validateShape(data.length, options.shape);
    this.data = data;
    this.dtype = dtype;
    this.shape = [...options.shape];
    this.layout = options.layout;
  }

  static fromFloat32(data: Float32Array | readonly number[], options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data instanceof Float32Array ? data : new Float32Array(data), "float32", options);
  }

  static fromFloat64(data: Float64Array | readonly number[], options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data instanceof Float64Array ? data : new Float64Array(data), "float64", options);
  }

  static fromInt32(data: Int32Array | readonly number[], options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data instanceof Int32Array ? data : new Int32Array(data), "int32", options);
  }

  static fromInt64(data: BigInt64Array | readonly bigint[], options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data instanceof BigInt64Array ? data : new BigInt64Array(data), "int64", options);
  }

  static fromUint8(data: Uint8Array | readonly number[], options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data instanceof Uint8Array ? data : new Uint8Array(data), "uint8", options);
  }

  static fromInt8(data: Int8Array | readonly number[], options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data instanceof Int8Array ? data : new Int8Array(data), "int8", options);
  }

  static fromUint16(data: Uint16Array | readonly number[], options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data instanceof Uint16Array ? data : new Uint16Array(data), "uint16", options);
  }

  static fromInt16(data: Int16Array | readonly number[], options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data instanceof Int16Array ? data : new Int16Array(data), "int16", options);
  }

  static fromBool(data: Uint8Array | readonly boolean[] | readonly number[], options: XybridTensorOptions): XybridTensor {
    const bytes =
      data instanceof Uint8Array
        ? data
        : new Uint8Array(data.map((value) => (value ? 1 : 0)));
    return new XybridTensor(bytes, "bool", options);
  }

  static fromData(data: XybridTensorData, dtype: TensorDType, options: XybridTensorOptions): XybridTensor {
    return new XybridTensor(data, dtype, options);
  }

  validateAgainst(metadata: TensorMetadata): void {
    if (this.dtype !== metadata.dtype) {
      throw new XybridError(
        "tensor_invalid",
        `Tensor '${metadata.name}' expected dtype ${metadata.dtype}, received ${this.dtype}`,
      );
    }
    validateConcreteShape(this.shape, metadata.shape, metadata.name);
  }
}

export function validateConcreteShape(actual: readonly number[], expected: TensorShape, name: string): void {
  if (actual.length !== expected.length) {
    throw new XybridError(
      "tensor_invalid",
      `Tensor '${name}' expected rank ${expected.length}, received rank ${actual.length}`,
    );
  }
  for (let index = 0; index < expected.length; index += 1) {
    const dim = expected[index];
    if (typeof dim === "number" && actual[index] !== dim) {
      throw new XybridError(
        "tensor_invalid",
        `Tensor '${name}' expected shape ${formatShape(expected)}, received ${formatShape(actual)}`,
      );
    }
  }
}

function validateShape(length: number, shape: readonly number[]): void {
  if (shape.length === 0 || shape.some((dim) => !Number.isInteger(dim) || dim < 0)) {
    throw new XybridError("tensor_invalid", "Tensor shape must contain non-negative integer dimensions");
  }
  const expectedLength = shape.reduce((product, dim) => product * dim, 1);
  if (length !== expectedLength) {
    throw new XybridError("tensor_invalid", `Tensor data has ${length} elements but shape requires ${expectedLength}`);
  }
}

function formatShape(shape: readonly (number | string | null)[]): string {
  return `[${shape.map((dim) => (dim === null ? "?" : String(dim))).join(", ")}]`;
}
