export type XybridErrorCode =
  | "aborted"
  | "backend_unavailable"
  | "cache_failed"
  | "fetch_failed"
  | "hash_mismatch"
  | "manifest_invalid"
  | "runtime_error"
  | "unsupported_step"
  | "tensor_invalid";

export class XybridError extends Error {
  readonly code: XybridErrorCode;

  constructor(code: XybridErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "XybridError";
    this.code = code;
  }
}

export function abortError(message = "Operation aborted"): XybridError {
  return new XybridError("aborted", message);
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}
