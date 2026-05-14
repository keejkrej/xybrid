export class XybridError extends Error {
  readonly code: string;
  readonly nativeStack: string | undefined;

  constructor(code: string, message: string, nativeStack?: string) {
    super(message);
    this.name = "XybridError";
    this.code = code;
    this.nativeStack = nativeStack;
  }
}

export function toXybridError(error: unknown): XybridError {
  if (error instanceof XybridError) {
    return error;
  }
  if (error instanceof Error) {
    const code = "code" in error && typeof error.code === "string" ? error.code : "native_error";
    return new XybridError(code, error.message, error.stack);
  }
  return new XybridError("native_error", String(error));
}
