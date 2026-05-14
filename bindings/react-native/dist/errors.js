export class XybridError extends Error {
    code;
    nativeStack;
    constructor(code, message, nativeStack) {
        super(message);
        this.name = "XybridError";
        this.code = code;
        this.nativeStack = nativeStack;
    }
}
export function toXybridError(error) {
    if (error instanceof XybridError) {
        return error;
    }
    if (error instanceof Error) {
        const code = "code" in error && typeof error.code === "string" ? error.code : "native_error";
        return new XybridError(code, error.message, error.stack);
    }
    return new XybridError("native_error", String(error));
}
//# sourceMappingURL=errors.js.map