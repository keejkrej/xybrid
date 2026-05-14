export declare class XybridError extends Error {
    readonly code: string;
    readonly nativeStack: string | undefined;
    constructor(code: string, message: string, nativeStack?: string);
}
export declare function toXybridError(error: unknown): XybridError;
//# sourceMappingURL=errors.d.ts.map