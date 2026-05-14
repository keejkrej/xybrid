const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
export function bytesToBase64(bytes) {
    let output = "";
    let i = 0;
    for (; i + 2 < bytes.length; i += 3) {
        const triple = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
        output += alphabet[(triple >> 18) & 63];
        output += alphabet[(triple >> 12) & 63];
        output += alphabet[(triple >> 6) & 63];
        output += alphabet[triple & 63];
    }
    if (i < bytes.length) {
        const first = bytes[i];
        const second = i + 1 < bytes.length ? bytes[i + 1] : 0;
        const triple = (first << 16) | (second << 8);
        output += alphabet[(triple >> 18) & 63];
        output += alphabet[(triple >> 12) & 63];
        output += i + 1 < bytes.length ? alphabet[(triple >> 6) & 63] : "=";
        output += "=";
    }
    return output;
}
export function base64ToBytes(value) {
    const normalized = value.replace(/\s/g, "");
    if (normalized.length === 0) {
        return new Uint8Array();
    }
    if (normalized.length % 4 !== 0) {
        throw new Error("Invalid base64 length");
    }
    const padding = normalized.endsWith("==") ? 2 : normalized.endsWith("=") ? 1 : 0;
    const output = new Uint8Array((normalized.length / 4) * 3 - padding);
    let out = 0;
    for (let i = 0; i < normalized.length; i += 4) {
        const a = decodeChar(normalized[i]);
        const b = decodeChar(normalized[i + 1]);
        const c = normalized[i + 2] === "=" ? 0 : decodeChar(normalized[i + 2]);
        const d = normalized[i + 3] === "=" ? 0 : decodeChar(normalized[i + 3]);
        const triple = (a << 18) | (b << 12) | (c << 6) | d;
        if (out < output.length)
            output[out++] = (triple >> 16) & 255;
        if (out < output.length)
            output[out++] = (triple >> 8) & 255;
        if (out < output.length)
            output[out++] = triple & 255;
    }
    return output;
}
function decodeChar(char) {
    const index = alphabet.indexOf(char);
    if (index < 0) {
        throw new Error(`Invalid base64 character: ${char}`);
    }
    return index;
}
//# sourceMappingURL=base64.js.map