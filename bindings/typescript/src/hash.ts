import { XybridError } from "./errors.js";

export async function sha256Hex(data: ArrayBuffer | Uint8Array | string): Promise<string> {
  const bytes =
    typeof data === "string"
      ? new TextEncoder().encode(data)
      : data instanceof Uint8Array
        ? data
        : new Uint8Array(data);
  const cryptoApi = globalThis.crypto;
  if (!cryptoApi?.subtle) {
    throw new XybridError("hash_mismatch", "SHA-256 verification requires crypto.subtle");
  }
  const buffer = new Uint8Array(bytes.byteLength);
  buffer.set(bytes);
  const digest = await cryptoApi.subtle.digest("SHA-256", buffer.buffer);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function cacheKeyDigest(key: string): Promise<string> {
  try {
    return await sha256Hex(key);
  } catch {
    return encodeURIComponent(key).replaceAll("%", "_").slice(0, 180);
  }
}
