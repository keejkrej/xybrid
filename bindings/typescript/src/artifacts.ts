import { abortError, isAbortError, XybridError } from "./errors.js";
import { sha256Hex } from "./hash.js";
import { resolveArtifactUrl } from "./manifest.js";
import type { ArtifactCache } from "./cache.js";
import type { FetchLike, LoadProgressEvent, XybridWebManifest, XybridWebManifestFile } from "./types.js";

export interface LoadArtifactsOptions {
  readonly manifest: XybridWebManifest;
  readonly manifestUrl: string;
  readonly cache: ArtifactCache;
  readonly fetch: FetchLike;
  readonly onProgress?: ((event: LoadProgressEvent) => void) | undefined;
  readonly signal?: AbortSignal | undefined;
}

export async function loadArtifacts(options: LoadArtifactsOptions): Promise<Map<string, ArrayBuffer>> {
  const totalFiles = options.manifest.files.length;
  const totalBytes = sumKnownBytes(options.manifest.files);
  const artifacts = new Map<string, ArrayBuffer>();
  let loadedFiles = 0;
  let loadedBytes = 0;

  for (const file of options.manifest.files) {
    throwIfAborted(options.signal);
    const key = cacheKey(options.manifest, file);
    const cached = await readVerifiedCache(options.cache, key, file);
    if (cached !== null) {
      artifacts.set(file.path, cached);
      loadedFiles += 1;
      loadedBytes += file.sizeBytes ?? cached.byteLength;
      options.onProgress?.({
        type: "cache-hit",
        path: file.path,
        loadedFiles,
        totalFiles,
        loadedBytes,
        ...(totalBytes === undefined ? {} : { totalBytes }),
      });
      continue;
    }

    options.onProgress?.({
      type: "download-start",
      path: file.path,
      loadedFiles,
      totalFiles,
      loadedBytes,
      ...(totalBytes === undefined ? {} : { totalBytes }),
    });

    const downloaded = await downloadFile({
      file,
      url: resolveArtifactUrl(options.manifestUrl, file.path),
      fetch: options.fetch,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      onProgress: (fileLoadedBytes, fileTotalBytes) => {
        options.onProgress?.({
          type: "download-progress",
          path: file.path,
          loadedFiles,
          totalFiles,
          loadedBytes: loadedBytes + fileLoadedBytes,
          ...(totalBytes === undefined ? {} : { totalBytes }),
          fileLoadedBytes,
          ...(fileTotalBytes === undefined ? {} : { fileTotalBytes }),
        });
      },
    });

    options.onProgress?.({
      type: "verify",
      path: file.path,
      loadedFiles,
      totalFiles,
      loadedBytes,
      ...(totalBytes === undefined ? {} : { totalBytes }),
    });
    await verifyArtifact(file, downloaded);
    await options.cache.put(key, downloaded);

    artifacts.set(file.path, downloaded);
    loadedFiles += 1;
    loadedBytes += file.sizeBytes ?? downloaded.byteLength;
  }

  return artifacts;
}

export function cacheKey(manifest: XybridWebManifest, file: XybridWebManifestFile): string {
  return `${manifest.modelId}@${manifest.version}/${file.path}/${file.sha256 ?? file.sizeBytes ?? "unverified"}`;
}

export async function verifyArtifact(file: XybridWebManifestFile, data: ArrayBuffer): Promise<void> {
  if (file.sizeBytes !== undefined && data.byteLength !== file.sizeBytes) {
    throw new XybridError(
      "hash_mismatch",
      `Artifact '${file.path}' expected ${file.sizeBytes} bytes, received ${data.byteLength}`,
    );
  }
  if (file.sha256 !== undefined) {
    const actual = await sha256Hex(data);
    if (actual.toLowerCase() !== file.sha256.toLowerCase()) {
      throw new XybridError("hash_mismatch", `Artifact '${file.path}' failed SHA-256 verification`);
    }
  }
}

async function readVerifiedCache(
  cache: ArtifactCache,
  key: string,
  file: XybridWebManifestFile,
): Promise<ArrayBuffer | null> {
  const cached = await cache.get(key);
  if (cached === null) {
    return null;
  }
  try {
    await verifyArtifact(file, cached);
    return cached;
  } catch {
    await cache.delete(key);
    return null;
  }
}

async function downloadFile(options: {
  readonly file: XybridWebManifestFile;
  readonly url: string;
  readonly fetch: FetchLike;
  readonly signal?: AbortSignal | undefined;
  readonly onProgress: (loadedBytes: number, totalBytes?: number) => void;
}): Promise<ArrayBuffer> {
  throwIfAborted(options.signal);
  let response: Response;
  try {
    response = await options.fetch(
      options.url,
      options.signal === undefined ? {} : { signal: options.signal },
    );
  } catch (error) {
    if (isAbortError(error) || options.signal?.aborted) {
      throw abortError(`Download aborted for '${options.file.path}'`);
    }
    throw new XybridError("fetch_failed", `Failed to fetch '${options.file.path}'`, { cause: error });
  }
  if (!response.ok) {
    throw new XybridError("fetch_failed", `Failed to fetch '${options.file.path}': HTTP ${response.status}`);
  }

  const contentLength = response.headers.get("content-length");
  const parsedTotalBytes = contentLength === null ? options.file.sizeBytes : Number(contentLength);
  const fileTotalBytes =
    parsedTotalBytes !== undefined && Number.isFinite(parsedTotalBytes) && parsedTotalBytes >= 0
      ? parsedTotalBytes
      : undefined;

  if (!response.body) {
    const data = await response.arrayBuffer();
    options.onProgress(data.byteLength, fileTotalBytes);
    return data;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;

  try {
    while (true) {
      throwIfAborted(options.signal);
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      chunks.push(value);
      loadedBytes += value.byteLength;
      options.onProgress(loadedBytes, fileTotalBytes);
    }
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    if (error instanceof XybridError || isAbortError(error) || options.signal?.aborted) {
      throw abortError(`Download aborted for '${options.file.path}'`);
    }
    throw error;
  }

  return mergeChunks(chunks, loadedBytes);
}

function mergeChunks(chunks: readonly Uint8Array[], totalLength: number): ArrayBuffer {
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output.buffer;
}

function sumKnownBytes(files: readonly XybridWebManifestFile[]): number | undefined {
  let total = 0;
  for (const file of files) {
    if (file.sizeBytes === undefined) {
      return undefined;
    }
    total += file.sizeBytes;
  }
  return total;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw abortError();
  }
}
