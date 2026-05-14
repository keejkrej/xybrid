import { describe, expect, it } from "vitest";
import { loadArtifacts } from "../src/artifacts.js";
import { MemoryArtifactCache } from "../src/cache.js";
import { XybridError } from "../src/errors.js";
import { sha256Hex } from "../src/hash.js";
import type { FetchLike, LoadProgressEvent, XybridWebManifest } from "../src/types.js";

describe("artifact loading", () => {
  it("orders progress events and reuses verified cache entries", async () => {
    const modelBytes = new Uint8Array([1, 2, 3, 4]).buffer;
    const manifest = await manifestFor(modelBytes);
    const cache = new MemoryArtifactCache();
    let fetchCount = 0;
    const fetchLike: FetchLike = async () => {
      fetchCount += 1;
      return responseFromBytes(modelBytes);
    };
    const firstEvents: LoadProgressEvent[] = [];

    await loadArtifacts({
      manifest,
      manifestUrl: "https://cdn.example.com/models/tiny/xybrid.web.json",
      cache,
      fetch: fetchLike,
      onProgress: (event) => firstEvents.push(event),
    });

    expect(firstEvents.map((event) => event.type)).toEqual(["download-start", "download-progress", "verify"]);
    expect(fetchCount).toBe(1);

    const secondEvents: LoadProgressEvent[] = [];
    await loadArtifacts({
      manifest,
      manifestUrl: "https://cdn.example.com/models/tiny/xybrid.web.json",
      cache,
      fetch: fetchLike,
      onProgress: (event) => secondEvents.push(event),
    });

    expect(secondEvents.map((event) => event.type)).toEqual(["cache-hit"]);
    expect(fetchCount).toBe(1);
  });

  it("aborts while streaming a download", async () => {
    const modelBytes = new Uint8Array([1, 2, 3, 4]).buffer;
    const manifest = await manifestFor(modelBytes, false);
    const abortController = new AbortController();

    await expect(
      loadArtifacts({
        manifest,
        manifestUrl: "https://cdn.example.com/models/tiny/xybrid.web.json",
        cache: new MemoryArtifactCache(),
        fetch: async () => streamedResponse([new Uint8Array([1, 2]), new Uint8Array([3, 4])]),
        signal: abortController.signal,
        onProgress: () => abortController.abort(),
      }),
    ).rejects.toMatchObject({ code: "aborted" } satisfies Partial<XybridError>);
  });
});

async function manifestFor(modelBytes: ArrayBuffer, includeHash = true): Promise<XybridWebManifest> {
  return {
    modelId: "xybrid/tiny",
    version: "1.0.0",
    executionTemplate: { type: "onnx", modelFile: "model.onnx" },
    files: [
      {
        path: "model.onnx",
        sizeBytes: modelBytes.byteLength,
        ...(includeHash ? { sha256: await sha256Hex(modelBytes) } : {}),
      },
    ],
    inputs: [{ name: "image", dtype: "float32", shape: [1, 1, 2, 2], layout: "NCHW" }],
    outputs: [{ name: "mask", dtype: "float32", shape: [1, 1, 2, 2], layout: "NCHW" }],
  };
}

function responseFromBytes(bytes: ArrayBuffer): Response {
  return new Response(bytes, {
    status: 200,
    headers: { "content-length": String(bytes.byteLength) },
  });
}

function streamedResponse(chunks: Uint8Array[]): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      pull(controller) {
        const chunk = chunks.shift();
        if (chunk === undefined) {
          controller.close();
        } else {
          controller.enqueue(chunk);
        }
      },
    }),
    { status: 200 },
  );
}
