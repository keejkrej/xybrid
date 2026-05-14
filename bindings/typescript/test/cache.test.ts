import { describe, expect, it } from "vitest";
import { createArtifactCache, MemoryArtifactCache, NoArtifactCache } from "../src/cache.js";

describe("artifact caches", () => {
  it("records memory cache hits and misses", async () => {
    const cache = new MemoryArtifactCache();

    expect(await cache.get("missing")).toBeNull();
    await cache.put("model", new Uint8Array([1, 2, 3]).buffer);

    expect([...new Uint8Array((await cache.get("model")) ?? new ArrayBuffer(0))]).toEqual([1, 2, 3]);
  });

  it("does not persist entries when disabled", async () => {
    const cache = new NoArtifactCache();

    await cache.put("model", new Uint8Array([1]).buffer);

    expect(await cache.get("model")).toBeNull();
  });

  it("falls back to memory when persistent browser stores are unavailable", async () => {
    const cache = await createArtifactCache("persistent");

    expect(cache.kind).toBe("memory");
  });
});
