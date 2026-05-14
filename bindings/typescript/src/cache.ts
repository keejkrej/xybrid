import { XybridError } from "./errors.js";
import { cacheKeyDigest } from "./hash.js";
import type { CacheMode } from "./types.js";

export interface ArtifactCache {
  readonly kind: "opfs" | "indexeddb" | "memory" | "none";
  get(key: string): Promise<ArrayBuffer | null>;
  put(key: string, value: ArrayBuffer): Promise<void>;
  delete(key: string): Promise<void>;
}

export class MemoryArtifactCache implements ArtifactCache {
  readonly kind = "memory";
  private readonly entries = new Map<string, ArrayBuffer>();

  async get(key: string): Promise<ArrayBuffer | null> {
    const value = this.entries.get(key);
    return value === undefined ? null : value.slice(0);
  }

  async put(key: string, value: ArrayBuffer): Promise<void> {
    this.entries.set(key, value.slice(0));
  }

  async delete(key: string): Promise<void> {
    this.entries.delete(key);
  }
}

export class NoArtifactCache implements ArtifactCache {
  readonly kind = "none";

  async get(_key: string): Promise<ArrayBuffer | null> {
    return null;
  }

  async put(_key: string, _value: ArrayBuffer): Promise<void> {
    return undefined;
  }

  async delete(_key: string): Promise<void> {
    return undefined;
  }
}

export class OpfsArtifactCache implements ArtifactCache {
  readonly kind = "opfs";

  private constructor(private readonly root: FileSystemDirectoryHandle) {}

  static async create(namespace = "xybrid-sdk"): Promise<OpfsArtifactCache> {
    const storage = (globalThis.navigator as Navigator & {
      storage?: { getDirectory?: () => Promise<FileSystemDirectoryHandle> };
    } | undefined)?.storage;
    if (!storage?.getDirectory) {
      throw new XybridError("cache_failed", "OPFS is not available");
    }
    const root = await storage.getDirectory();
    const directory = await root.getDirectoryHandle(namespace, { create: true });
    return new OpfsArtifactCache(directory);
  }

  async get(key: string): Promise<ArrayBuffer | null> {
    try {
      const handle = await this.root.getFileHandle(await cacheKeyDigest(key));
      return await (await handle.getFile()).arrayBuffer();
    } catch (error) {
      if (isNotFoundError(error)) {
        return null;
      }
      throw new XybridError("cache_failed", "Failed to read OPFS cache entry", { cause: error });
    }
  }

  async put(key: string, value: ArrayBuffer): Promise<void> {
    try {
      const handle = await this.root.getFileHandle(await cacheKeyDigest(key), { create: true });
      const writable = await handle.createWritable();
      await writable.write(value);
      await writable.close();
    } catch (error) {
      throw new XybridError("cache_failed", "Failed to write OPFS cache entry", { cause: error });
    }
  }

  async delete(key: string): Promise<void> {
    try {
      await this.root.removeEntry(await cacheKeyDigest(key));
    } catch (error) {
      if (!isNotFoundError(error)) {
        throw new XybridError("cache_failed", "Failed to delete OPFS cache entry", { cause: error });
      }
    }
  }
}

export class IndexedDbArtifactCache implements ArtifactCache {
  readonly kind = "indexeddb";

  private constructor(
    private readonly db: IDBDatabase,
    private readonly storeName: string,
  ) {}

  static async create(dbName = "xybrid-sdk", storeName = "artifacts"): Promise<IndexedDbArtifactCache> {
    const indexedDb = globalThis.indexedDB;
    if (!indexedDb) {
      throw new XybridError("cache_failed", "IndexedDB is not available");
    }

    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDb.open(dbName, 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore(storeName);
      };
      request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
      request.onsuccess = () => resolve(request.result);
    });

    return new IndexedDbArtifactCache(db, storeName);
  }

  async get(key: string): Promise<ArrayBuffer | null> {
    return this.transaction("readonly", (store) => store.get(key), (value) =>
      value instanceof ArrayBuffer ? value.slice(0) : null,
    );
  }

  async put(key: string, value: ArrayBuffer): Promise<void> {
    await this.transaction("readwrite", (store) => store.put(value.slice(0), key), () => undefined);
  }

  async delete(key: string): Promise<void> {
    await this.transaction("readwrite", (store) => store.delete(key), () => undefined);
  }

  private async transaction<T>(
    mode: IDBTransactionMode,
    createRequest: (store: IDBObjectStore) => IDBRequest,
    read: (value: unknown) => T,
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, mode);
      const request = createRequest(tx.objectStore(this.storeName));
      request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
      request.onsuccess = () => resolve(read(request.result));
      tx.onerror = () => reject(tx.error ?? new Error("IndexedDB transaction failed"));
    }).catch((error: unknown) => {
      throw new XybridError("cache_failed", "IndexedDB cache operation failed", { cause: error });
    });
  }
}

export async function createArtifactCache(mode: CacheMode): Promise<ArtifactCache> {
  if (mode === "none") {
    return new NoArtifactCache();
  }
  if (mode === "memory") {
    return new MemoryArtifactCache();
  }

  try {
    return await OpfsArtifactCache.create();
  } catch {
    try {
      return await IndexedDbArtifactCache.create();
    } catch {
      return new MemoryArtifactCache();
    }
  }
}

function isNotFoundError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}
