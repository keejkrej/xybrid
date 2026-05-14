import { XybridError } from "./errors.js";

export class ArtifactResolver {
  constructor(private readonly artifacts: ReadonlyMap<string, ArrayBuffer>) {}

  has(path: string): boolean {
    return this.artifacts.has(path);
  }

  bytes(path: string): ArrayBuffer {
    const data = this.artifacts.get(path);
    if (data === undefined) {
      throw new XybridError("manifest_invalid", `Artifact '${path}' was not loaded`);
    }
    return data;
  }

  text(path: string): string {
    return new TextDecoder().decode(this.bytes(path));
  }
}
