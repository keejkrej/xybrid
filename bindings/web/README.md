# @xybrid/web

Browser-first TypeScript SDK for running small Xybrid ONNX models fully in the tab with `onnxruntime-web`.

V1 uses ONNX Runtime Web as the execution backend. Xybrid owns the stable manifest, package loading, cache, and typed tensor API; model kernels still execute through ORT WASM or WebGPU.

## Install

```bash
pnpm add @xybrid/web onnxruntime-web
```

## Usage

```ts
import { Xybrid, XybridEnvelope, XybridTensor } from "@xybrid/web";

const xybrid = await Xybrid.create({
  backend: "ort-wasm",
  cache: "persistent",
});

const model = await xybrid.model("/models/unet/xybrid.web.json").load({
  onProgress: (event) => console.log(event),
  signal,
});

const result = await model.runTensor({
  inputs: {
    image: XybridTensor.fromFloat32(input, {
      shape: [1, 1, 256, 256],
      layout: "NCHW",
    }),
  },
});

const maskLogits = result.outputs.mask;
```

Models with Xybrid preprocessing/postprocessing metadata can use the native-style
envelope API:

```ts
const tts = await xybrid.model("/models/kokoro-82m/xybrid.web.json").load();
const speech = await tts.run(XybridEnvelope.text("Hello from Xybrid"), {
  voice: "af_heart",
  speed: 1.0,
});

const wavBytes = speech.unwrapAudio();
```

## Manifest

Web artifacts are described by `xybrid.web.json`:

```json
{
  "model_id": "xybrid/unet-tiny",
  "version": "1.0.0",
  "execution_template": {
    "type": "onnx",
    "model_file": "model.onnx"
  },
  "artifacts": [
    {
      "path": "model.onnx",
      "sizeBytes": 123456,
      "sha256": "..."
    }
  ],
  "inputs": [
    {
      "name": "image",
      "dtype": "float32",
      "shape": [1, 1, 256, 256],
      "layout": "NCHW"
    }
  ],
  "outputs": [
    {
      "name": "mask",
      "dtype": "float32",
      "shape": [1, 1, 256, 256],
      "layout": "NCHW"
    }
  ],
  "preprocessing": [
    { "type": "Normalize", "mean": [0.5], "std": [0.5] }
  ],
  "postprocessing": [
    { "type": "Softmax" }
  ],
  "files": ["model.onnx"]
}
```

Artifact paths resolve relative to the manifest URL. If `sha256` is present, the SDK verifies downloaded and cached bytes before creating the ORT session.

## Backends and Cache

Backends:

- `ort-wasm`: default, runs through ONNX Runtime Web WASM.
- `ort-webgpu`: requests ORT WebGPU when `navigator.gpu` exists and falls back to WASM otherwise.

Cache modes:

- `persistent`: OPFS when available, then IndexedDB, then memory.
- `memory`: in-memory cache for the page lifetime.
- `none`: no artifact cache.

`runTensor()` remains available for apps that want to provide tensors directly.
`run()` uses the manifest's `preprocessing`, `postprocessing`, and `voices`
metadata to match native Xybrid examples.

## Kokoro Web Demo

The repo's platform examples use `kokoro-82m` for the official text-to-speech
demo. The TypeScript package includes a vanilla Vite demo for the same flow:

```bash
pnpm install
pnpm demo
```

Serve a web-prepared Kokoro model at `/models/kokoro-82m/xybrid.web.json`, or
pass a manifest URL explicitly:

```text
http://127.0.0.1:5173/?manifest=/models/kokoro-82m/xybrid.web.json
```

The demo loads the manifest with `@xybrid/web`, runs Xybrid preprocessing and
postprocessing in TypeScript, executes the ONNX model through ORT Web, and plays
the returned WAV bytes.
