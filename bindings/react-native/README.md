# @xybrid/react-native

React Native New Architecture bindings for Xybrid native mobile inference. This package is for bare React Native apps; Expo Go and React Native Web are out of scope.

React Native mobile uses the same Rust + UniFFI + native ONNX Runtime stack as the Kotlin and Swift SDKs. It does not use `onnxruntime-web`. Browser and React Native Web apps should use `@xybrid/web`.

## Install

```sh
npm install @xybrid/react-native
```

Bare apps must have the New Architecture enabled. Android consumes the existing Kotlin binding and native `.so` libraries. iOS consumes `bindings/apple/XCFrameworks/XybridFFI.xcframework` through CocoaPods.

For local monorepo Android development, point Gradle at the checked-out Kotlin binding:

```properties
xybridKotlinDir=/absolute/path/to/xybrid/bindings/kotlin
```

For iOS local development, build the XCFramework before `pod install`:

```sh
cargo xtask build-xcframework
cd ios && pod install
```

## Usage

```ts
import { Envelope, Xybrid } from "@xybrid/react-native";

await Xybrid.init();

const model = await Xybrid.model("kokoro-82m").load({
  onProgress: (event) => console.log(event.phase, event.progress),
});

const result = await model.run(
  Envelope.text("Hello from React Native", {
    voiceId: "af_heart",
    speed: 1.0,
  }),
);

await model.dispose();
```

## API

- `Xybrid.init({ cacheDir? })`
- `Xybrid.setApiKey(apiKey)`
- `Xybrid.model("model-id" | ModelSource).load({ onProgress? })`
- `model.run(Envelope.text(...) | Envelope.audio(...) | Envelope.embedding(...), config?)`
- `model.voices()`
- `model.defaultVoiceId()`
- `model.dispose()`
- `Xybrid.isModelCached(modelId)`

`ModelSource` supports `registry`, `bundle`, `directory`, and `huggingface`. Binary payloads cross the TurboModule boundary as base64 in v1.

The native binding reports `binding=react-native` in `X-Xybrid-Client` registry metadata requests.
