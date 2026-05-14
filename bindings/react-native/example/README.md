# Xybrid React Native Example

Create a bare React Native app with the New Architecture enabled, then point it at this package:

```sh
npx @react-native-community/cli@latest init XybridRnExample --version 0.85
cd XybridRnExample
npm install /absolute/path/to/xybrid/bindings/react-native
```

Android local development should set `xybridKotlinDir` in `android/gradle.properties` so the package can reuse the monorepo Kotlin binding and native libraries.

```tsx
import React from "react";
import { Button, SafeAreaView, Text } from "react-native";
import { Envelope, Xybrid } from "@xybrid/react-native";

export default function App() {
  const [status, setStatus] = React.useState("idle");

  async function run() {
    setStatus("initializing");
    await Xybrid.init();
    const model = await Xybrid.model("kokoro-82m").load({
      onProgress: (event) => setStatus(`${event.phase} ${Math.round(event.progress * 100)}%`),
    });
    const result = await model.run(Envelope.text("Hello from React Native", { voiceId: "af_heart" }));
    setStatus(result.audioBytes ? `audio bytes: ${result.audioBytes.byteLength}` : result.text ?? "done");
    model.dispose();
  }

  return (
    <SafeAreaView>
      <Text>{status}</Text>
      <Button title="Run Kokoro" onPress={run} />
    </SafeAreaView>
  );
}
```
