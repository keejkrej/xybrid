import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
  },
  resolve: {
    alias: {
      "react-native": new URL("./test/react-native-mock.ts", import.meta.url).pathname,
    },
  },
});
