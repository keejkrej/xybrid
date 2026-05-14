import { Xybrid, XybridEnvelope, type BackendKind, type XybridModel } from "../src/index.js";
import "./styles.css";

const DEFAULT_MANIFEST_URL = "/models/kokoro-82m/xybrid.web.json";

const voiceSelect = element<HTMLSelectElement>("voice-select");
const deviceSelect = element<HTMLSelectElement>("device-select");
const dtypeSelect = element<HTMLSelectElement>("dtype-select");
const speedRange = element<HTMLInputElement>("speed-range");
const speedOutput = element<HTMLOutputElement>("speed-output");
const textInput = element<HTMLTextAreaElement>("text-input");
const loadButton = element<HTMLButtonElement>("load-button");
const speakButton = element<HTMLButtonElement>("speak-button");
const modelStatus = element<HTMLElement>("model-status");
const latencyStatus = element<HTMLElement>("latency-status");
const audioStatus = element<HTMLElement>("audio-status");
const runtimePill = element<HTMLElement>("runtime-pill");
const activityLog = element<HTMLElement>("activity-log");
const audioPlayer = element<HTMLAudioElement>("audio-player");
const canvas = element<HTMLCanvasElement>("waveform-canvas");

let model: XybridModel | null = null;
let audioUrl: string | null = null;
let loadingBackend: BackendKind | null = null;

initialize();

function initialize(): void {
  dtypeSelect.disabled = true;
  dtypeSelect.replaceChildren(option("xybrid", "Xybrid manifest"));
  populateVoices([
    { id: "af_heart", name: "Heart" },
    { id: "af_bella", name: "Bella" },
    { id: "af_nicole", name: "Nicole" },
    { id: "af_sarah", name: "Sarah" },
    { id: "am_adam", name: "Adam" },
    { id: "am_michael", name: "Michael" },
    { id: "bf_emma", name: "Emma" },
    { id: "bm_george", name: "George" },
  ]);
  if (!("gpu" in navigator)) {
    for (const option of deviceSelect.options) {
      if (option.value === "webgpu") {
        option.disabled = true;
        option.textContent = "WebGPU unavailable";
      }
    }
  }
  speedRange.addEventListener("input", () => {
    speedOutput.value = Number(speedRange.value).toFixed(2);
  });
  deviceSelect.addEventListener("change", resetLoadedModel);
  loadButton.addEventListener("click", () => void loadModel());
  speakButton.addEventListener("click", () => void synthesize());
  drawIdleWaveform();
  log("Ready");
}

async function loadModel(): Promise<void> {
  const backend = selectedBackend();
  if (model && loadingBackend === backend) {
    return;
  }

  setBusy(true, "Loading");
  modelStatus.textContent = "Loading";
  runtimePill.textContent = backend;
  log(`Loading ${manifestUrl()}`);

  try {
    const startedAt = performance.now();
    const xybrid = await Xybrid.create({ backend, cache: "persistent" });
    model = await xybrid.model(manifestUrl()).load({
      onProgress: (event) => {
        if ("path" in event) {
          const percent =
            event.totalBytes && event.totalBytes > 0 ? Math.round((event.loadedBytes / event.totalBytes) * 100) : undefined;
          modelStatus.textContent = percent === undefined ? event.type : `${percent}%`;
          log(`${event.type}: ${event.path}`, true);
        }
      },
    });
    loadingBackend = backend;
    const voices = model.voices();
    if (voices.length > 0) {
      populateVoices(voices);
      voiceSelect.value = model.defaultVoice()?.id ?? voices[0]?.id ?? voiceSelect.value;
    }
    const elapsed = Math.round(performance.now() - startedAt);
    modelStatus.textContent = "Loaded";
    latencyStatus.textContent = `${elapsed} ms load`;
    speakButton.disabled = false;
    log("Model loaded");
  } catch (error) {
    model = null;
    loadingBackend = null;
    speakButton.disabled = true;
    modelStatus.textContent = "Load failed";
    log(formatError(error));
  } finally {
    setBusy(false);
  }
}

async function synthesize(): Promise<void> {
  if (!model) {
    await loadModel();
  }
  if (!model) {
    return;
  }

  const text = textInput.value.trim();
  if (!text) {
    log("Text is empty");
    return;
  }

  setBusy(true, "Running");
  latencyStatus.textContent = "Running";
  audioStatus.textContent = "-";
  log("Synthesizing");

  try {
    const startedAt = performance.now();
    const result = await model.run(XybridEnvelope.text(text), {
      voice: voiceSelect.value,
      speed: Number(speedRange.value),
    });
    const elapsed = Math.round(performance.now() - startedAt);
    const audio = result.unwrapAudio();
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    audioUrl = URL.createObjectURL(new Blob([audio], { type: "audio/wav" }));
    audioPlayer.src = audioUrl;
    audioPlayer.load();
    const samples = result.value.type === "audio" ? result.value.samples : undefined;
    if (samples) {
      drawWaveform(samples);
      audioStatus.textContent = `${(samples.length / (result.value.sampleRate ?? 24000)).toFixed(2)} s`;
    } else {
      drawIdleWaveform();
      audioStatus.textContent = `${Math.round(audio.byteLength / 1024)} KB`;
    }
    latencyStatus.textContent = `${elapsed} ms`;
    log("Audio ready");
    await audioPlayer.play().catch(() => undefined);
  } catch (error) {
    latencyStatus.textContent = "Failed";
    log(formatError(error));
  } finally {
    setBusy(false);
  }
}

function resetLoadedModel(): void {
  if (!model) {
    return;
  }
  void model.unload();
  model = null;
  loadingBackend = null;
  speakButton.disabled = true;
  modelStatus.textContent = "Not loaded";
  latencyStatus.textContent = "-";
  audioStatus.textContent = "-";
  drawIdleWaveform();
  log("Runtime changed");
}

function populateVoices(voices: readonly { id: string; name: string }[]): void {
  const selected = voiceSelect.value || "af_heart";
  voiceSelect.replaceChildren(
    ...voices.map((voice) => {
      const item = option(voice.id, `${voice.name} (${voice.id})`);
      return item;
    }),
  );
  voiceSelect.value = voices.some((voice) => voice.id === selected) ? selected : (voices[0]?.id ?? "");
}

function selectedBackend(): BackendKind {
  if (deviceSelect.value === "webgpu" && "gpu" in navigator) {
    return "ort-webgpu";
  }
  return "ort-wasm";
}

function manifestUrl(): string {
  return new URLSearchParams(window.location.search).get("manifest") ?? DEFAULT_MANIFEST_URL;
}

function option(value: string, label: string): HTMLOptionElement {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}

function setBusy(isBusy: boolean, label = "Working"): void {
  loadButton.disabled = isBusy;
  speakButton.disabled = isBusy || !model;
  loadButton.textContent = isBusy ? label : "Load Model";
  speakButton.textContent = isBusy ? label : "Synthesize";
}

function log(message: string, replaceLast = false): void {
  const line = document.createElement("div");
  line.textContent = `${new Date().toLocaleTimeString()}  ${message}`;
  if (replaceLast && activityLog.lastElementChild) {
    activityLog.replaceChild(line, activityLog.lastElementChild);
  } else {
    activityLog.prepend(line);
  }
  while (activityLog.children.length > 7) {
    activityLog.lastElementChild?.remove();
  }
}

function drawIdleWaveform(): void {
  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvasContext();
  ctx.clearRect(0, 0, width, height);
  drawCanvasBackground(ctx, width, height);
  ctx.strokeStyle = "#71c7a9";
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    const y = height / 2 + Math.sin(x / 22) * 22 + Math.sin(x / 53) * 8;
    if (x === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

function drawWaveform(samples: Float32Array): void {
  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvasContext();
  ctx.clearRect(0, 0, width, height);
  drawCanvasBackground(ctx, width, height);

  const step = Math.max(1, Math.floor(samples.length / width));
  const mid = height / 2;
  const scale = height * 0.42;

  ctx.strokeStyle = "#e0b15a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    let min = 1;
    let max = -1;
    const start = x * step;
    const end = Math.min(samples.length, start + step);
    for (let index = start; index < end; index += 1) {
      const value = samples[index] ?? 0;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    ctx.moveTo(x, mid + min * scale);
    ctx.lineTo(x, mid + max * scale);
  }
  ctx.stroke();
}

function drawCanvasBackground(ctx: CanvasRenderingContext2D, width: number, height: number): void {
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#121a20");
  gradient.addColorStop(1, "#0c1016");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let y = 40; y < height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function canvasContext(): CanvasRenderingContext2D {
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas 2D context is unavailable");
  }
  return context;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) {
    throw new Error(`Missing element #${id}`);
  }
  return found as T;
}
