import { XybridError } from "./errors.js";

export interface DecodedAudio {
  readonly samples: Float32Array;
  readonly sampleRate: number;
  readonly channels: number;
}

export function decodeWav(bytes: Uint8Array, targetSampleRate: number, channels: number): DecodedAudio {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (readAscii(bytes, 0, 4) !== "RIFF" || readAscii(bytes, 8, 4) !== "WAVE") {
    throw new XybridError("runtime_error", "AudioDecode currently requires WAV input");
  }

  let offset = 12;
  let audioFormat = 0;
  let sourceChannels = 0;
  let sourceSampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataLength = 0;

  while (offset + 8 <= bytes.byteLength) {
    const id = readAscii(bytes, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      audioFormat = view.getUint16(body, true);
      sourceChannels = view.getUint16(body + 2, true);
      sourceSampleRate = view.getUint32(body + 4, true);
      bitsPerSample = view.getUint16(body + 14, true);
    } else if (id === "data") {
      dataOffset = body;
      dataLength = size;
      break;
    }
    offset = body + size + (size % 2);
  }

  if (dataOffset < 0 || sourceChannels <= 0 || sourceSampleRate <= 0) {
    throw new XybridError("runtime_error", "Invalid WAV file");
  }
  if (audioFormat !== 1 && audioFormat !== 3) {
    throw new XybridError("runtime_error", `Unsupported WAV format ${audioFormat}`);
  }

  const frameCount = dataLength / (sourceChannels * (bitsPerSample / 8));
  const mono = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < sourceChannels; channel += 1) {
      const sampleOffset = dataOffset + (frame * sourceChannels + channel) * (bitsPerSample / 8);
      sum += readSample(view, sampleOffset, bitsPerSample, audioFormat);
    }
    mono[frame] = sum / sourceChannels;
  }

  const resampled = sourceSampleRate === targetSampleRate ? mono : resampleLinear(mono, sourceSampleRate, targetSampleRate);
  const output = channels === 1 ? resampled : duplicateChannels(resampled, channels);
  return { samples: output, sampleRate: targetSampleRate, channels };
}

export function wavFromFloat32(samples: Float32Array, sampleRate: number): Uint8Array {
  const pcm = pcm16FromFloat32(samples);
  const bytes = new Uint8Array(44 + pcm.byteLength);
  const view = new DataView(bytes.buffer);
  writeAscii(bytes, 0, "RIFF");
  view.setUint32(4, 36 + pcm.byteLength, true);
  writeAscii(bytes, 8, "WAVE");
  writeAscii(bytes, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(bytes, 36, "data");
  view.setUint32(40, pcm.byteLength, true);
  bytes.set(pcm, 44);
  return bytes;
}

export function pcm16FromFloat32(samples: Float32Array): Uint8Array {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i] ?? 0));
    view.setInt16(i * 2, Math.trunc(clamped * 32767), true);
  }
  return bytes;
}

export function postprocessTtsAudio(samples: Float32Array): Float32Array {
  let peak = 0;
  for (const sample of samples) {
    peak = Math.max(peak, Math.abs(sample));
  }
  if (peak <= 0 || peak <= 1) {
    return samples;
  }
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    out[i] = (samples[i] ?? 0) / peak;
  }
  return out;
}

export function trimTrailingNearSilence(samples: Float32Array, sampleRate: number): Float32Array {
  const threshold = 0.01;
  const minSilenceSamples = Math.floor((sampleRate * 50) / 1000);
  if (samples.length <= minSilenceSamples) {
    return samples;
  }
  let silenceRun = 0;
  let lastNonSilent = samples.length;
  for (let i = samples.length - 1; i >= 0; i -= 1) {
    if (Math.abs(samples[i] ?? 0) < threshold) {
      silenceRun += 1;
    } else {
      if (silenceRun >= minSilenceSamples) {
        const fadeBuffer = Math.floor((sampleRate * 10) / 1000);
        lastNonSilent = Math.min(samples.length, i + 1 + fadeBuffer);
      }
      break;
    }
  }
  return silenceRun >= minSilenceSamples ? samples.slice(0, lastNonSilent) : samples;
}

export function trimSamples(samples: Float32Array, count: number): Float32Array {
  if (count <= 0 || samples.length <= count) {
    return samples;
  }
  return samples.slice(0, samples.length - count);
}

export function concatFloat32(chunks: readonly Float32Array[]): Float32Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function readSample(view: DataView, offset: number, bitsPerSample: number, audioFormat: number): number {
  if (audioFormat === 3 && bitsPerSample === 32) {
    return view.getFloat32(offset, true);
  }
  if (bitsPerSample === 16) {
    return view.getInt16(offset, true) / 32768;
  }
  if (bitsPerSample === 32) {
    return view.getInt32(offset, true) / 2147483648;
  }
  if (bitsPerSample === 8) {
    return (view.getUint8(offset) - 128) / 128;
  }
  throw new XybridError("runtime_error", `Unsupported WAV bit depth ${bitsPerSample}`);
}

function resampleLinear(samples: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  const outLength = Math.max(1, Math.round((samples.length * targetRate) / sourceRate));
  const out = new Float32Array(outLength);
  const ratio = sourceRate / targetRate;
  for (let i = 0; i < outLength; i += 1) {
    const source = i * ratio;
    const left = Math.floor(source);
    const right = Math.min(samples.length - 1, left + 1);
    const frac = source - left;
    out[i] = (samples[left] ?? 0) * (1 - frac) + (samples[right] ?? 0) * frac;
  }
  return out;
}

function duplicateChannels(samples: Float32Array, channels: number): Float32Array {
  const out = new Float32Array(samples.length * channels);
  for (let i = 0; i < samples.length; i += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      out[i * channels + channel] = samples[i] ?? 0;
    }
  }
  return out;
}

function readAscii(bytes: Uint8Array, offset: number, length: number): string {
  return String.fromCharCode(...bytes.slice(offset, offset + length));
}

function writeAscii(bytes: Uint8Array, offset: number, value: string): void {
  for (let i = 0; i < value.length; i += 1) {
    bytes[offset + i] = value.charCodeAt(i);
  }
}
