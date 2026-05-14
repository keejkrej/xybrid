import { ArtifactResolver } from "./artifact-resolver.js";
import { XybridError } from "./errors.js";
import type { PhonemizerBackendKind } from "./types.js";

const PHONEME_LINK_START = "\x01";
const PHONEME_LINK_END = "\x02";
const SILENCE_TOKEN_ID = 30n;

export interface PhonemizeOptions {
  readonly tokensFile: string;
  readonly backend?: PhonemizerBackendKind | undefined;
  readonly language?: string | null | undefined;
  readonly addPadding?: boolean | undefined;
  readonly normalizeText?: boolean | undefined;
  readonly silenceTokens?: number | null | undefined;
}

export interface PhonemizeResult {
  readonly ids: readonly bigint[];
  readonly phonemes: string;
}

export function phonemizeText(text: string, options: PhonemizeOptions, artifacts: ArtifactResolver): PhonemizeResult {
  const backend = options.backend ?? "MisakiDictionary";
  if (backend !== "MisakiDictionary") {
    throw new XybridError("unsupported_step", `Phonemizer backend '${backend}' is not supported in the browser SDK yet`);
  }

  const tokens = loadTokensMap(artifacts.text(options.tokensFile));
  const processed = options.normalizeText ? normalizeTextForTts(text) : text;
  const phonemes = phonemizeMisaki(processed, artifacts);
  const ids: bigint[] = [];

  if (options.addPadding ?? true) {
    ids.push(0n);
  }
  const silence = options.silenceTokens ?? 0;
  for (let i = 0; i < silence; i += 1) {
    ids.push(SILENCE_TOKEN_ID);
  }
  for (const char of phonemes) {
    const id = tokens.get(char);
    if (id !== undefined) {
      ids.push(BigInt(id));
    } else if (char === " " && tokens.has(" ")) {
      ids.push(BigInt(tokens.get(" ") ?? 0));
    }
  }
  if (options.addPadding ?? true) {
    ids.push(0n);
  }
  return { ids, phonemes };
}

export function loadTokensMap(content: string): Map<string, number> {
  const trimmed = content.trim();
  if (trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed) as { model?: { vocab?: Record<string, number> } };
    const vocab = parsed.model?.vocab;
    if (vocab !== undefined) {
      return new Map(Object.entries(vocab));
    }
  }
  const map = new Map<string, number>();
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const tab = line.lastIndexOf("\t");
    const space = line.lastIndexOf(" ");
    const split = Math.max(tab, space);
    if (split > 0) {
      const token = line.slice(0, split);
      const id = Number(line.slice(split + 1));
      if (Number.isInteger(id)) {
        map.set(unescapeToken(token), id);
        continue;
      }
    }
    map.set(unescapeToken(line), map.size);
  }
  return map;
}

export function normalizeTextForTts(text: string): string {
  let result = parsePhonemeLinks(text);
  result = result
    .replaceAll("\u3001", ", ")
    .replaceAll("\u3002", ". ")
    .replaceAll("\uFF01", "! ")
    .replaceAll("\uFF0C", ", ")
    .replaceAll("\uFF1A", ": ")
    .replaceAll("\uFF1B", "; ")
    .replaceAll("\uFF1F", "? ")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, "\"")
    .replace(/\bDr\./g, "Doctor")
    .replace(/\bMr\./g, "Mister")
    .replace(/\bMrs\./g, "Missus")
    .replace(/\bMs\./g, "Miss")
    .replace(/\betc\./g, "etcetera");
  result = expandCurrency(result);
  result = expandPercentage(result);
  result = expandNumbers(result);
  result = result.replaceAll("...", "\u2026");
  result = result.replace(/\s+([.?!,;])/g, "$1");
  result = result.replace(/([.!?\u2026])(?=[A-Za-z0-9])/g, "$1 ");
  return result.replace(/\s+/g, " ").trim();
}

function phonemizeMisaki(text: string, artifacts: ArtifactResolver): string {
  const gold = readDictionary(artifacts, "misaki/us_gold.json");
  const silver = readDictionary(artifacts, "misaki/us_silver.json");
  const tokens = tokenizeForPhonemizer(text);
  const phonemes: string[] = [];
  for (const token of tokens) {
    if (token.startsWith(PHONEME_LINK_START) && token.endsWith(PHONEME_LINK_END)) {
      phonemes.push(token.slice(1, -1));
    } else if (/^[.,;:!?\u2026]$/.test(token)) {
      phonemes.push(token);
    } else if (/^\s+$/.test(token)) {
      phonemes.push(" ");
    } else if (/^[A-Za-z'-]+$/.test(token)) {
      phonemes.push(lookupWord(token, gold, silver) ?? ruleBasedG2p(token));
    } else {
      phonemes.push(token);
    }
  }
  return phonemes.join("").replace(/\s+/g, " ").trim();
}

function readDictionary(artifacts: ArtifactResolver, path: string): Map<string, string> {
  if (!artifacts.has(path)) {
    return new Map();
  }
  const parsed = JSON.parse(artifacts.text(path)) as Record<string, unknown>;
  const map = new Map<string, string>();
  for (const [word, value] of Object.entries(parsed)) {
    if (typeof value === "string") {
      map.set(word.toLowerCase(), value);
    } else if (value && typeof value === "object") {
      const record = value as Record<string, unknown>;
      const preferred = record.DEFAULT ?? Object.values(record)[0];
      if (typeof preferred === "string") {
        map.set(word.toLowerCase(), preferred);
      }
    }
  }
  return map;
}

function lookupWord(word: string, gold: Map<string, string>, silver: Map<string, string>): string | undefined {
  const lower = word.toLowerCase();
  return gold.get(lower) ?? silver.get(lower) ?? gold.get(word) ?? silver.get(word);
}

function tokenizeForPhonemizer(text: string): string[] {
  const tokens: string[] = [];
  let index = 0;
  while (index < text.length) {
    const char = text[index] ?? "";
    if (char === PHONEME_LINK_START) {
      const end = text.indexOf(PHONEME_LINK_END, index + 1);
      if (end >= 0) {
        tokens.push(text.slice(index, end + 1));
        index = end + 1;
        continue;
      }
    }
    const match = text.slice(index).match(/^([A-Za-z'-]+|\s+|[.,;:!?\u2026])/);
    if (match?.[0]) {
      tokens.push(match[0]);
      index += match[0].length;
    } else {
      tokens.push(char);
      index += 1;
    }
  }
  return tokens;
}

function ruleBasedG2p(word: string): string {
  const lower = word.toLowerCase();
  const parts: string[] = [];
  for (const char of lower) {
    parts.push(
      ({
        a: "æ",
        b: "b",
        c: "k",
        d: "d",
        e: "ɛ",
        f: "f",
        g: "g",
        h: "h",
        i: "ɪ",
        j: "dʒ",
        k: "k",
        l: "l",
        m: "m",
        n: "n",
        o: "oʊ",
        p: "p",
        q: "k",
        r: "ɹ",
        s: "s",
        t: "t",
        u: "ʌ",
        v: "v",
        w: "w",
        x: "ks",
        y: "j",
        z: "z",
      } as Record<string, string>)[char] ?? "",
    );
  }
  return parts.join("");
}

function parsePhonemeLinks(text: string): string {
  return text.replace(/\[[^\]]*\]\(\/([^)]*)\/\)/g, `${PHONEME_LINK_START}$1${PHONEME_LINK_END}`);
}

function expandCurrency(text: string): string {
  return text.replace(/\$(\d+)(?:\.(\d{1,2}))?/g, (_, dollars: string, cents?: string) => {
    const dollarText = `${numberToWords(Number(dollars))} dollars`;
    return cents ? `${dollarText} and ${numberToWords(Number(cents.padEnd(2, "0")))} cents` : dollarText;
  });
}

function expandPercentage(text: string): string {
  return text.replace(/\b(\d+(?:\.\d+)?)%/g, (_, number: string) => `${numberToWords(Number(number))} percent`);
}

function expandNumbers(text: string): string {
  return text.replace(/\b\d+\b/g, (number) => numberToWords(Number(number)));
}

function numberToWords(value: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  const ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
  const tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
  const integer = Math.trunc(value);
  if (integer < 20) {
    return ones[integer] ?? String(integer);
  }
  if (integer < 100) {
    const ten = Math.floor(integer / 10);
    const one = integer % 10;
    return one === 0 ? tens[ten] ?? String(integer) : `${tens[ten]} ${ones[one]}`;
  }
  if (integer < 1000) {
    const rest = integer % 100;
    return rest === 0 ? `${ones[Math.floor(integer / 100)]} hundred` : `${ones[Math.floor(integer / 100)]} hundred ${numberToWords(rest)}`;
  }
  return String(integer);
}

function unescapeToken(token: string): string {
  if (token === "\\n") {
    return "\n";
  }
  if (token === "\\t") {
    return "\t";
  }
  if (token === "<space>") {
    return " ";
  }
  return token;
}
