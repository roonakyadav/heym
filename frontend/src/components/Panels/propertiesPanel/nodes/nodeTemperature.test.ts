import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  DEFAULT_TEMPERATURE,
  parseTemperatureInput,
} from "./nodeTemperature";

describe("parseTemperatureInput", () => {
  it("preserves zero as a valid temperature", () => {
    expect(parseTemperatureInput("0")).toBe(0);
    expect(parseTemperatureInput(0)).toBe(0);
    expect(parseTemperatureInput("0.0")).toBe(0);
  });

  it("parses valid positive decimal temperatures", () => {
    expect(parseTemperatureInput("0.2")).toBe(0.2);
    expect(parseTemperatureInput("1.0")).toBe(1.0);
    expect(parseTemperatureInput("1.5")).toBe(1.5);
    expect(parseTemperatureInput(0.8)).toBe(0.8);
  });

  it("restores the default 0.7 when field is cleared or whitespace", () => {
    expect(parseTemperatureInput("")).toBe(DEFAULT_TEMPERATURE);
    expect(parseTemperatureInput("   ")).toBe(DEFAULT_TEMPERATURE);
    expect(parseTemperatureInput(null)).toBe(DEFAULT_TEMPERATURE);
    expect(parseTemperatureInput(undefined)).toBe(DEFAULT_TEMPERATURE);
  });

  it("restores the default 0.7 on NaN or non-numeric input", () => {
    expect(parseTemperatureInput("abc")).toBe(DEFAULT_TEMPERATURE);
    expect(parseTemperatureInput(NaN)).toBe(DEFAULT_TEMPERATURE);
  });
});

describe("temperature configuration in LLM and Agent nodes", () => {
  it("uses nullish fallback ?? 0.7 so zero evaluates to 0 and empty/unset evaluates to 0.7", () => {
    const resolveDisplay = (temp: unknown): number => (temp as number | null | undefined) ?? DEFAULT_TEMPERATURE;

    // Stored 0 must stay 0, not fall back to 0.7 (fixing the || 0.7 bug)
    expect(resolveDisplay(0)).toBe(0);

    // Stored valid number remains that number
    expect(resolveDisplay(0.5)).toBe(0.5);

    // Unset/nullish defaults to 0.7
    expect(resolveDisplay(null)).toBe(0.7);
    expect(resolveDisplay(undefined)).toBe(0.7);
  });

  it("updates stored temperature correctly on edit, clearing, and reopening", () => {
    const nodeData: Record<string, unknown> = { temperature: 0.7 };
    const updateNodeData = (key: string, val: unknown): void => {
      nodeData[key] = val;
    };

    // User edits to 0:
    updateNodeData("temperature", parseTemperatureInput("0"));
    expect(nodeData.temperature).toBe(0);

    // Reopening with stored 0 displays 0:
    const displayAfterZero = (nodeData.temperature as number | undefined) ?? DEFAULT_TEMPERATURE;
    expect(displayAfterZero).toBe(0);

    // User clears the field:
    updateNodeData("temperature", parseTemperatureInput(""));
    expect(nodeData.temperature).toBe(0.7);

    // Reopening with restored default displays 0.7:
    const displayAfterClear = (nodeData.temperature as number | undefined) ?? DEFAULT_TEMPERATURE;
    expect(displayAfterClear).toBe(0.7);
  });

  it("configures LlmNodeProperties with ?? 0.7 and parseTemperatureInput", async () => {
    const source = await readFile(new URL("./LlmNodeProperties.vue", import.meta.url), "utf8");

    // Nullish coalescing fallback
    expect(source).toContain('selectedNode.data.temperature ?? 0.7');
    expect(source).not.toContain('selectedNode.data.temperature || 0.7');

    // Input handler parses empty as default 0.7 instead of parseFloat NaN
    expect(source).toContain("parseTemperatureInput($event as string)");
  });

  it("configures AgentNodeProperties identically with ?? 0.7 and parseTemperatureInput", async () => {
    const source = await readFile(new URL("./AgentNodeProperties.vue", import.meta.url), "utf8");

    // Nullish coalescing fallback
    expect(source).toContain('selectedNode.data.temperature ?? 0.7');
    expect(source).not.toContain('selectedNode.data.temperature || 0.7');

    // Input handler parses empty as default 0.7 instead of parseFloat NaN
    expect(source).toContain("parseTemperatureInput($event as string)");
  });
});
