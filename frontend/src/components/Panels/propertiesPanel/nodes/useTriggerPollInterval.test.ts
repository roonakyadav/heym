import { ref } from "vue";
import { describe, expect, it, vi } from "vitest";

import type { WorkflowNode } from "@/types/workflow";
import {
  useTriggerPollInterval,
  validatePollIntervalInput,
} from "./useTriggerPollInterval";

describe("validatePollIntervalInput", () => {
  it("rejects empty and whitespace-only input", () => {
    expect(validatePollIntervalInput("")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
    expect(validatePollIntervalInput("   ")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
  });

  it("rejects decimals including typing intermediate values", () => {
    expect(validatePollIntervalInput("2.9")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
    expect(validatePollIntervalInput("1.5")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
    expect(validatePollIntervalInput("0.5")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
    expect(validatePollIntervalInput("2.")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
  });

  it("rejects zero and negative numbers", () => {
    expect(validatePollIntervalInput("0")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
    expect(validatePollIntervalInput("-1")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
    expect(validatePollIntervalInput("-5")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
  });

  it("rejects non-numeric characters and exponential values", () => {
    expect(validatePollIntervalInput("abc")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
    expect(validatePollIntervalInput("1e4")).toEqual({
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    });
  });

  it("accepts positive integers", () => {
    expect(validatePollIntervalInput("1")).toEqual({
      isValid: true,
      value: 1,
      error: null,
    });
    expect(validatePollIntervalInput("5")).toEqual({
      isValid: true,
      value: 5,
      error: null,
    });
    expect(validatePollIntervalInput("60")).toEqual({
      isValid: true,
      value: 60,
      error: null,
    });
    expect(validatePollIntervalInput("+10")).toEqual({
      isValid: true,
      value: 10,
      error: null,
    });
  });
});

describe("useTriggerPollInterval composable", () => {
  function createTestNode(id: string, type: string, minutes?: number): WorkflowNode {
    return {
      id,
      type,
      position: { x: 0, y: 0 },
      data: {
        label: `${type}Label`,
        ...(minutes !== undefined ? { pollIntervalMinutes: minutes } : {}),
      },
    } as WorkflowNode;
  }

  const triggerTypes = [
    { type: "imapTrigger", defaultMinutes: 5 },
    { type: "googleDriveTrigger", defaultMinutes: 1 },
    { type: "googleSheetsTrigger", defaultMinutes: 1 },
    { type: "gmailTrigger", defaultMinutes: 1 },
    { type: "outlookTrigger", defaultMinutes: 1 },
  ];

  for (const { type, defaultMinutes } of triggerTypes) {
    it(`initializes draft and preserves saved value for ${type}`, () => {
      const node = ref<WorkflowNode | null>(createTestNode("node-1", type, 15));
      const updateMock = vi.fn();

      const { pollIntervalDraft, pollIntervalError } = useTriggerPollInterval(
        node,
        updateMock,
        defaultMinutes,
      );

      expect(pollIntervalDraft.value).toBe("15");
      expect(pollIntervalError.value).toBeNull();
      expect(updateMock).not.toHaveBeenCalled();
    });

    it(`uses default ${defaultMinutes} when interval is unset for ${type}`, () => {
      const node = ref<WorkflowNode | null>(createTestNode("node-1", type));
      const updateMock = vi.fn();

      const { pollIntervalDraft, pollIntervalError } = useTriggerPollInterval(
        node,
        updateMock,
        defaultMinutes,
      );

      expect(pollIntervalDraft.value).toBe(String(defaultMinutes));
      expect(pollIntervalError.value).toBeNull();
    });
  }

  it("preserves in-progress draft on decimal input without corrupting stored value", () => {
    const node = ref<WorkflowNode | null>(createTestNode("imap-1", "imapTrigger", 5));
    const updateMock = vi.fn((key: string, val: unknown) => {
      (node.value!.data as Record<string, unknown>)[key] = val;
    });

    const {
      pollIntervalDraft,
      pollIntervalError,
      handlePollIntervalInput,
    } = useTriggerPollInterval(node, updateMock, 5);

    // Typing '2.9' key by key simulation:
    // Key 1: '2'
    handlePollIntervalInput("2");
    expect(pollIntervalDraft.value).toBe("2");
    expect(pollIntervalError.value).toBeNull();
    expect(updateMock).toHaveBeenCalledWith("pollIntervalMinutes", 2);
    expect(node.value?.data.pollIntervalMinutes).toBe(2);

    updateMock.mockClear();

    // Key 2: at '2.', browser emits '' or '2.'
    handlePollIntervalInput("");
    expect(pollIntervalDraft.value).toBe("");
    expect(pollIntervalError.value).toBe("Poll interval must be a whole number of 1 or more");
    // CRITICAL: updateMock must NOT be called with fallback 1 or empty
    expect(updateMock).not.toHaveBeenCalled();
    expect(node.value?.data.pollIntervalMinutes).toBe(2);

    // Key 3: '9' appends to draft '2.9'
    handlePollIntervalInput("2.9");
    expect(pollIntervalDraft.value).toBe("2.9");
    expect(pollIntervalError.value).toBe("Poll interval must be a whole number of 1 or more");
    expect(updateMock).not.toHaveBeenCalled();
    expect(node.value?.data.pollIntervalMinutes).toBe(2);

    // Correcting to a valid integer '3'
    handlePollIntervalInput("3");
    expect(pollIntervalDraft.value).toBe("3");
    expect(pollIntervalError.value).toBeNull();
    expect(updateMock).toHaveBeenCalledWith("pollIntervalMinutes", 3);
    expect(node.value?.data.pollIntervalMinutes).toBe(3);
  });

  it("rejects zero and negative input without updating stored value", () => {
    const node = ref<WorkflowNode | null>(createTestNode("gdt-1", "googleDriveTrigger", 10));
    const updateMock = vi.fn();

    const {
      pollIntervalDraft,
      pollIntervalError,
      handlePollIntervalInput,
    } = useTriggerPollInterval(node, updateMock, 1);

    handlePollIntervalInput("0");
    expect(pollIntervalDraft.value).toBe("0");
    expect(pollIntervalError.value).toBe("Poll interval must be a whole number of 1 or more");
    expect(updateMock).not.toHaveBeenCalled();

    handlePollIntervalInput("-5");
    expect(pollIntervalDraft.value).toBe("-5");
    expect(pollIntervalError.value).toBe("Poll interval must be a whole number of 1 or more");
    expect(updateMock).not.toHaveBeenCalled();
  });
});
