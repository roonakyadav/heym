import { ref, watch, type Ref } from "vue";

import type { WorkflowNode } from "@/types/workflow";

export interface TriggerPollIntervalValidation {
  isValid: boolean;
  value: number | null;
  error: string | null;
}

/**
 * Validates trigger poll interval in minutes.
 * Rejects empty values, decimals, zero, negative numbers, and non-numeric strings.
 * Only positive integers (>= 1) are valid.
 */
export function validatePollIntervalInput(raw: string): TriggerPollIntervalValidation {
  const trimmed = raw.trim();
  if (trimmed === "") {
    return {
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    };
  }
  if (!/^\+?\d+$/.test(trimmed)) {
    return {
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    };
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed < 1) {
    return {
      isValid: false,
      value: null,
      error: "Poll interval must be a whole number of 1 or more",
    };
  }
  return {
    isValid: true,
    value: parsed,
    error: null,
  };
}

export interface UseTriggerPollIntervalResult {
  pollIntervalDraft: Ref<string>;
  pollIntervalError: Ref<string | null>;
  handlePollIntervalInput: (raw: string) => void;
}

/**
 * Shared helper for trigger polling interval input management across trigger node properties.
 * Preserves the in-progress draft while typing, rejects invalid input without overwriting
 * stored values or falling back to 1, and saves only validated positive integers.
 */
export function useTriggerPollInterval(
  selectedNode: Ref<WorkflowNode | null | undefined>,
  updateNodeData: (key: string, value: unknown) => void,
  defaultInterval = 1,
): UseTriggerPollIntervalResult {
  const pollIntervalDraft = ref<string>("");
  const pollIntervalError = ref<string | null>(null);

  watch(
    [
      () => selectedNode.value?.id,
      () => selectedNode.value?.data?.pollIntervalMinutes,
    ],
    ([newId, newMinutes], [oldId, oldMinutes]) => {
      if (newId !== oldId) {
        const initialValue =
          typeof newMinutes === "number" && Number.isInteger(newMinutes) && newMinutes >= 1
            ? newMinutes
            : defaultInterval;
        pollIntervalDraft.value = String(initialValue);
        pollIntervalError.value = null;
        return;
      }

      if (
        typeof newMinutes === "number" &&
        Number.isInteger(newMinutes) &&
        newMinutes >= 1 &&
        newMinutes !== oldMinutes &&
        Number(pollIntervalDraft.value) !== newMinutes &&
        !pollIntervalError.value
      ) {
        pollIntervalDraft.value = String(newMinutes);
      }
    },
    { immediate: true },
  );

  function handlePollIntervalInput(raw: string): void {
    pollIntervalDraft.value = raw;
    const validation = validatePollIntervalInput(raw);

    if (!validation.isValid || validation.value === null) {
      pollIntervalError.value = validation.error;
      return;
    }

    pollIntervalError.value = null;
    updateNodeData("pollIntervalMinutes", validation.value);
  }

  return {
    pollIntervalDraft,
    pollIntervalError,
    handlePollIntervalInput,
  };
}
