import type { CreateShareRequest } from "@/types/file";

export interface ShareLimitsInput {
  expiresHours?: string | null;
  maxDownloads?: string | null;
}

export interface ShareLimitValidation {
  value: number | null;
  error: string | null;
}

export interface ShareLimitsValidationResult {
  isValid: boolean;
  expiresHours: number | null;
  maxDownloads: number | null;
  errors: {
    expiresHours?: string;
    maxDownloads?: string;
  };
  errorMessage: string | null;
}

/**
 * Validates a share limit field (e.g. expires_hours or max_downloads).
 * Accepts whole numbers >= 1. Blank/empty values are treated as optional (null value, no error).
 * Rejects decimals, zero, negative numbers, and non-numeric input.
 */
export function validateShareLimit(
  raw: string | null | undefined,
  fieldLabel: string,
): ShareLimitValidation {
  if (raw == null) {
    return { value: null, error: null };
  }
  const trimmed = raw.trim();
  if (trimmed === "") {
    return { value: null, error: null };
  }

  // Reject decimals, scientific notation, negative signs, non-digits.
  // Only allow optional leading '+' and ASCII digits.
  if (!/^\+?\d+$/.test(trimmed)) {
    return {
      value: null,
      error: `${fieldLabel} must be a whole number of 1 or more`,
    };
  }

  const num = Number(trimmed);
  if (!Number.isFinite(num) || !Number.isInteger(num) || num < 1 || num > Number.MAX_SAFE_INTEGER) {
    return {
      value: null,
      error: `${fieldLabel} must be a whole number of 1 or more`,
    };
  }

  return { value: num, error: null };
}

/**
 * Validates both expires_hours and max_downloads fields for single or bulk file sharing.
 */
export function validateShareLimits(input: ShareLimitsInput): ShareLimitsValidationResult {
  const expires = validateShareLimit(input.expiresHours, "Expires (hours)");
  const downloads = validateShareLimit(input.maxDownloads, "Max downloads");

  const errors: { expiresHours?: string; maxDownloads?: string } = {};
  if (expires.error) {
    errors.expiresHours = expires.error;
  }
  if (downloads.error) {
    errors.maxDownloads = downloads.error;
  }

  const errorMessage = expires.error ?? downloads.error ?? null;

  return {
    isValid: !expires.error && !downloads.error,
    expiresHours: expires.value,
    maxDownloads: downloads.value,
    errors,
    errorMessage,
  };
}

/**
 * Builds the CreateShareRequest payload when limits are valid.
 * Preserves blank optional fields as undefined/omitted.
 */
export function buildCreateSharePayload(
  validation: ShareLimitsValidationResult,
  basicAuthPassword?: string,
): CreateShareRequest {
  const payload: CreateShareRequest = {};
  if (validation.expiresHours !== null) {
    payload.expires_hours = validation.expiresHours;
  }
  if (basicAuthPassword) {
    payload.basic_auth_password = basicAuthPassword;
  }
  if (validation.maxDownloads !== null) {
    payload.max_downloads = validation.maxDownloads;
  }
  return payload;
}
