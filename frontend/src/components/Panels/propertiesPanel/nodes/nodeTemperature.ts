export const DEFAULT_TEMPERATURE = 0.7;

/**
 * Parses user input for LLM and Agent temperature fields.
 * Preserves 0 as a valid numeric temperature.
 * Restores the default 0.7 when input is empty or invalid.
 */
export function parseTemperatureInput(raw: string | number | null | undefined): number {
  if (raw === "" || raw === null || raw === undefined) {
    return DEFAULT_TEMPERATURE;
  }
  const str = String(raw).trim();
  if (str === "") {
    return DEFAULT_TEMPERATURE;
  }
  const parsed = parseFloat(str);
  if (!Number.isFinite(parsed) || Number.isNaN(parsed)) {
    return DEFAULT_TEMPERATURE;
  }
  return parsed;
}
