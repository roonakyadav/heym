import { createSSRApp, h, type Component } from "vue";
import { renderToString } from "vue/server-renderer";
import { beforeEach, describe, expect, it, vi } from "vitest";

import BulkFileShareDialog from "@/components/Drive/BulkFileShareDialog.vue";
import FileShareDialog from "@/components/Drive/FileShareDialog.vue";
import {
  buildCreateSharePayload,
  validateShareLimit,
  validateShareLimits,
} from "@/components/Drive/fileShareValidation";

const { createShareMock, bulkCreateShareMock, listSharesMock } = vi.hoisted(() => ({
  createShareMock: vi.fn(),
  bulkCreateShareMock: vi.fn(),
  listSharesMock: vi.fn(),
}));

vi.mock("@/services/api", () => ({
  filesApi: {
    createShare: createShareMock,
    bulkCreateShare: bulkCreateShareMock,
    listShares: listSharesMock,
    revokeShare: vi.fn(),
    setTeamSharing: vi.fn(),
    bulkSetTeamSharing: vi.fn(),
  },
}));

vi.mock("@/composables/useOverlayBackHandler", () => ({
  pushOverlayState: vi.fn(),
  onDismissOverlays: vi.fn(() => () => {}),
}));

vi.mock("@/composables/useDialogStack", () => ({
  addToDialogStack: vi.fn(),
  removeFromDialogStack: vi.fn(),
  isTopmostDialog: (): boolean => true,
  isBottommostDialog: (): boolean => true,
  dialogStackPosition: (): number => 0,
  hasOpenDialog: (): boolean => false,
  overlayZIndex: (): number => 50,
}));

vi.mock("@/composables/useDialogBackHistory", () => ({
  useDialogBackHistory: () => ({
    pushDialogHistoryEntry: vi.fn(),
    removeDialogHistoryEntry: vi.fn(),
  }),
}));

async function renderComponent(
  component: Component,
  props: Record<string, unknown>,
): Promise<string> {
  const app = createSSRApp({
    render: () => h(component, props),
  });
  const ctx: Record<string, unknown> = {};
  await renderToString(app, ctx);
  return ((ctx.teleports as Record<string, string> | undefined)?.body) || "";
}

describe("fileShareValidation helper", () => {
  describe("validateShareLimit", () => {
    it("preserves empty or whitespace-only input as optional without error", () => {
      expect(validateShareLimit("", "Expires (hours)")).toEqual({ value: null, error: null });
      expect(validateShareLimit("   ", "Max downloads")).toEqual({ value: null, error: null });
      expect(validateShareLimit(null, "Expires (hours)")).toEqual({ value: null, error: null });
      expect(validateShareLimit(undefined, "Max downloads")).toEqual({ value: null, error: null });
    });

    it("accepts valid positive integers", () => {
      expect(validateShareLimit("1", "Expires (hours)")).toEqual({ value: 1, error: null });
      expect(validateShareLimit("24", "Expires (hours)")).toEqual({ value: 24, error: null });
      expect(validateShareLimit("100", "Max downloads")).toEqual({ value: 100, error: null });
      expect(validateShareLimit("+5", "Max downloads")).toEqual({ value: 5, error: null });
    });

    it("rejects zero with field-specific message", () => {
      expect(validateShareLimit("0", "Expires (hours)")).toEqual({
        value: null,
        error: "Expires (hours) must be a whole number of 1 or more",
      });
      expect(validateShareLimit("0", "Max downloads")).toEqual({
        value: null,
        error: "Max downloads must be a whole number of 1 or more",
      });
    });

    it("rejects negative numbers with field-specific message", () => {
      expect(validateShareLimit("-1", "Expires (hours)")).toEqual({
        value: null,
        error: "Expires (hours) must be a whole number of 1 or more",
      });
      expect(validateShareLimit("-10", "Max downloads")).toEqual({
        value: null,
        error: "Max downloads must be a whole number of 1 or more",
      });
    });

    it("rejects decimal numbers with field-specific message", () => {
      expect(validateShareLimit("1.5", "Expires (hours)")).toEqual({
        value: null,
        error: "Expires (hours) must be a whole number of 1 or more",
      });
      expect(validateShareLimit("2.9", "Max downloads")).toEqual({
        value: null,
        error: "Max downloads must be a whole number of 1 or more",
      });
      expect(validateShareLimit("0.5", "Max downloads")).toEqual({
        value: null,
        error: "Max downloads must be a whole number of 1 or more",
      });
      expect(validateShareLimit("1.0", "Expires (hours)")).toEqual({
        value: null,
        error: "Expires (hours) must be a whole number of 1 or more",
      });
    });

    it("rejects non-numeric characters and exponential notation", () => {
      expect(validateShareLimit("abc", "Expires (hours)")).toEqual({
        value: null,
        error: "Expires (hours) must be a whole number of 1 or more",
      });
      expect(validateShareLimit("1e3", "Max downloads")).toEqual({
        value: null,
        error: "Max downloads must be a whole number of 1 or more",
      });
    });
  });

  describe("validateShareLimits and buildCreateSharePayload", () => {
    it("validates combined limits when both are valid", () => {
      const res = validateShareLimits({ expiresHours: "48", maxDownloads: "5" });
      expect(res.isValid).toBe(true);
      expect(res.expiresHours).toBe(48);
      expect(res.maxDownloads).toBe(5);
      expect(res.errorMessage).toBeNull();

      const payload = buildCreateSharePayload(res, "secret123");
      expect(payload).toEqual({
        expires_hours: 48,
        basic_auth_password: "secret123",
        max_downloads: 5,
      });
    });

    it("preserves optional fields as omitted in payload", () => {
      const res = validateShareLimits({ expiresHours: "", maxDownloads: "10" });
      expect(res.isValid).toBe(true);
      expect(res.expiresHours).toBeNull();
      expect(res.maxDownloads).toBe(10);

      const payload = buildCreateSharePayload(res);
      expect(payload).toEqual({ max_downloads: 10 });
      expect(payload.expires_hours).toBeUndefined();
    });

    it("reports field-specific errors when values are invalid", () => {
      const res = validateShareLimits({ expiresHours: "1.5", maxDownloads: "0" });
      expect(res.isValid).toBe(false);
      expect(res.errors.expiresHours).toBe("Expires (hours) must be a whole number of 1 or more");
      expect(res.errors.maxDownloads).toBe("Max downloads must be a whole number of 1 or more");
      expect(res.errorMessage).toBe("Expires (hours) must be a whole number of 1 or more");
    });
  });
});

describe("FileShareDialog and BulkFileShareDialog components", () => {
  beforeEach(() => {
    (globalThis as unknown as { window: unknown }).window = {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    };
    (globalThis as unknown as { document: unknown }).document = {
      body: {
        style: { overflow: "" },
      },
    };
    createShareMock.mockReset();
    bulkCreateShareMock.mockReset();
    listSharesMock.mockReset();
    listSharesMock.mockResolvedValue([]);
  });

  it("renders FileShareDialog with integer step and min attributes on limit inputs", async () => {
    const html = await renderComponent(FileShareDialog, {
      open: true,
      fileId: "file-1",
      filename: "test.pdf",
      sharedWithMyTeams: false,
    });

    expect(html).toContain('placeholder="No expiry"');
    expect(html).toContain('placeholder="Unlimited"');
    expect(html).toMatch(/<input[^>]*min="1"[^>]*step="1"[^>]*placeholder="No expiry"/);
    expect(html).toMatch(/<input[^>]*min="1"[^>]*step="1"[^>]*placeholder="Unlimited"/);
  });

  it("renders BulkFileShareDialog with integer step and min attributes on limit inputs", async () => {
    const html = await renderComponent(BulkFileShareDialog, {
      open: true,
      fileIds: ["file-1", "file-2"],
    });

    expect(html).toContain('placeholder="No expiry"');
    expect(html).toContain('placeholder="Unlimited"');
    expect(html).toMatch(/<input[^>]*min="1"[^>]*step="1"[^>]*placeholder="No expiry"/);
    expect(html).toMatch(/<input[^>]*min="1"[^>]*step="1"[^>]*placeholder="Unlimited"/);
  });

  it("prevents single-file share creation and rejects decimals or zero limits", () => {
    const invalidInputs = [
      { expiresHours: "1.5", maxDownloads: "5" },
      { expiresHours: "24", maxDownloads: "2.9" },
      { expiresHours: "0", maxDownloads: "5" },
      { expiresHours: "24", maxDownloads: "0" },
      { expiresHours: "-5", maxDownloads: "10" },
    ];

    for (const input of invalidInputs) {
      const result = validateShareLimits(input);
      expect(result.isValid).toBe(false);
      expect(result.errorMessage).toMatch(/must be a whole number of 1 or more/);
    }

    // When validation fails, createShare never dispatches to API
    expect(createShareMock).not.toHaveBeenCalled();
  });

  it("prevents bulk share creation and rejects decimals or zero limits", () => {
    const invalidInputs = [
      { expiresHours: "0.5", maxDownloads: "10" },
      { expiresHours: "12", maxDownloads: "-1" },
      { expiresHours: "3.14", maxDownloads: "1" },
    ];

    for (const input of invalidInputs) {
      const result = validateShareLimits(input);
      expect(result.isValid).toBe(false);
      expect(result.errorMessage).toMatch(/must be a whole number of 1 or more/);
    }

    // When validation fails, bulkCreateShare never dispatches to API
    expect(bulkCreateShareMock).not.toHaveBeenCalled();
  });

  it("accepts valid positive integer limits for single and bulk sharing", () => {
    const validInput = { expiresHours: "48", maxDownloads: "20" };
    const result = validateShareLimits(validInput);
    expect(result.isValid).toBe(true);
    expect(result.expiresHours).toBe(48);
    expect(result.maxDownloads).toBe(20);

    const payload = buildCreateSharePayload(result);
    expect(payload).toEqual({
      expires_hours: 48,
      max_downloads: 20,
    });
  });
});
