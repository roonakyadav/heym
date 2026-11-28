import { expect, test } from "@playwright/test";

import {
  createWorkflow,
  deleteWorkflow,
  prepareAuthenticatedPage,
} from "./support";

test.beforeEach(async ({ page }) => {
  await prepareAuthenticatedPage(page);
});

test("Trigger poll intervals reject decimal input typed key-by-key and save valid integer", async ({
  page,
}) => {
  const triggers = [
    { id: "imap-trigger", type: "imapTrigger", initialMinutes: 5, inputId: "#imap-poll-interval" },
    { id: "drive-trigger", type: "googleDriveTrigger", initialMinutes: 1, inputId: "#gdt-poll-interval" },
    { id: "sheet-trigger", type: "googleSheetsTrigger", initialMinutes: 1, inputId: "#gst-poll-interval" },
    { id: "gmail-trigger", type: "gmailTrigger", initialMinutes: 1, inputId: "#gmail-poll-interval" },
    { id: "outlook-trigger", type: "outlookTrigger", initialMinutes: 1, inputId: "#outlook-poll-interval" },
  ];

  const workflow = await createWorkflow(
    page,
    `Triggers Poll Test ${Date.now()}`,
    triggers.map((t, index) => ({
      id: t.id,
      type: t.type,
      position: { x: 100 + index * 120, y: 150 },
      data: {
        label: t.id,
        pollIntervalMinutes: t.initialMinutes,
      },
    })),
  );

  try {
    await page.goto(`/workflows/${workflow.id}`);
    await expect(page.locator(".vue-flow__node")).toHaveCount(5);
    await page.getByRole("button", { name: "Properties", exact: true }).click();

    for (const trigger of triggers) {
      await page.locator(`.vue-flow__node[data-id="${trigger.id}"]`).click();

      const input = page.locator(trigger.inputId);
      await expect(input).toBeVisible();
      await expect(input).toHaveValue(String(trigger.initialMinutes));

      // Clear input and type 2.9 key by key
      await input.click();
      await input.fill("");
      await input.pressSequentially("2.9");

      // Verify validation message is displayed and stored value is NOT corrupted
      const panel = page.locator(".properties-panel");
      await expect(
        panel.locator("text=Poll interval must be a whole number of 1 or more"),
      ).toBeVisible();

      // Clear and type valid positive whole number 12
      await input.fill("");
      await input.pressSequentially("12");

      // Validation error should disappear
      await expect(
        panel.locator("text=Poll interval must be a whole number of 1 or more"),
      ).toHaveCount(0);
      await expect(input).toHaveValue("12");
    }

    // Save the workflow
    const saved = page.waitForResponse(
      (response) =>
        response.request().method() === "PUT" &&
        new URL(response.url()).pathname === `/api/workflows/${workflow.id}`,
    );
    await page.getByRole("button", { name: "Save", exact: true }).click();
    await saved;

    // Verify stored data in the database via API
    const stored = await page.request.get(`/api/workflows/${workflow.id}`);
    expect(stored.ok()).toBeTruthy();
    const body = (await stored.json()) as {
      nodes: { id: string; data: Record<string, unknown> }[];
    };

    for (const trigger of triggers) {
      const node = body.nodes.find((item) => item.id === trigger.id);
      expect(node?.data.pollIntervalMinutes).toBe(12);
    }
  } finally {
    await deleteWorkflow(page, workflow.id);
  }
});
