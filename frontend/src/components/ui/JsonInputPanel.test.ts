import { createSSRApp, h } from "vue";
import { renderToString } from "vue/server-renderer";
import { describe, expect, it, vi } from "vitest";

import JsonInputPanel from "@/components/ui/JsonInputPanel.vue";

vi.mock("@/components/ui/ExpressionInput.vue", () => ({
  default: {
    name: "ExpressionInput",
    props: ["modelValue"],
    ssrRender: (_ctx: unknown, push: (val: string) => void) => {
      push("<div>ExpressionInput</div>");
    },
  },
}));

interface SetupResult {
  handleFormat: () => void;
}

async function renderComponent(modelValue: string): Promise<{
  html: string;
  handleFormat: () => void;
  emittedUpdates: string[];
}> {
  const emittedUpdates: string[] = [];
  const holder: { componentSetup: SetupResult | null } = { componentSetup: null };

  const app = createSSRApp({
    setup() {
      holder.componentSetup = (JsonInputPanel as unknown as { setup: (props: unknown, ctx: unknown) => SetupResult }).setup(
        {
          modelValue,
          inputMode: "raw",
          columns: [],
        },
        {
          emit: (event: string, value: string) => {
            if (event === "update:modelValue") {
              emittedUpdates.push(value);
            }
          },
          expose: () => {},
          attrs: {},
          slots: {},
        },
      );
      return () =>
        h(JsonInputPanel, {
          modelValue,
          inputMode: "raw",
        });
    },
  });

  const html = await renderToString(app);

  if (!holder.componentSetup) {
    throw new Error("Failed to initialize JsonInputPanel setup");
  }

  const setup = holder.componentSetup as SetupResult;

  return {
    html,
    handleFormat: setup.handleFormat,
    emittedUpdates,
  };
}

describe("JsonInputPanel formatting safety", () => {
  it("renders the Format button in raw mode", async () => {
    const { html } = await renderComponent('{"a": 1}');
    expect(html).toContain("Format");
    expect(html).toContain("JSON object (supports expressions)");
  });

  it("pretty-prints valid JSON", async () => {
    const unformatted = '{"a":1,"b":["item1","item2"]}';
    const { handleFormat, emittedUpdates } = await renderComponent(unformatted);

    handleFormat();

    expect(emittedUpdates).toHaveLength(1);
    expect(emittedUpdates[0]).toBe(
      JSON.stringify(JSON.parse(unformatted), null, 2),
    );
  });

  it("leaves invalid JSON unchanged without emitting an update", async () => {
    const invalidJson = '{"a": 1, invalid_syntax}';
    const { handleFormat, emittedUpdates } = await renderComponent(invalidJson);

    handleFormat();

    expect(emittedUpdates).toHaveLength(0);
  });

  it("leaves expression-containing input unchanged without emitting an update", async () => {
    const expressionInput = '{"Age": $input.age, "Name": $input.name}';
    const { handleFormat, emittedUpdates } = await renderComponent(expressionInput);

    handleFormat();

    expect(emittedUpdates).toHaveLength(0);
  });
});
