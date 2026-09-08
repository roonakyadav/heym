import { beforeEach, describe, expect, it, vi } from "vitest";

import { useSkillBuilder } from "@/composables/useSkillBuilder";
import { skillBuilderStream } from "@/services/skillBuilderApi";

vi.mock("@/services/skillBuilderApi", () => ({ skillBuilderStream: vi.fn() }));

describe("skill builder conversation identity", () => {
  beforeEach(() => vi.clearAllMocks());

  it("keeps the session for followups and errors, then rotates it on reset", () => {
    const chat = useSkillBuilder();
    const stream = vi.mocked(skillBuilderStream);
    chat.sendMessage("Create a skill", "credential", "model");
    const first = stream.mock.calls[0];
    expect(first[0].conversationId).toBeTruthy();
    first[3]();
    chat.sendMessage("Improve it", "credential", "model");
    const second = stream.mock.calls[1];
    expect(second[0].conversationId).toBe(first[0].conversationId);
    second[4](new Error("Temporary failure"));
    chat.sendMessage("Try again", "credential", "model");
    expect(stream.mock.calls[2][0].conversationId).toBe(first[0].conversationId);
    chat.reset();
    chat.sendMessage("Another skill", "credential", "model");
    expect(stream.mock.calls[3][0].conversationId).not.toBe(first[0].conversationId);
  });

  it("uses different sessions for independent conversations", () => {
    useSkillBuilder().sendMessage("One", "credential", "model");
    useSkillBuilder().sendMessage("Two", "credential", "model");
    const calls = vi.mocked(skillBuilderStream).mock.calls;
    expect(calls[0][0].conversationId).not.toBe(calls[1][0].conversationId);
  });
});
