"""Scriptable stand-ins for a chat provider.

The assistant's guardrails are the part worth testing, and they are all
conditionals on what a model returned — an ungrounded answer, a refusal, an
unparseable grade, a loop that never stops asking for tools.  Reaching those
branches with a real model would need it to misbehave on cue, which it will not
do reliably.  So the provider is scripted, and the tests assert on what the
platform does with each kind of reply.
"""

from __future__ import annotations

import json

from cnms_fom.rag_backend.providers import ChatResult, ToolInvocation


class ScriptedProvider:
    """Returns pre-scripted replies in order, recording what it was sent.

    Running past the end of the script is an error rather than a repeat: a loop
    that makes more calls than the test expected is exactly the bug worth
    failing on.
    """

    name = "scripted"

    def __init__(self, replies: list[ChatResult], model: str = "test-model") -> None:
        self.model = model
        self._replies = list(replies)
        self.calls: list[dict] = []

    def send(self, system, messages, *, tools=None, temperature: float = 0.0) -> ChatResult:
        self.calls.append(
            {
                "system": system,
                "messages": [dict(m) for m in messages],
                "tools": [t.name for t in (tools or [])],
                "temperature": temperature,
            }
        )
        if not self._replies:
            raise AssertionError(
                f"ScriptedProvider ran out of replies after {len(self.calls)} call(s). "
                "The code under test made more model calls than the test scripted."
            )
        reply = self._replies.pop(0)
        #  Stamp the model so assertions on AgentAnswer.model work without every
        #  test having to set it on each scripted reply.
        return ChatResult(
            text=reply.text,
            tool_calls=reply.tool_calls,
            model=reply.model or self.model,
            provider=reply.provider or self.name,
            stop_reason=reply.stop_reason,
            refused=reply.refused,
            refusal_category=reply.refusal_category,
            usage=reply.usage,
        )

    @property
    def exhausted(self) -> bool:
        return not self._replies


class GradingProvider:
    """A provider that grades every passage the same way.

    For the retrieval pipeline, where the interesting question is what happens
    when everything grades 3, or everything grades 0, not how any individual
    passage scores.
    """

    name = "grading-stub"
    model = "test-grader"

    def __init__(
        self,
        grade: int = 3,
        *,
        answer: str = "Answer from the excerpts [1].",
        rewrite: str | None = None,
        malformed_grades: bool = False,
    ) -> None:
        self.grade = grade
        self.answer = answer
        self.rewrite = rewrite
        self.malformed_grades = malformed_grades
        self.calls: list[str] = []

    def send(self, system, messages, *, tools=None, temperature: float = 0.0) -> ChatResult:  # noqa: ARG002
        body = " ".join(str(m.get("content", "")) for m in messages)

        if "Grade this passage" in body:
            self.calls.append("grade")
            text = (
                "I think this is relevant, honestly."
                if self.malformed_grades
                else json.dumps({"grade": self.grade, "reason": "stub"})
            )
            return ChatResult(text=text, model=self.model, provider=self.name)

        if "Original query:" in body:
            self.calls.append("rewrite")
            text = json.dumps({"query": self.rewrite}) if self.rewrite else "{}"
            return ChatResult(text=text, model=self.model, provider=self.name)

        self.calls.append("answer")
        return ChatResult(text=self.answer, model=self.model, provider=self.name)


def tool_reply(tool: str, arguments: dict | None = None, *, text: str = "") -> ChatResult:
    """A reply that asks for one tool call."""
    return ChatResult(
        text=text,
        tool_calls=[ToolInvocation(id=f"call_{tool}", name=tool, arguments=arguments or {})],
        stop_reason="tool_use",
    )


def text_reply(text: str) -> ChatResult:
    return ChatResult(text=text, stop_reason="end_turn")


def refusal_reply(category: str = "cyber") -> ChatResult:
    return ChatResult(text="", stop_reason="refusal", refused=True, refusal_category=category)
