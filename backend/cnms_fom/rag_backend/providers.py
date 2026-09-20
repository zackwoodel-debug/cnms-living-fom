"""Chat-model providers: local Ollama by default, Anthropic by explicit opt-in.

Why this abstraction exists at all
----------------------------------
The corpus this assistant reads is MBE/PLD/ALD process notes and CNMS user
documents — unpublished work belonging to facility users.  ``rag_backend`` was
built local-only for that reason, and that remains the default: nothing in a
retrieved passage leaves the machine unless someone sets
``RAG_LLM_PROVIDER=anthropic`` on purpose.

The reason to offer the alternative anyway is that the retrieval discipline this
platform needs is hard for a small local model.  Declining to answer when the
evidence is thin, carrying a growth parameter's full context along with its
number, and reporting two sources' disagreement instead of averaging them are
all instruction-following problems, and an 8B model follows those instructions
less reliably than a frontier one.  So the provider is a choice with a stated
trade-off — locality against retrieval fidelity — rather than a default anyone
should drift into.

``send()`` is deliberately small: system prompt in, messages in, optional tools
in, text and tool calls out.  Everything above it (grading, reranking, the
correction loop, the agent) is provider-agnostic, which is the only way to keep
the guardrails identical on both paths.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from cnms_fom.config import get_settings

logger = logging.getLogger(__name__)

Provider = Literal["ollama", "anthropic"]

#  Anthropic defaults.  Claude Opus 5 with adaptive thinking: this is a
#  reasoning task under hard constraints, not a formatting task, and the
#  constraint that matters most ("say you do not know") is the one a model
#  skips when it is not thinking.
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5"
ANTHROPIC_MAX_TOKENS = 16_000
#  Server-side refusal fallback: a policy decline re-runs on a fallback model
#  inside the same call instead of returning nothing. "default" routes by
#  refusal category, so there is no model list to maintain.
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ToolSpec:
    """A tool the model may call, in the one shape both providers accept."""

    name: str
    description: str
    input_schema: dict

    def as_anthropic(self) -> dict:
        #  strict=True guarantees the arguments validate against the schema,
        #  which matters here because a tool argument is a database query: a
        #  hallucinated field name should fail at the schema, not at the SQL.
        #
        #  ``required`` is whatever the tool declared, defaulting to empty —
        #  never every property. Auto-requiring all of them would force the model
        #  to supply each optional filter on every call, which turns "list the
        #  property values for HfO2" into a call that also has to invent a
        #  property_key.
        schema = {**self.input_schema, "additionalProperties": False}
        schema.setdefault("required", [])
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
            "strict": True,
        }

    def as_openai_style(self) -> dict:
        """The ``{type: function, function: {...}}`` shape LangChain passes on."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class ToolInvocation:
    """One tool call the model asked for."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    """What a provider returned: text, tool calls, and how it stopped."""

    text: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    stop_reason: str | None = None
    #  True when the provider's safety layer declined. Surfaced rather than
    #  swallowed: a refusal that reads as an empty answer would be recorded as a
    #  data gap, which is a different finding entirely.
    refused: bool = False
    refusal_category: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ChatProvider(Protocol):
    """The whole provider contract."""

    name: str
    model: str

    def send(
        self,
        system: str,
        messages: list[dict],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
    ) -> ChatResult: ...


# ---------------------------------------------------------------------------
# Ollama — the default
# ---------------------------------------------------------------------------


class OllamaProvider:
    """Local Ollama through ``langchain_ollama.ChatOllama``.

    Temperature defaults to 0 everywhere in this package: retrieval is not a
    creative task, and sampling diversity here buys nothing but a wider
    distribution of invented numbers.
    """

    name = "ollama"

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_chat_model
        self._base_url = settings.ollama_base_url

    def _client(self, temperature: float, tools: list[ToolSpec] | None):
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError("RAG needs the 'rag' extra: pip install -e '.[rag]'") from exc

        chat = ChatOllama(model=self.model, base_url=self._base_url, temperature=temperature)
        if not tools:
            return chat
        try:
            return chat.bind_tools([tool.as_openai_style() for tool in tools])
        except (AttributeError, NotImplementedError):
            #  Not every local model supports tool calling. Returning the
            #  unbound client lets the caller fall back to plain retrieval
            #  rather than failing the request.
            logger.warning(
                "Model %s does not support tool calling; falling back to single-shot retrieval.",
                self.model,
            )
            return chat

    def send(
        self,
        system: str,
        messages: list[dict],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
    ) -> ChatResult:
        try:
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError("RAG needs the 'rag' extra: pip install -e '.[rag]'") from exc

        converted: list[Any] = [SystemMessage(content=system)]
        for message in messages:
            role = message.get("role")
            if role == "user":
                converted.append(HumanMessage(content=message["content"]))
            elif role == "assistant":
                converted.append(
                    AIMessage(
                        content=message.get("content") or "",
                        tool_calls=[
                            {"id": c["id"], "name": c["name"], "args": c["arguments"]}
                            for c in message.get("tool_calls", [])
                        ],
                    )
                )
            elif role == "tool":
                converted.append(
                    ToolMessage(
                        content=message["content"],
                        tool_call_id=message.get("tool_call_id", ""),
                    )
                )

        response = self._client(temperature, tools).invoke(converted)
        text = response.content if isinstance(response.content, str) else str(response.content)
        calls = [
            ToolInvocation(
                id=call.get("id") or f"call_{index}",
                name=call["name"],
                arguments=call.get("args") or {},
            )
            for index, call in enumerate(getattr(response, "tool_calls", None) or [])
        ]
        return ChatResult(
            text=text,
            tool_calls=calls,
            model=self.model,
            provider=self.name,
            stop_reason="tool_use" if calls else "end_turn",
            usage=getattr(response, "usage_metadata", None) or {},
        )


# ---------------------------------------------------------------------------
# Anthropic — opt-in
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """Claude through the official ``anthropic`` SDK.

    Prompt caching is applied to the system block, which is where the retrieval
    contract lives: it is long, it is byte-identical across every request, and
    it sits ahead of everything volatile.  Caching is a prefix match, so the
    retrieved excerpts go in the *messages*, never in the system prompt — moving
    them there would invalidate the cache on every single question.
    """

    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.anthropic_model
        self._api_key = settings.anthropic_api_key
        self._enable_fallbacks = settings.anthropic_refusal_fallbacks

    def _client(self):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "The Anthropic provider needs the 'anthropic' extra: "
                "pip install -e '.[anthropic]'"
            ) from exc

        #  No api_key argument when the setting is unset: the SDK then resolves
        #  ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
        #  profile on its own, and passing an explicit None defeats that.
        if self._api_key:
            return anthropic.Anthropic(api_key=self._api_key)
        return anthropic.Anthropic()

    @staticmethod
    def _to_blocks(messages: list[dict]) -> list[dict]:
        """Convert the internal message list to Anthropic content blocks.

        Consecutive tool results are gathered into one user message.  The API
        requires every ``tool_result`` for a turn in a single message, and
        splitting them teaches the model to stop making parallel calls.
        """
        out: list[dict] = []
        pending_results: list[dict] = []

        def flush() -> None:
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for message in messages:
            role = message.get("role")
            if role == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": message["content"],
                        **({"is_error": True} if message.get("is_error") else {}),
                    }
                )
                continue

            flush()
            if role == "user":
                out.append({"role": "user", "content": message["content"]})
            elif role == "assistant":
                blocks: list[dict] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message.get("tool_calls", []):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": call["name"],
                            "input": call["arguments"],
                        }
                    )
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
        flush()
        return out

    def send(
        self,
        system: str,
        messages: list[dict],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,  # noqa: ARG002 - see below
    ) -> ChatResult:
        #  temperature is accepted for interface parity and ignored: sampling
        #  parameters are rejected outright on Claude Opus 5 and the rest of the
        #  adaptive-thinking family. Determinism comes from the prompt and from
        #  effort, not from temperature=0.
        client = self._client()

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": self._to_blocks(messages),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        if tools:
            request["tools"] = [tool.as_anthropic() for tool in tools]

        if self._enable_fallbacks:
            response = client.beta.messages.create(
                betas=[ANTHROPIC_FALLBACK_BETA], fallbacks="default", **request
            )
        else:
            response = client.messages.create(**request)

        #  Check how it stopped before reading content: a refusal is an HTTP 200
        #  with a stop_reason, and treating it as an answer would file the
        #  model's decline as a retrieval result.
        refused = getattr(response, "stop_reason", None) == "refusal"
        category = None
        if refused and getattr(response, "stop_details", None) is not None:
            category = getattr(response.stop_details, "category", None)

        text_parts: list[str] = []
        calls: list[ToolInvocation] = []
        for block in response.content or []:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolInvocation(id=block.id, name=block.name, arguments=dict(block.input)))

        usage = getattr(response, "usage", None)
        return ChatResult(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            model=getattr(response, "model", self.model),
            provider=self.name,
            stop_reason=getattr(response, "stop_reason", None),
            refused=refused,
            refusal_category=category,
            usage={
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            }
            if usage
            else {},
        )


def get_provider(provider: str | None = None, model: str | None = None) -> ChatProvider:
    """Build the configured provider.

    Defaults to Ollama.  Choosing Anthropic is logged at WARNING rather than
    INFO on purpose: it is the moment unpublished process documentation starts
    leaving the machine, and that belongs in the log at a level someone reads.
    """
    settings = get_settings()
    name = (provider or settings.rag_llm_provider or "ollama").strip().lower()

    if name == "ollama":
        return OllamaProvider(model)
    if name == "anthropic":
        logger.warning(
            "RAG is using the Anthropic provider (%s). Retrieved excerpts from the synthesis "
            "corpus — including CNMS user documents — are sent to the Anthropic API. Set "
            "RAG_LLM_PROVIDER=ollama to keep everything on this machine.",
            model or settings.anthropic_model,
        )
        return AnthropicProvider(model)

    raise ValueError(
        f"Unknown RAG_LLM_PROVIDER {name!r}. Supported: 'ollama' (local, default), 'anthropic'."
    )


def serialise_tool_result(result: Any) -> str:
    """Render a tool result as the text a model reads.

    JSON rather than a repr, and never truncated silently: a tool result that
    has been cut in half mid-number is worse than one that is long.
    """
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, default=str, sort_keys=False)
