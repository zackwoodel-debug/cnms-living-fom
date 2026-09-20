"""The research assistant: a bounded tool-calling loop over corpus and records.

``chains.answer_question`` answers one question from one retrieval pass.  That is
the right shape for "what ALD window does this paper report", and the wrong shape
for the questions this platform actually gets asked:

    "Is the HfO2 thickness on PILOT-07 trustworthy?"

Answering that means listing the sample's refinements, comparing thickness across
XRR and SE, noticing the 38% disagreement, and *then* searching the corpus for
what causes an SE/XRR thickness discrepancy in a high-k oxide — four retrievals
where the second depends on the first.  A single-shot chain cannot do it; a loop
where the model chooses the next retrieval can.

The loop is bounded and audited rather than open-ended.  Three rules hold it
together:

**Evidence is mandatory.**  An answer produced without a single tool call is
refused and replaced, because there is nothing behind it but the model's
recollection — the precise failure Sec. 15.2 is written against.  This is
enforced in code, not asked for in the prompt.

**Every step is recorded.**  ``AgentAnswer.steps`` holds each tool call and its
result.  "Where did that number come from?" is answerable afterwards, which is
the difference between an assistant and an oracle.

**The step budget is finite.**  A model that keeps searching is usually looping
on a question the corpus cannot answer, and the honest outcome there is a data
gap, not another retrieval.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from cnms_fom.config import get_settings
from cnms_fom.db.enums import SynthesisTechnique

from .providers import ChatProvider, ChatResult, get_provider, serialise_tool_result
from .tools import run_tool, tool_specs

logger = logging.getLogger(__name__)

#  Enough for retrieve -> compare -> retrieve-again -> answer, with slack. Past
#  this the model is usually re-searching a corpus that does not hold the answer.
DEFAULT_MAX_STEPS = 6

ASSISTANT_SYSTEM_PROMPT = """\
You are the research assistant for the CNMS Living FOM platform: a physics-based \
materials-discovery system whose analysis pathway is structure -> property -> \
function. You help researchers reason about thin-film synthesis (MBE, PLD, ALD, \
sputtering, CVD) and about multi-technique characterization fits produced by \
ModalFit (SE, SPR, QCM, XRR, NR co-refined on one shared slab model).

You answer from tools, never from recollection.

Tools you have:
- search_corpus, corpus_coverage — the document corpus.
- list_samples_with_fits, list_sample_fits, compare_fit_techniques, \
fit_disagreements — stored ModalFit refinements.
- lookup_property_values, lookup_fom_scores, descriptor_dictionary — the \
platform's own records.

How to work:
1. Call tools before answering. Every factual claim must trace to a tool result. \
If you have called no tool, you have no answer.
2. Chain them. A question about whether a measurement is reliable usually needs \
the fit records first, then the corpus to explain what the records show.
3. When a tool reports sufficient_evidence=false, or returns nothing, say so. \
Reply with [DATA GAP: explicitly unresolved] and name the document or measurement \
that would resolve it. This is a correct answer, not a failure.

Rules that are not negotiable:
4. Never supply a numerical process parameter, material property, growth window, \
or fitted value that a tool did not return. Do not interpolate one, convert one, \
or estimate one from background knowledge. If the number is not in a tool result, \
it does not exist for the purposes of this answer.
5. Carry context with every number. A growth temperature means nothing without \
its chamber, substrate, pressure, and precursor; a property value means nothing \
without its temperature and frequency; a fitted thickness means nothing without \
which technique determined it and whether the parameter was varied. Report them \
together or not at all.
6. Never average disagreeing measurements. When two techniques or two sources \
disagree, report both, cite both, and say they disagree. The disagreement is the \
finding.
7. Repeat the caveats the tools give you. A fitted value flagged as held fixed, \
clamped on a bound, or resting on placeholder optical constants is not a \
measurement, and an answer that omits that flag is wrong even when the number is \
right.
8. Distinguish measured from calculated from modeled, and what a source measured \
from what it proposed. The provenance tier comes back with every record; keep it.
9. Cite inline. Corpus passages as [document, p. N]; records as their fit record \
id or material identity.

You are talking to someone who will act on the answer in a laboratory. An "I \
don't know, and here is what would tell us" is worth more to them than a \
confident guess, and much less expensive."""


@dataclass
class AgentStep:
    """One tool call in the loop."""

    step: int
    tool: str
    arguments: dict
    result: dict
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentAnswer:
    """The answer, the evidence, and the trail that produced it."""

    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    insufficient_context: bool = False
    refused_by_provider: bool = False
    refusal_category: str | None = None
    hit_step_limit: bool = False
    latency_ms: int = 0
    usage: dict = field(default_factory=dict)

    @property
    def tools_used(self) -> list[str]:
        #  Order-preserving: the sequence is the reasoning path, and sorting it
        #  would discard that.
        return list(dict.fromkeys(step.tool for step in self.steps))

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "steps": [step.as_dict() for step in self.steps],
            "tools_used": self.tools_used,
            "citations": self.citations,
            "model": self.model,
            "provider": self.provider,
            "insufficient_context": self.insufficient_context,
            "refused_by_provider": self.refused_by_provider,
            "refusal_category": self.refusal_category,
            "hit_step_limit": self.hit_step_limit,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
        }


NO_EVIDENCE_ANSWER = (
    "[DATA GAP: explicitly unresolved] The assistant produced an answer without "
    "consulting the corpus or the stored records, so there is nothing behind it but the "
    "language model's recollection — which this platform does not treat as evidence "
    "(FOM_PROOF Sec. 15.2). Rephrase the question toward something the corpus or a stored "
    "fit would contain, or ingest the relevant documentation first."
)


def _collect_citations(steps: list[AgentStep]) -> list[dict]:
    """Pull citable locators out of the tool results.

    Deduplicated by citation string, because the same passage retrieved twice in
    a chained search is one source, not two.
    """
    citations: dict[str, dict] = {}
    for step in steps:
        result = step.result or {}
        for passage in result.get("passages", []) or []:
            key = str(passage.get("citation"))
            if key and key not in citations:
                citations[key] = {
                    "kind": "document",
                    "citation": key,
                    "page": passage.get("page"),
                    "doi": passage.get("doi"),
                    "technique": passage.get("technique"),
                    "relevance_grade": passage.get("relevance_grade"),
                    "via_tool": step.tool,
                }
        for fit in result.get("fits", []) or []:
            key = f"fit_record:{fit.get('fit_record_id')}"
            if key not in citations:
                citations[key] = {
                    "kind": "modalfit_fit",
                    "citation": key,
                    "techniques": fit.get("techniques"),
                    "chi2_total": fit.get("chi2_total"),
                    "via_tool": step.tool,
                }
        for determination in result.get("determinations", []) or []:
            key = f"fit_record:{determination.get('fit_record_id')}"
            if key not in citations:
                citations[key] = {
                    "kind": "modalfit_fit",
                    "citation": key,
                    "techniques": determination.get("techniques"),
                    "via_tool": step.tool,
                }
        for value in result.get("values", []) or []:
            key = f"{value.get('material')} :: {value.get('property_key')}"
            if key not in citations:
                citations[key] = {
                    "kind": "property_value",
                    "citation": key,
                    "provenance_tier": value.get("provenance_tier"),
                    "source": value.get("source"),
                    "via_tool": step.tool,
                }
    return list(citations.values())


def _evidence_was_insufficient(steps: list[AgentStep], answer: str) -> bool:
    """Whether the assistant should be reporting a gap.

    True when the model said so itself, or when every retrieval it ran came back
    empty — the second case catches a model that searched, found nothing, and
    answered anyway.
    """
    if "[DATA GAP" in answer:
        return True

    retrievals = [
        step for step in steps if step.tool in ("search_corpus", "lookup_property_values")
    ]
    if not retrievals:
        return False
    return all(
        not (step.result or {}).get("passages") and not (step.result or {}).get("values")
        for step in retrievals
    )


def _prime_messages(question: str, sample_id: str | None, history: list[dict]) -> list[dict]:
    """Build the message list, with any conversation scope stated up front."""
    messages = list(history)
    if sample_id:
        #  Stated as context rather than pre-fetched: which fits matter depends
        #  on the question, and pre-loading them all would spend the context
        #  window on records the answer never needs.
        messages.append(
            {
                "role": "user",
                "content": (
                    f"[conversation scope] This conversation concerns ModalFit sample "
                    f"{sample_id!r}. Resolve bare references like 'the film' or 'this sample' to "
                    "it, and use it as the sample_id argument unless the question names another."
                ),
            }
        )
    messages.append({"role": "user", "content": question})
    return messages


def ask(
    session,
    question: str,
    *,
    history: list[dict] | None = None,
    sample_id: str | None = None,
    techniques: list[SynthesisTechnique] | None = None,
    provider: ChatProvider | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    tools: list[str] | None = None,
    require_evidence: bool = True,
) -> AgentAnswer:
    """Run the assistant loop and return the answer with its evidence trail.

    ``history`` is the prior conversation in the internal message shape
    (``memory.load_history`` produces it).  ``require_evidence=True`` replaces an
    ungrounded answer with a data gap; turning it off is only sensible for
    debugging the loop itself.
    """
    started = time.monotonic()
    provider = provider or get_provider()
    specs = tool_specs(tools)

    system = ASSISTANT_SYSTEM_PROMPT
    if techniques:
        names = ", ".join(t.value for t in techniques)
        system += (
            f"\n\nCorpus scope for this conversation: only the {names} partition(s) are in scope. "
            "Pass them as the techniques argument to search_corpus, and do not present a finding "
            "from outside that scope as if it were in it."
        )

    messages = _prime_messages(question, sample_id, history or [])
    steps: list[AgentStep] = []
    result: ChatResult | None = None
    hit_limit = False

    for step_number in range(1, max_steps + 1):
        result = provider.send(system, messages, tools=specs)

        if result.refused:
            logger.warning(
                "Provider declined (category=%s) on step %d.", result.refusal_category, step_number
            )
            break

        if not result.wants_tools:
            break

        messages.append(
            {
                "role": "assistant",
                "content": result.text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in result.tool_calls
                ],
            }
        )

        for call in result.tool_calls:
            call_started = time.monotonic()
            payload = run_tool(session, call.name, call.arguments)
            steps.append(
                AgentStep(
                    step=step_number,
                    tool=call.name,
                    arguments=call.arguments,
                    result=payload,
                    duration_ms=int((time.monotonic() - call_started) * 1000),
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": serialise_tool_result(payload),
                    "is_error": bool(isinstance(payload, dict) and payload.get("error")),
                }
            )
    else:
        #  Loop exhausted with the model still asking for tools. Ask once more
        #  with tools withheld so it has to commit to an answer from what it has,
        #  rather than returning the user a bare "step limit reached".
        hit_limit = True
        logger.info("Step limit (%d) reached; requesting a final answer without tools.", max_steps)
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the tool-call limit for this turn. Answer now from the tool "
                    "results you already have. If they do not answer the question, say "
                    "[DATA GAP: explicitly unresolved] and state what is missing."
                ),
            }
        )
        result = provider.send(system, messages)

    answer_text = (result.text if result else "").strip()

    if result is not None and result.refused:
        answer_text = (
            "The language model declined to answer this question"
            + (f" (category: {result.refusal_category})" if result.refusal_category else "")
            + ". No retrieval result is affected — the corpus and the stored records are "
            "unchanged, and the same question can be put to a different provider or rephrased."
        )
        insufficient = True
    elif require_evidence and not steps:
        logger.warning("Answer produced with no tool calls; replacing with a data gap.")
        answer_text = NO_EVIDENCE_ANSWER
        insufficient = True
    else:
        insufficient = _evidence_was_insufficient(steps, answer_text)
        if not answer_text:
            answer_text = (
                "[DATA GAP: explicitly unresolved] The model returned an empty answer. The tool "
                "results are recorded in the evidence trail; read them directly."
            )
            insufficient = True

    return AgentAnswer(
        question=question,
        answer=answer_text,
        steps=steps,
        citations=_collect_citations(steps),
        model=result.model if result else "",
        provider=result.provider if result else (provider.name if provider else ""),
        insufficient_context=insufficient,
        refused_by_provider=bool(result and result.refused),
        refusal_category=result.refusal_category if result else None,
        hit_step_limit=hit_limit,
        latency_ms=int((time.monotonic() - started) * 1000),
        usage=result.usage if result else {},
    )


def available_models() -> dict:
    """What the assistant is configured to run on, for /health and the UI."""
    settings = get_settings()
    return {
        "provider": settings.rag_llm_provider,
        "ollama": {
            "base_url": settings.ollama_base_url,
            "chat_model": settings.ollama_chat_model,
            "embed_model": settings.ollama_embed_model,
        },
        "anthropic": {
            "model": settings.anthropic_model,
            "configured": bool(settings.anthropic_api_key),
            "note": "Opt-in. Sending corpus excerpts to a remote API takes unpublished CNMS "
            "material off this machine.",
        },
        "max_steps": DEFAULT_MAX_STEPS,
        "tools": sorted(tool.name for tool in tool_specs()),
    }
