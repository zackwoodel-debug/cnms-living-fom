"""Assembling a research brief: what the evidence says, and what it does not.

The order of operations is the design.

1. **The campaign snapshot runs first, and without a model.** Its warnings — an
   unapproved objective, a stalled search, suggestions piled on a bound — are
   attached to the brief unconditionally. A warning that depended on the model
   noticing it would not be a guardrail.
2. **Reviewed cards are read before the corpus.** A card is the integrated version
   of work already done, so starting there is how knowledge compounds instead of
   being re-derived. Proposed and stale cards are read too, and every one used
   earns a warning naming it.
3. **The corpus fills the gaps the cards leave**, through the same hybrid
   retrieval, grading and corrective-rewrite path the assistant uses.
4. **Extraction is per passage**, so every claim is attributable to one page.
5. **Contradictions are found arithmetically**, never by asking a model whether two
   numbers conflict.
6. **Abstention is a first-class outcome.** A brief with no usable evidence is a
   successful brief that says so and names what would resolve it.

Nothing here writes to a scientific table. Generating a brief reads the corpus, the
cards, the fits, and the campaign, and returns a document. The only thing it can
cause is a person deciding to act.
"""

from __future__ import annotations

import logging
import re
import time

from cnms_fom.db.enums import CardStatus, StatementKind, SynthesisTechnique
from cnms_fom.research.campaign import CampaignSnapshot, snapshot
from cnms_fom.research.contracts import (
    DataGap,
    EvidenceItem,
    ExtractedClaim,
    LabelledStatement,
    ResearchBrief,
)
from cnms_fom.research.extract import extract_claims, extraction_provider, find_contradictions
from cnms_fom.research.policy import BASELINE, ResearchPolicy

logger = logging.getLogger(__name__)

INTERPRETATION_SYSTEM_PROMPT = """\
You are writing the interpretation section of a research brief for a materials \
scientist. The evidence has already been gathered and the numbers already \
extracted; you are not retrieving anything and not adding any number that is not \
in front of you.

You will be given: the research question, the extracted claims with their context \
and citations, any contradictions found, the data gaps, and the state of the \
Bayesian-optimization campaign with its warnings.

Produce JSON only:

{
  "statements": [
    {"kind": "evidence" | "interpretation" | "proposal", "text": "..."}
  ],
  "proposed_actions": ["..."]
}

Label every statement:
  evidence        restates something in the claims, records, or warnings above. \
Cite it as [document, p. N] or as the record id.
  interpretation  your reading of that evidence. Say what it rests on.
  proposal        something to do next. Say what it assumes.

Rules:
1. Introduce no number that is not in the material above. Not a typical value, not \
a converted one, not one you recall. If a number would help and is absent, that is \
a data gap and it is already listed.
2. Carry context with every number you restate. A growth temperature without its \
chamber and precursor is not a result.
3. Never average disagreeing values. Where a contradiction is listed, report both \
and say what differs between them.
4. Repeat the campaign warnings that bear on the question. An unapproved objective \
means the ranking is not yet a result, and saying so is more useful than a ranking.
5. Prefer the shortest experiment that would discriminate between two explanations \
over the most thorough one.
6. Say when you would not act. "One determination, so I would measure it a second \
way before building on it" is a useful answer.
7. If the evidence does not support an answer, say so plainly and propose what \
would."""


def generate_brief(
    db,
    research_question: str,
    *,
    bo_run_id: int | None = None,
    experiment_id: int | None = None,
    material: str | None = None,
    sample_id: str | None = None,
    target_property: str | None = None,
    provider=None,
    policy: ResearchPolicy | None = None,
    include_cards: bool = True,
    interpret: bool = True,
    cache_db=None,
) -> ResearchBrief:
    """Build an audited, read-only brief for one question.

    ``cache_db`` is where the per-passage model-call cache lives, defaulting to ``db``.
    They separate for the benchmark, which builds a throwaway corpus database per run
    but wants the cache to outlive it — the cache is content-addressed, so sharing it
    across runs is safe by construction and is the whole reason a policy sweep is
    affordable.

    ``provider`` is any :class:`rag_backend.providers.ChatProvider`; a scripted one
    makes this whole path testable without a model server. ``interpret=False``
    assembles the evidence and skips the narrative, which is what the benchmark
    scores — the numbers and the abstention decision are what a policy changes, and
    the prose is not.
    """
    started = time.monotonic()
    policy = policy or BASELINE
    tool_calls: list[dict] = []
    warnings: list[str] = []

    #  1. The campaign, deterministically.
    snap: CampaignSnapshot | None = None
    if bo_run_id is not None:
        snap = snapshot(db, bo_run_id)
        warnings.extend(snap.warnings)
        tool_calls.append(
            {"tool": "campaign_snapshot", "arguments": {"bo_run_id": bo_run_id},
             "result": snap.as_dict()}
        )

    #  2. Cards before the corpus.
    card_evidence: list[EvidenceItem] = []
    if include_cards:
        card_evidence, card_warnings, card_trace = _read_cards(db, research_question)
        warnings.extend(card_warnings)
        tool_calls.extend(card_trace)

    #  3. The corpus, for what the cards do not settle.
    passages, retrieval_trace, retrieval_warnings = _retrieve(
        db, research_question, policy=policy, provider=provider, cache_db=cache_db
    )
    warnings.extend(retrieval_warnings)
    tool_calls.append(retrieval_trace)

    #  3b. The platform's own records: fits, their plausibility, stored properties.
    #  Read through the assistant's typed tools rather than fresh queries, so a
    #  brief sees exactly what the chat path sees and the two cannot drift.
    record_warnings, record_trace = _read_records(
        db, sample_id=sample_id, material=material, target_property=target_property
    )
    warnings.extend(record_warnings)
    tool_calls.extend(record_trace)

    #  4. Extraction, per passage.
    claims: list[ExtractedClaim] = []
    if passages and provider is not None:
        extractor = extraction_provider(provider, policy)
        extraction_record: dict = {}
        claims, extraction_problems = extract_claims(
            extractor, passages, policy=policy,
            db=cache_db if cache_db is not None else db, record=extraction_record,
        )
        warnings.extend(extraction_problems)
        tool_calls.append(
            {"tool": "extract_claims",
             "arguments": {"n_passages": len(passages), "model": extractor.model},
             "result": {"n_claims": len(claims), "problems": extraction_problems,
                        **extraction_record}}
        )

    #  What this brief actually cost in model calls, and what the cache saved.
    #  Surfaced because grading and extraction are essentially the whole cost, and a
    #  latency number with no call count behind it cannot be acted on.
    tool_calls.append({
        "tool": "cost_report",
        "arguments": {},
        "result": _cost_report(tool_calls),
    })

    #  5. Contradictions, arithmetically.
    contradictions = find_contradictions(claims)
    if contradictions:
        warnings.append(
            f"{len(contradictions)} contradiction(s) between sources were found and are reported "
            "separately. FOM_PROOF Sec. 2.1 forbids merging records without a declared "
            "aggregation rule, so neither value has been preferred and no average was taken."
        )

    #  Claim-level context completeness. Sec. 16: a value whose context is unknown
    #  is not a usable value, and saying which field is absent is the actionable part.
    for claim in claims:
        if claim.missing_context:
            warnings.append(
                f"The claim {claim.field_name}={claim.value} from "
                f"{claim.evidence[0].citation} is missing required context "
                f"({', '.join(claim.missing_context)}), so it cannot be compared with a stored "
                "value. The source did not state it; nothing was assumed."
            )

    data_gaps = _data_gaps(research_question, passages, claims, policy, target_property)

    #  Did any model call fail outright? A failed grade drops its passage and a failed
    #  extraction yields no claims, so an outage arrives looking like a thin corpus.
    #  Sec. 15.2: an infrastructure failure and a genuine absence of evidence are not
    #  the same finding and must not be reported as one.
    cost = _cost_report(tool_calls)
    n_failed = cost["grading_failed"] + cost["extraction_failed"]
    degraded_reason = ""
    if n_failed:
        parts = []
        if cost["grading_failed"]:
            parts.append(f"{cost['grading_failed']} grading call(s)")
        if cost["extraction_failed"]:
            parts.append(f"{cost['extraction_failed']} extraction call(s)")
        degraded_reason = (
            f"{' and '.join(parts)} failed to reach the model. A failed grade drops its "
            "passage and a failed extraction yields no claims, so this brief saw less of "
            "the corpus than the policy asked for. Absence of evidence here is not "
            "evidence of absence: re-run once the model is reachable before treating any "
            "gap below as a real gap."
        )
        warnings.append(degraded_reason)

    abstained = _should_abstain(
        passages, claims, policy, extraction_ran=bool(passages and provider is not None)
    )
    if abstained and not degraded_reason:
        warnings.append(
            "This brief abstains: the evidence did not clear the policy's threshold, so no "
            "interpretation was generated. The data gaps below are the actionable output."
        )
    elif abstained:
        warnings.append(
            "This brief abstains, but the model calls above failed, so it abstained for want "
            "of a working model rather than for want of evidence. This is not a finding."
        )

    brief = ResearchBrief(
        research_question=research_question,
        bo_run_id=bo_run_id,
        experiment_id=experiment_id,
        material=material,
        target_property=target_property,
        fom_definition=snap.fom_definition if snap else None,
        evidence=[*card_evidence, *passages],
        claims=claims,
        contradictions=contradictions,
        data_gaps=data_gaps,
        warnings=warnings,
        model=getattr(provider, "model", "") or "",
        provider=getattr(provider, "name", "") or "",
        policy_version=policy.version,
        tool_calls=tool_calls,
        abstained=abstained,
        degraded_reason=degraded_reason,
    )

    #  6. Interpretation, last, and only over what is already assembled.
    if interpret and provider is not None and not abstained:
        statements, actions, problem = _interpret(provider, brief, snap)
        brief.statements = statements
        brief.proposed_actions = actions
        if problem:
            brief.warnings.append(problem)

    logger.info(
        "Brief for %r: %d evidence, %d claims, %d contradictions, %d gaps, %d warnings, "
        "abstained=%s, %d ms",
        research_question[:60], len(brief.evidence), len(claims), len(contradictions),
        len(data_gaps), len(brief.warnings), abstained,
        int((time.monotonic() - started) * 1000),
    )
    return brief


def _cost_report(tool_calls: list[dict]) -> dict:
    """Model calls made and avoided, summed over the stages that make them."""
    grading_calls = grading_hits = grading_failed = 0
    for call in tool_calls:
        if call.get("tool") != "retrieve":
            continue
        for attempt in (call.get("result") or {}).get("attempts") or []:
            grading_calls += int(attempt.get("grader_calls") or 0)
            grading_hits += int(attempt.get("grader_cache_hits") or 0)
            grading_failed += int(attempt.get("grader_failed") or 0)

    extraction = next(
        (c for c in tool_calls if c.get("tool") == "extract_claims"), {"arguments": {}}
    )
    passages = int((extraction.get("arguments") or {}).get("n_passages") or 0)
    extraction_failed = int((extraction.get("result") or {}).get("extraction_failed") or 0)

    return {
        "grading_calls": grading_calls,
        "grading_cache_hits": grading_hits,
        "grading_failed": grading_failed,
        "passages_extracted": passages,
        "extraction_failed": extraction_failed,
        "note": (
            "Grading and extraction call a model once per passage and are together nearly the "
            "whole cost of a brief; retrieval itself is about a second. Cache hits are calls that "
            "did not happen — extraction is cached on the passage alone, so a passage is extracted "
            "once ever."
        ),
    }


def _read_cards(db, question: str) -> tuple[list[EvidenceItem], list[str], list[dict]]:
    """Reviewed cards as evidence; proposed and stale ones as flagged context."""
    from cnms_fom.knowledge.cards import search_cards

    result = search_cards(db, question, limit=10)
    evidence: list[EvidenceItem] = []
    warnings: list[str] = []

    for card in result.get("cards", []):
        slug = card.get("slug", "")
        if card.get("citable"):
            #  A reviewed card's own sources are its provenance; the card is a
            #  locator into them, so it is recorded as evidence with the card as
            #  the document and its review as the warrant.
            evidence.append(
                EvidenceItem(
                    document_id=None,
                    document_title=f"card:{slug}",
                    page=None,
                    quote=(card.get("summary") or card.get("title") or slug)[:2000],
                    retrieval_method="card",
                    grade=3,
                    grade_reason=f"reviewed by {card.get('reviewed_by')}",
                )
            )
            continue

        if card.get("status") == "proposed":
            warnings.append(
                f"Card {slug!r} matched this question but is PROPOSED — an unreviewed draft "
                f"written by {card.get('authored_by')}. It is not treated as evidence and nothing "
                "in this brief rests on it."
            )
        elif card.get("review_is_stale"):
            warnings.append(
                f"Card {slug!r} was reviewed by {card.get('reviewed_by')} but has been edited "
                "since, so the review no longer covers what it says. It is not treated as "
                "evidence."
            )
        elif card.get("status") == CardStatus.SUPERSEDED.value:
            warnings.append(f"Card {slug!r} has been superseded and was not used.")

    trace = [{
        "tool": "search_cards",
        "arguments": {"query": question},
        "result": {
            "n_cards": result.get("n_cards", 0),
            "n_citable": len(evidence),
            "cards": [
                {"slug": c.get("slug"), "status": c.get("status"), "citable": c.get("citable")}
                for c in result.get("cards", [])
            ],
        },
    }]
    return evidence, warnings, trace


def _read_records(
    db, *, sample_id: str | None, material: str | None, target_property: str | None
) -> tuple[list[str], list[dict]]:
    """Read the fits, their plausibility, and any stored property values.

    Everything here is instrument-derived or human-entered, so it is a different
    kind of evidence from a literature claim and is kept that way: these go into the
    tool trace and into warnings, never into ``claims``. A fitted thickness is not a
    thing a paper said.
    """
    from cnms_fom.rag_backend.tools import run_tool

    warnings: list[str] = []
    trace: list[dict] = []

    if sample_id:
        fits = run_tool(db, "list_sample_fits", {"sample_id": sample_id})
        trace.append({"tool": "list_sample_fits", "arguments": {"sample_id": sample_id},
                      "result": fits})

        disagreements = run_tool(db, "fit_disagreements", {"sample_id": sample_id})
        trace.append({"tool": "fit_disagreements", "arguments": {"sample_id": sample_id},
                      "result": disagreements})
        for parameter in disagreements.get("disagreements") or []:
            comparison = (disagreements.get("comparisons") or {}).get(parameter, {})
            warnings.append(
                f"Techniques disagree on {parameter} for sample {sample_id!r}: "
                f"{comparison.get('summary', 'see the fit records')} Both determinations are kept; "
                "nothing was averaged (Sec. 2.1)."
            )

        plausibility = run_tool(db, "check_fit_plausibility", {"sample_id": sample_id})
        trace.append({"tool": "check_fit_plausibility", "arguments": {"sample_id": sample_id},
                      "result": plausibility})
        for layer in plausibility.get("layers") or []:
            for finding in layer.get("violations") or []:
                warnings.append(
                    f"A fitted value for sample {sample_id!r} is physically impossible "
                    f"(fit #{layer.get('fit_record_id')}, {layer.get('layer')}): "
                    f"{finding.get('message')}"
                )
            for finding in layer.get("inconsistencies") or []:
                warnings.append(
                    f"A fit for sample {sample_id!r} disagrees with itself "
                    f"(fit #{layer.get('fit_record_id')}, {layer.get('layer')}): "
                    f"{finding.get('message')}"
                )

        #  A clamped or held-fixed parameter is not a measurement. Read from the
        #  stored bounds rather than from the fit's prose: matching words in
        #  generated text is how a detector stops working when the wording improves.
        from cnms_fom.modalfit.compare import fit_process_warnings, fits_for_sample

        for record in fits_for_sample(db, sample_id):
            for finding in fit_process_warnings(record):
                warnings.append(f"Fit #{record.id} on sample {sample_id!r}: {finding}")

    if material:
        properties = run_tool(
            db, "lookup_property_values",
            {"formula": material, **({"property_key": target_property} if target_property else {})},
        )
        trace.append({"tool": "lookup_property_values",
                      "arguments": {"formula": material, "property_key": target_property},
                      "result": properties})

    return warnings, trace


def _retrieve(
    db, question: str, *, policy: ResearchPolicy, provider, cache_db=None
) -> tuple[list[EvidenceItem], dict, list[str]]:
    """Hybrid retrieval through the assistant's own path, as evidence items."""
    from cnms_fom.rag_backend.grading import retrieve_with_correction

    techniques = None
    if policy.techniques:
        techniques = []
        for name in policy.techniques:
            try:
                techniques.append(SynthesisTechnique(name))
            except ValueError:
                logger.warning("Policy names an unknown technique %r; ignored.", name)

    warnings: list[str] = []
    try:
        outcome = retrieve_with_correction(
            db,
            question,
            k=policy.top_k,
            techniques=techniques or None,
            min_similarity=policy.min_similarity,
            provider=provider,
            grade=policy.grade,
            allow_rewrite=policy.allow_query_rewrite,
            depth=policy.candidate_depth,
            use_vector=policy.use_dense,
            use_lexical=policy.use_lexical,
            cache_db=cache_db,
        )
    except ImportError as exc:
        #  The rag extra is absent. A brief can still be assembled from cards and
        #  records, and saying retrieval did not run is very different from saying
        #  the corpus is empty.
        warnings.append(
            f"Corpus retrieval did not run: {exc}. This brief covers only the cards and records, "
            "and its data gaps must not be read as gaps in the literature."
        )
        return [], {"tool": "retrieve", "arguments": {"query": question},
                    "result": {"error": str(exc)}}, warnings
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            f"Corpus retrieval failed: {exc}. As above, the gaps below are not evidence that the "
            "literature is silent."
        )
        return [], {"tool": "retrieve", "arguments": {"query": question},
                    "result": {"error": str(exc)}}, warnings

    useful = outcome.useful if policy.grade else outcome.hits
    useful = _reweight_structured(useful, policy)
    evidence = [
        EvidenceItem(
            document_id=hit.fused.hit.document_id,
            document_title=hit.fused.hit.document_title,
            page=hit.fused.hit.page,
            quote=hit.fused.hit.text,
            chunk_id=hit.fused.hit.chunk_id,
            doi=hit.fused.hit.doi,
            source_url=hit.fused.hit.source_url,
            technique=hit.fused.hit.technique,
            retrieval_method="+".join(
                name
                for name, rank in (
                    ("vector", hit.fused.vector_rank),
                    ("lexical", hit.fused.lexical_rank),
                )
                if rank is not None
            ) or "unknown",
            retrieval_rank=index,
            grade=hit.grade,
            grade_reason=hit.reason,
        )
        for index, hit in enumerate(useful, start=1)
    ]

    if outcome.rewritten:
        warnings.append(
            f"The first retrieval found nothing useful, so the query was rewritten once to "
            f"{outcome.effective_query!r}. The evidence below answers the rewritten query."
        )
    unlocatable = [item.citation for item in evidence if not item.is_locatable]
    if unlocatable:
        warnings.append(
            f"{len(unlocatable)} retrieved passage(s) have no page number, so a claim taken from "
            "them could not be checked against a source: " + ", ".join(unlocatable[:3])
        )

    trace = {
        "tool": "retrieve",
        "arguments": {
            "query": question,
            "policy": policy.version,
            "grade": policy.grade,
            "depth": policy.candidate_depth,
            "top_k": policy.top_k,
        },
        "result": {
            "attempts": outcome.attempts,
            "effective_query": outcome.effective_query,
            "rewritten": outcome.rewritten,
            "n_useful": len(useful),
            "sufficient": outcome.sufficient,
        },
    }
    return evidence, trace, warnings


#  Markers of a tabular or caption passage. Process parameters live in tables far
#  more often than in prose, and a ranking that treats them alike buries them — the
#  benchmark's PLD case exists to measure exactly that.
_TABLE_MARKERS = ("|", "\t")
_CAPTION_PATTERN = re.compile(r"\b(table|figure|fig\.)\s*\d+", re.IGNORECASE)
#  Fraction of whitespace-separated tokens that are numeric, above which a passage
#  reads as tabular whatever punctuation it uses.
_NUMERIC_TOKEN_FRACTION = 0.25


def _looks_structured(text: str) -> tuple[bool, bool]:
    """Whether a passage looks tabular, and whether it looks like a caption."""
    if not text:
        return False, False
    caption = bool(_CAPTION_PATTERN.search(text))
    if any(marker in text for marker in _TABLE_MARKERS):
        return True, caption

    tokens = text.split()
    if len(tokens) >= 8:
        numeric = sum(
            1 for token in tokens if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", token)
        )
        if numeric / len(tokens) >= _NUMERIC_TOKEN_FRACTION:
            return True, caption
    return False, caption


def _reweight_structured(hits: list, policy: ResearchPolicy) -> list:
    """Reorder graded hits by the policy's table and caption preference.

    Applied after grading rather than before, so a weight can promote a passage the
    grader already accepted and can never smuggle in one it rejected. With both
    weights at 1.0 — the default — the order is returned untouched, so the knob
    genuinely does nothing until it is asked to.
    """
    if policy.table_weight == 1.0 and policy.caption_weight == 1.0:
        return hits

    def weighted(index_and_hit) -> tuple:
        index, hit = index_and_hit
        tabular, caption = _looks_structured(hit.fused.hit.text)
        weight = 1.0
        if tabular:
            weight *= policy.table_weight
        if caption:
            weight *= policy.caption_weight
        #  Grade first, then the weighted fusion score. A weight reorders within a
        #  grade; it does not let a grade-2 passage overtake a grade-3 one, because
        #  relevance is a judgement and this is a formatting preference.
        return (hit.grade or 0, hit.fused.rrf_score * weight, -index)

    return [hit for _, hit in sorted(enumerate(hits), key=weighted, reverse=True)]


def _data_gaps(
    question: str,
    passages: list[EvidenceItem],
    claims: list[ExtractedClaim],
    policy: ResearchPolicy,
    target_property: str | None,
) -> list[DataGap]:
    """What was asked for and is not here.

    The gap list is the actionable half of a brief: it is the reading list, and it
    is what distinguishes "we do not know" from "we did not look".
    """
    gaps: list[DataGap] = []
    searched = (
        f"hybrid retrieval under policy {policy.version}, "
        f"{len(passages)} passage(s) cleared grade {policy.min_useful_grade}"
    )

    if not passages:
        gaps.append(DataGap(
            question=question,
            what_was_searched=searched,
            what_would_resolve_it=(
                "Ingest a document that addresses this question, or widen the technique filter. "
                "If the corpus should already contain one, run POST /rag/search with "
                "diagnostics=true to see what each retriever found before concluding it is absent."
            ),
        ))
        return gaps

    if target_property and not any(c.field_name == target_property for c in claims):
        gaps.append(DataGap(
            question=f"What is {target_property} under the conditions asked about?",
            field_name=target_property,
            what_was_searched=searched,
            what_would_resolve_it=(
                f"The retrieved passages discuss the subject but state no value for "
                f"{target_property}. A source that reports it with its measurement context, or a "
                "measurement on this platform, would resolve it."
            ),
        ))

    for claim in claims:
        if claim.missing_context:
            gaps.append(DataGap(
                question=(
                    f"Under what {', '.join(claim.missing_context)} was "
                    f"{claim.field_name}={claim.value} measured?"
                ),
                field_name=claim.field_name,
                what_was_searched=f"the passage at {claim.evidence[0].citation}",
                what_would_resolve_it=(
                    "The source states the value without this context, so the value cannot be "
                    "compared with any other (Sec. 16). The paper's methods section, its "
                    "supplementary information, or a direct measurement would resolve it."
                ),
            ))
    return gaps


def _should_abstain(
    passages: list[EvidenceItem],
    claims: list[ExtractedClaim],
    policy: ResearchPolicy,
    *,
    extraction_ran: bool,
) -> bool:
    """Whether the brief should decline to interpret.

    Abstention is an outcome, not a failure: a brief that says "the corpus does not
    settle this, and here is what would" is more useful than one that reasons over
    two loosely related passages.
    """
    if len(passages) < policy.abstain_below_passages:
        return True
    if policy.abstain_without_page and passages and not any(p.is_locatable for p in passages):
        return True
    #  Passages but no numbers in any of them. ``claims`` was accepted by this function
    #  and never consulted, and this is the case it was needed for: the benchmark's
    #  GaAs-on-Ge question retrieves GaAs-on-GaAs passages, which are close enough to
    #  pass grading, and extraction correctly finds nothing in them that answers the
    #  question. Zero claims and no abstention meant the brief went on to interpret
    #  passages it had extracted nothing from — the exact shape of answering from a
    #  near-miss source. Guarded on ``extraction_ran`` so a brief assembled without a
    #  provider is not read as an abstention about the corpus.
    return extraction_ran and not claims


def _interpret(provider, brief: ResearchBrief, snap: CampaignSnapshot | None):
    """Ask the model to explain what was assembled. Adds no new facts."""
    from cnms_fom.research.extract import _extract_json

    payload = {
        "research_question": brief.research_question,
        "claims": [
            {
                "field": c.field_name,
                "value": c.value,
                "units": c.units,
                "value_text": c.value_text,
                "tier": c.tier.value,
                "context": c.context,
                "missing_context": c.missing_context,
                "citation": c.evidence[0].citation,
            }
            for c in brief.claims
        ],
        "contradictions": [
            {"field": c.field_name, "basis": c.basis,
             "left": {"value": c.left.value, "citation": c.left.evidence[0].citation},
             "right": {"value": c.right.value, "citation": c.right.evidence[0].citation}}
            for c in brief.contradictions
        ],
        "data_gaps": [g.as_dict() for g in brief.data_gaps],
        "campaign": snap.as_dict() if snap else None,
        "warnings": brief.warnings,
    }

    import json as _json

    try:
        result = provider.send(
            INTERPRETATION_SYSTEM_PROMPT,
            [{"role": "user", "content": _json.dumps(payload, indent=2, default=str)}],
        )
    except Exception as exc:  # noqa: BLE001
        return [], [], f"Interpretation failed: {exc}. The evidence above stands on its own."

    if getattr(result, "refused", False):
        return [], [], (
            "The model declined to write the interpretation. The evidence, claims, and gaps are "
            "unaffected."
        )

    parsed = _extract_json(result.text)
    if parsed is None:
        return [], [], (
            "The interpretation reply was not parseable, so no narrative was recorded. The "
            "evidence and claims above are unaffected."
        )

    statements: list[LabelledStatement] = []
    unsupported = 0
    for raw in parsed.get("statements") or []:
        if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
            continue
        try:
            kind = StatementKind(str(raw.get("kind", "interpretation")).strip().lower())
        except ValueError:
            kind = StatementKind.INTERPRETATION
        if kind is StatementKind.EVIDENCE:
            #  An EVIDENCE statement must carry evidence, and the model does not
            #  supply it. Attach the brief's own evidence, which is what it was
            #  shown — and if the brief has none, demote rather than refuse, so a
            #  mislabelled sentence does not discard the whole narrative.
            if not brief.evidence:
                kind = StatementKind.INTERPRETATION
                unsupported += 1
        statements.append(
            LabelledStatement(
                kind=kind,
                text=str(raw["text"]).strip(),
                evidence=list(brief.evidence) if kind is StatementKind.EVIDENCE else [],
            )
        )

    actions = [str(a).strip() for a in (parsed.get("proposed_actions") or []) if str(a).strip()]
    problem = (
        f"{unsupported} statement(s) were labelled as evidence but the brief has none; they were "
        "demoted to interpretation."
        if unsupported
        else ""
    )
    return statements, actions, problem
