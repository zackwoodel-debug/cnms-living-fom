"""Turning graded passages into typed claims, and finding where they disagree.

Two halves with very different reliability, kept apart on purpose.

**Extraction is model work.**  Reading "the growth per cycle was constant at 0.98
angstrom per cycle" and producing ``{field: growth_per_cycle_ang, value: 0.98,
units: A/cycle, context: {...}}`` is what a language model is genuinely good at,
and :func:`extract_claims` asks it to do exactly that and nothing more — no
interpretation, no comparison, no filling in of context the passage does not state.

**Contradiction detection is not.**  :func:`find_contradictions` compares numbers
in Python.  Asking a model whether two values conflict invites it to reconcile
them, and a reconciled contradiction is a lost finding.  So the comparison is
arithmetic, the threshold is explicit, and the differing context is computed by set
difference rather than described in prose.

Every extracted value carries the quote it came from.  An extraction whose page
cannot be checked is worse than no extraction, because it looks identical to a good
one once it is a number in a table (FOM_PROOF Sec. 2.2).
"""

from __future__ import annotations

import json
import logging
import re

from cnms_fom.db.enums import ClaimTier
from cnms_fom.rag_backend import cache
from cnms_fom.research.contracts import (
    CLAIM_CONTEXT_FIELDS,
    Contradiction,
    EvidenceItem,
    ExtractedClaim,
    ResearchContractError,
)
from cnms_fom.research.policy import ResearchPolicy

logger = logging.getLogger(__name__)

#  Relative spread above which two claims for the same field are called a
#  disagreement. Matches modalfit.compare's threshold deliberately: a 10%
#  discrepancy should mean the same thing whether it is between two papers or
#  between two techniques on one sample.
DISAGREEMENT_FRACTION = 0.10

EXTRACTION_SYSTEM_PROMPT = """\
You extract quantitative claims from one passage of a materials-science document. \
You are not answering a question, comparing sources, or interpreting anything.

For each distinct quantity the passage states, emit one object:

  field       the registry key for the quantity. If the quantity appears in this \
list you MUST use its key exactly as written; inventing a descriptive name for a \
listed quantity is an error, because a claim filed under a name nobody searches for \
is the same as a claim you never made.

    k             relative permittivity; dielectric constant; kappa; "k value"
    eps_inf       high-frequency/optical permittivity; epsilon-infinity
    eps_ionic     ionic or lattice contribution to permittivity
    Eg            band gap; optical gap; energy gap
    dEc           conduction band offset
    Ebd           breakdown field; dielectric strength; breakdown voltage per \
thickness
    tan_delta     loss tangent; dielectric loss; dissipation factor
    kappa_th      thermal conductivity
    rho           mass density; film density; "density" of a material
    sld_xray      X-ray scattering length density
    sld_neutron   neutron scattering length density
    growth_per_cycle_ang    growth per cycle; GPC; angstrom or A per cycle
    growth_rate_nm_min      growth or deposition rate per unit time
    thickness_nm            film or layer thickness
    roughness_ang           surface or interface roughness; RMS roughness

Only if the quantity is none of these, invent a descriptive snake_case name. A \
resistivity, a refractive index or a lattice constant has no key above, so a \
descriptive name is right for those.
  value       the number, or null if the claim is not numeric
  units       exactly as the passage writes them, or null. Keep any "per" part: a \
growth per cycle written "1.42 A/cycle" has units "A/cycle", not "angstrom" — \
dropping the per-cycle turns a rate into a length.
  value_text  the claim in words when it is not a single number (a range, a window)
  tier        what the passage says about where the number came from: measured, \
calculated, modeled, fitted, reported (it states it without saying how), or unknown
  context     an object with only the fields the passage actually states (see below)
  quote       the shortest verbatim span from the passage that contains the claim
  confidence  0-1, how sure you are that you read the passage correctly

The context fields, and what counts as the passage stating one:

  material        the substance measured, e.g. HfO2, SrTiO3. A title naming it \
counts. Fill this on every claim about that substance, not only the first.
  polymorph       amorphous, monoclinic, anatase, ...
  specimen_form   thin film, ceramic, single crystal, ...
  technique       how it was grown or measured: ald, pld, mbe, sputtering, cvd, \
xrr, ellipsometry. A title or a section heading naming it counts.
  chamber         the reactor or tool. **A reactor geometry or type is a chamber**: \
"hot-wall", "cross-flow", "showerhead", "Beneq TFS-200", "Kretschmann". If the \
passage says the work was done in a cross-flow reactor, chamber is "cross-flow".
  substrate       Si(100), MgO(001), fused silica, ...
  electrode       Pt, TiN, Au, ...
  interface       any stated interlayer or roughness treatment
  precursor       the metal source, e.g. TDMAH, TMA, TEMAZ
  oxidant         water, ozone, O2 plasma, ...
  temperature_c   the temperature in Celsius, as a number, when the passage states \
Celsius — which is what almost every growth paper states. "250 degC" is \
temperature_c: 250. Use this rather than converting.
  temperature_k   only when the passage itself states kelvin. Never convert; the \
conversion from temperature_c is done downstream, exactly, in code.
  pressure_torr   the chamber pressure, in the passage's own units if unsure
  frequency_hz    the measurement frequency for an electrical property
  thickness_nm    the film thickness, when stated for the measured quantity
  anneal          any post-deposition treatment
  failure_criterion   for a breakdown field
  method, software, xc_functional   for a calculated value

Rules:
1. Extract only what the passage states. If it does not give a temperature, leave \
temperature_k out. Do not infer it, do not carry it from another sentence you \
remember, and do not supply a typical value.
2. **Be consistent.** If the passage names the material, the technique, the \
precursor and the reactor once, every claim from that passage carries all four. \
Two claims from one passage disagreeing about their own context is always an error \
on your part, and it makes two sources look as though they differ when they do not.
3. Convert nothing, ever. If the passage says 250 degC, that is temperature_c: 250 \
and temperature_k is absent. A range or a window is not a value: "200 to 300 degC" \
goes in value_text or under the field as written, not into temperature_c as a number.
4. The quote must appear verbatim in the passage. It is how a reader checks you.
5. A passage that states no quantity yields an empty list. That is a normal and \
useful answer.
6. Never emit a field you did not read. An invented claim is worse than a missing \
one, because it is indistinguishable from a real one downstream.

Reply with JSON only: {"claims": [ ... ]}"""


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply.

    Local models wrap JSON in prose, fenced blocks, and reasoning tags often
    enough that a parser which fails on formatting would silently report an empty
    corpus.
    """
    if not text:
        return None
    #  Reasoning models emit <think>...</think> before the answer.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned.strip(), flags=re.MULTILINE).strip()
    candidates = [cleaned]
    #  Greedy outermost object, for a reply with prose on both sides.
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"claims": parsed}
    return None


def _clean_context(raw: object) -> dict:
    """Keep the context keys we recognise; drop nothing silently."""
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if key in CLAIM_CONTEXT_FIELDS and value not in (None, "", [], {})
    }


def _coerce_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return None if value != value else float(value)
    if isinstance(value, str):
        #  "0.98", "0.98 A/cycle", "~250" all carry a number worth keeping.
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def extraction_provider(answer_provider, policy: ResearchPolicy):
    """The model that extracts, which need not be the one that answers.

    Extraction is structured output — read one passage, emit JSON — and it runs once
    per passage, so it dominates the wall clock while needing none of the answer
    model's reasoning. A reasoning model here spends its budget deliberating about
    the schema: measured on this scaffold, 90-250 seconds *per passage* against a
    few seconds for a small model.

    Resolution order: the policy's ``extraction_model``, then
    ``RAG_EXTRACTION_MODEL``, then the answer provider unchanged.
    """
    from cnms_fom.config import get_settings
    from cnms_fom.rag_backend.providers import get_provider

    model = policy.extraction_model or get_settings().rag_extraction_model
    if not model or model == getattr(answer_provider, "model", None):
        return answer_provider
    return get_provider(getattr(answer_provider, "name", None), model)


def extract_claims(
    provider,
    evidence: list[EvidenceItem],
    *,
    policy: ResearchPolicy,
    db=None,
    record: dict | None = None,
) -> tuple[list[ExtractedClaim], list[str]]:
    """Extract claims from each passage. Returns the claims and any problems.

    ``record``, when given, receives the call counts and the number of passages whose
    extraction failed for transport reasons — kept apart from ``problems``, which is
    about what the model *said*.

    One model call per passage rather than one over all of them: a single call
    would let the model blend two documents' numbers into one claim, and
    attributing a blended value back to a page is impossible. The cost is linear in
    passages, which is the same shape as grading and is bounded by ``top_k``.

    ``provider`` is used as given. Callers that want the cheap-model split should
    resolve it through :func:`extraction_provider` first — ``brief`` does.
    """
    claims: list[ExtractedClaim] = []
    problems: list[str] = []
    model = getattr(provider, "model", "unknown")

    #  Cached on a hash of the passage alone, because the prompt above contains the
    #  passage and not the question — so an extraction is a pure function of the
    #  passage and a given passage needs extracting once ever. Misses run
    #  concurrently. Both are exact; the claims are identical either way.
    payloads, cached, called, errors = cache.map_cached(
        db,
        evidence,
        kind=cache.KIND_EXTRACTION,
        key_of=lambda item: cache.extraction_key(item.quote),
        model=model,
        prompt_version=policy.extraction_prompt_version,
        call=lambda item: _call_extractor(provider, item),
    )
    if cached:
        logger.info(
            "Extraction: %d passage(s) served from cache, %d call(s) made.", cached, called
        )

    for index, (item, payload) in enumerate(zip(evidence, payloads, strict=True)):
        if payload is None:
            reason = errors.get(index)
            problems.append(
                f"Extraction failed for {item.citation}: {reason}"
                if reason
                else f"Extraction produced nothing usable for {item.citation}."
            )
            continue
        if payload.get("refused"):
            problems.append(f"The model declined to extract from {item.citation}.")
            continue
        if payload.get("unparseable"):
            problems.append(
                f"Unparseable extraction reply for {item.citation}; no claims taken from it."
            )
            continue

        for raw in payload.get("claims") or []:
            if not isinstance(raw, dict):
                continue
            claim = _claim_from_extraction(
                raw, item, payload.get("model", model), payload.get("provider", ""), policy, problems
            )
            if claim is not None:
                claims.append(claim)

    if record is not None:
        record["extraction_calls"] = called
        record["extraction_cache_hits"] = cached
        #  Distinct from the content problems in ``problems``: "the model was
        #  unreachable" and "the model read this passage and found nothing" must not
        #  be the same signal, because the first one invalidates the brief's
        #  conclusion and the second one *is* the conclusion.
        record["extraction_failed"] = len(errors)
    return claims, problems


def _call_extractor(provider, item: EvidenceItem) -> dict:
    """One extraction call, reduced to the JSON worth caching.

    A refusal and an unparseable reply are cached as such: both are deterministic
    properties of this passage and this model, and re-asking would get the same
    answer at the same price. A *transport* failure raises instead, so
    :func:`cache.map_cached` records a miss — caching that would let one unreachable
    server poison every later run.
    """
    message = (
        f"Passage from {item.citation}:\n\"\"\"\n{item.quote.strip()}\n\"\"\"\n\n"
        "Extract every quantity this passage states."
    )
    result = provider.send(EXTRACTION_SYSTEM_PROMPT, [{"role": "user", "content": message}])
    model = getattr(result, "model", "") or getattr(provider, "model", "")
    name = getattr(result, "provider", "") or getattr(provider, "name", "")

    if getattr(result, "refused", False):
        return {"refused": True, "model": model, "provider": name}

    payload = _extract_json(result.text)
    if payload is None:
        return {"unparseable": True, "model": model, "provider": name}
    return {"claims": payload.get("claims") or [], "model": model, "provider": name}


def _claim_from_extraction(
    raw: dict,
    item: EvidenceItem,
    model: str,
    provider_name: str,
    policy: ResearchPolicy,
    problems: list[str],
) -> ExtractedClaim | None:
    field_name = str(raw.get("field") or raw.get("field_name") or "").strip()
    if not field_name:
        problems.append(f"An extraction from {item.citation} had no field name; dropped.")
        return None

    quote = str(raw.get("quote") or "").strip()
    #  The quote must actually be in the passage. A model that paraphrases its own
    #  evidence has broken the only link a reader can follow back, and accepting it
    #  would make a hallucinated claim look sourced.
    if quote and quote.lower() not in item.quote.lower():
        problems.append(
            f"Extraction for {field_name!r} from {item.citation} quoted text that is not in the "
            f"passage ({quote[:60]!r}); the passage's own text was used instead and the claim is "
            "marked low confidence."
        )
        quote = ""

    evidence_item = EvidenceItem(
        document_id=item.document_id,
        document_title=item.document_title,
        page=item.page,
        quote=quote or item.quote,
        content_sha256=item.content_sha256,
        chunk_id=item.chunk_id,
        doi=item.doi,
        source_url=item.source_url,
        technique=item.technique,
        retrieval_method=item.retrieval_method,
        retrieval_rank=item.retrieval_rank,
        grade=item.grade,
        grade_reason=item.grade_reason,
    )

    try:
        tier = ClaimTier(str(raw.get("tier", "reported")).strip().lower())
    except ValueError:
        tier = ClaimTier.UNKNOWN

    confidence = _coerce_number(raw.get("confidence"))
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    if not quote and confidence is not None:
        confidence = min(confidence, policy.low_confidence_threshold)

    notes = ""
    if confidence is not None and confidence < policy.low_confidence_threshold:
        #  Recorded and marked, never dropped: a discarded extraction is invisible,
        #  and an underconfident extractor would look like an empty corpus.
        notes = (
            f"model confidence {confidence:.2f} is below the policy threshold "
            f"{policy.low_confidence_threshold:.2f}; treat as a lead, not a value"
        )

    try:
        return ExtractedClaim(
            field_name=field_name,
            evidence=[evidence_item],
            value=_coerce_number(raw.get("value")),
            units=(str(raw["units"]).strip() if raw.get("units") else None),
            value_text=(str(raw["value_text"]).strip() if raw.get("value_text") else None),
            tier=tier,
            context=_clean_context(raw.get("context")),
            model_confidence=confidence,
            extracted_by_model=model or "",
            extracted_by_provider=provider_name or "",
            prompt_version=policy.extraction_prompt_version,
            notes=notes,
        )
    except ResearchContractError as exc:
        problems.append(f"Rejected an extraction for {field_name!r} from {item.citation}: {exc}")
        return None


def find_contradictions(claims: list[ExtractedClaim]) -> list[Contradiction]:
    """Pairs of claims for one field whose values disagree.

    Arithmetic, not model judgement. Asking a model whether two numbers conflict
    invites it to explain the conflict away, and a reconciled contradiction is a
    lost finding.

    Two claims from the *same* document and page are not compared: that is one
    source stating a range or two conditions, not a disagreement. Claims whose
    context differs are still reported, with the difference attached — a different
    reactor is usually the explanation, and naming it is the useful part.
    """
    by_field: dict[str, list[ExtractedClaim]] = {}
    for claim in claims:
        if claim.value is not None:
            by_field.setdefault(claim.field_name, []).append(claim)

    found: list[Contradiction] = []
    for field_name, group in by_field.items():
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                if _same_source(left, right):
                    continue
                spread = _relative_spread(left.value, right.value)
                if spread is None or spread <= DISAGREEMENT_FRACTION:
                    continue
                if not _units_comparable(left, right):
                    continue

                differing = _differing_context(left, right)
                basis = (
                    f"{spread:.0%} relative spread between {left.value:g} "
                    f"{left.units or ''} and {right.value:g} {right.units or ''}".strip()
                )
                if differing:
                    basis += (
                        "; context differs in " + ", ".join(sorted(differing))
                        + ", which is the likely explanation and should be checked before "
                        "treating this as a conflict"
                    )
                else:
                    basis += (
                        "; no recorded context difference, so the two sources disagree about the "
                        "same quantity under the same stated conditions"
                    )
                found.append(
                    Contradiction(
                        field_name=field_name,
                        left=left,
                        right=right,
                        basis=basis,
                        differing_context=differing,
                    )
                )
    return found


def _same_source(left: ExtractedClaim, right: ExtractedClaim) -> bool:
    a, b = left.evidence[0], right.evidence[0]
    same_document = (
        (a.document_id is not None and a.document_id == b.document_id)
        or (a.content_sha256 is not None and a.content_sha256 == b.content_sha256)
    )
    return bool(same_document and a.page == b.page)


def _relative_spread(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    midpoint = (left + right) / 2.0
    return abs(left - right) / abs(midpoint) if midpoint else None


def _units_comparable(left: ExtractedClaim, right: ExtractedClaim) -> bool:
    """Whether two claims are in the same units, as written.

    No conversion is attempted. Two numbers in different units are not a
    disagreement until somebody has converted them, and doing that silently is how
    a factor of ten becomes a finding.
    """
    a = (left.units or "").strip().lower()
    b = (right.units or "").strip().lower()
    if not a or not b:
        #  One or both unitless: compare anyway, since a dimensionless quantity
        #  like a permittivity is often written without units.
        return True
    return a == b


def _differing_context(left: ExtractedClaim, right: ExtractedClaim) -> dict[str, tuple]:
    differing: dict[str, tuple] = {}
    for key in set(left.context) | set(right.context):
        mine, theirs = left.context.get(key), right.context.get(key)
        if mine != theirs:
            differing[key] = (mine, theirs)
    return differing
