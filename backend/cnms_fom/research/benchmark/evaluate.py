"""Scoring a brief against a case whose answer is known.

Design rules, in the order they matter:

**A metric that cannot be computed is ``None``, never zero.**  Running the
benchmark with a stub extractor measures retrieval and abstention honestly and
reports extraction as unavailable.  Scoring an absent capability as 0 would make a
retrieval-only run look like a broken pipeline, and scoring it as 1 would make it
look like a working one.

**Unsupported claims are counted, not tolerated.**  A claim whose quote is not in
the passage it cites is the failure mode with the worst consequences and the best
disguise, so it gets its own metric and its own place in the headline score.

**Abstention is scored as a capability.**  Answering a question the corpus cannot
answer and abstaining on one it can are both errors, and a pipeline that abstains
on everything must not out-score one that answers well.

**A capability the run could not exercise is skipped, not failed.**  Deciding that
"GaAs on GaAs(001)" does not answer "GaAs on germanium" is grading work: the two
share almost every term, so the offline term-overlap grader cannot separate them.
Scoring the pipeline zero for the stub's blindness would blame the wrong component
and would make every real-grader run look like an improvement it had not earned.
So an abstention case run without a real grader is marked ``exercised=False``, left
out of the aggregate, and counted in ``n_cases_skipped``.

**Nothing here writes to the case set.**  The truth data is frozen; a run produces
a score and a diagnosis, never an updated expectation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cnms_fom.research.benchmark.cases import DOCUMENTS_BY_KEY, BenchmarkCase, page_text
from cnms_fom.research.contracts import (
    ResearchBrief,
    classify_unit,
    convert_to_canonical,
)

logger = logging.getLogger(__name__)

#  Weights of the headline score. Retrieval and citation validity dominate, because
#  a pipeline that finds the wrong page cannot be rescued downstream; the
#  unsupported-claim rate is subtracted rather than weighted, so a run cannot buy a
#  good score with confident fabrication.
#  Recall alone is nearly useless here: it asks only whether the right page
#  *appeared* in the window, and the first run of this benchmark scored every policy
#  at 1.000 because it always did. What separates policies is where the answer
#  *ranked* and how much of the window it wasted, so reciprocal rank and precision
#  carry real weight.
SCORE_WEIGHTS: dict[str, float] = {
    "doc_recall": 0.12,
    "page_recall": 0.12,
    "reciprocal_rank": 0.20,
    "page_precision": 0.13,
    "citation_accuracy": 0.13,
    "extraction_f1": 0.15,
    "context_completeness": 0.08,
    "abstention_f1": 0.07,
}
UNSUPPORTED_CLAIM_PENALTY = 0.5

#  ``abstention_f1`` is a run-level metric: F1 needs a population, so no single case
#  has one. The per-case stand-in is the boolean ``abstention_correct``, carrying the
#  same weight. Without this mapping the lookup in ``CaseResult.score`` silently
#  missed — and because an abstention case computes *no other* metric, `available`
#  came out empty and every such case scored a hard 0.0 whether it abstained
#  correctly or not. A correct abstention was indistinguishable from a failed one in
#  the case score, the category score and the headline score, so no amount of policy
#  tuning could be rewarded for getting abstention right. It was invisible until the
#  first real-provider run because the stub grader marks abstention cases
#  unexercised and drops them entirely.
PER_CASE_METRIC: dict[str, str] = {"abstention_f1": "abstention_correct"}


@dataclass
class CaseResult:
    """One case's score, and enough detail to diagnose it."""

    case_id: str
    category: str
    #  None where the capability was not exercised.
    doc_recall: float | None = None
    page_recall: float | None = None
    #  1/rank of the first expected page. Rewards ranking the answer first rather
    #  than merely including it, which is the difference a policy change makes.
    reciprocal_rank: float | None = None
    #  Fraction of returned corpus passages that were expected. Penalises a policy
    #  that fills its window with plausible-looking distractors.
    page_precision: float | None = None
    citation_accuracy: float | None = None
    extraction_precision: float | None = None
    extraction_recall: float | None = None
    extraction_f1: float | None = None
    unit_accuracy: float | None = None
    #  Claim shapes the case forbids that appeared anyway. A fabrication is not a
    #  partial success, so any of these zeroes the case.
    forbidden_present: int = 0
    context_completeness: float | None = None
    contradiction_detected: bool | None = None
    abstained: bool = False
    abstention_correct: bool | None = None
    unsupported_claims: int = 0
    n_claims: int = 0
    latency_ms: int = 0
    #  False when the run lacked a capability the case depends on. Such a case is
    #  excluded from the aggregate rather than scored zero — see the module
    #  docstring.
    exercised: bool = True
    #  Human-readable reasons this case scored as it did.
    diagnostics: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Weighted score over the metrics this case actually exercised.

        Renormalised over the available components, so a case that does not test
        extraction is not penalised for it.
        """
        available = {}
        for name in SCORE_WEIGHTS:
            value = getattr(self, PER_CASE_METRIC.get(name, name), None)
            if value is None:
                continue
            #  abstention_correct is a bool; the rest are already 0..1.
            available[name] = float(value)
        if not available:
            return 0.0
        total_weight = sum(SCORE_WEIGHTS[name] for name in available)
        weighted = sum(SCORE_WEIGHTS[name] * value for name, value in available.items())
        base = weighted / total_weight if total_weight else 0.0

        if self.n_claims:
            base -= UNSUPPORTED_CLAIM_PENALTY * (self.unsupported_claims / self.n_claims)
        if self.forbidden_present:
            #  Not scaled: "an invented claim is worse than a missing one" (§10d). A case
            #  that reproduces a known fabrication has failed whatever else it got right,
            #  and averaging that away is how a regression ships.
            return 0.0
        return max(0.0, min(1.0, base))

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "score": self.score,
            "doc_recall": self.doc_recall,
            "page_recall": self.page_recall,
            "reciprocal_rank": self.reciprocal_rank,
            "page_precision": self.page_precision,
            "citation_accuracy": self.citation_accuracy,
            "extraction_precision": self.extraction_precision,
            "extraction_recall": self.extraction_recall,
            "extraction_f1": self.extraction_f1,
            "unit_accuracy": self.unit_accuracy,
            "context_completeness": self.context_completeness,
            "contradiction_detected": self.contradiction_detected,
            "abstained": self.abstained,
            "abstention_correct": self.abstention_correct,
            "unsupported_claims": self.unsupported_claims,
            "n_claims": self.n_claims,
            "latency_ms": self.latency_ms,
            "exercised": self.exercised,
            "diagnostics": self.diagnostics,
        }


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _document_key_for_title(title: str) -> str | None:
    for key, document in DOCUMENTS_BY_KEY.items():
        if document.title == title:
            return key
    return None


def evaluate_case(
    case: BenchmarkCase,
    brief: ResearchBrief,
    *,
    extraction_available: bool,
    grading_available: bool = True,
) -> CaseResult:
    """Score one brief against one case.

    ``grading_available=False`` says the run used the offline stub grader, which
    cannot separate a near-miss passage from an answer. Abstention cases are then
    skipped rather than failed.
    """
    result = CaseResult(
        case_id=case.case_id,
        category=case.category,
        abstained=brief.abstained,
        n_claims=len(brief.claims),
    )

    retrieved_keys: set[str] = set()
    retrieved_pages: set[tuple[str, int]] = set()
    #  Ordered, so the rank of the first expected page is recoverable.
    ranked_pages: list[tuple[str, int]] = []
    for item in brief.evidence:
        key = _document_key_for_title(item.document_title)
        if key is None:
            continue  # a card, or a record reference — not a corpus document
        retrieved_keys.add(key)
        if item.page is not None:
            retrieved_pages.add((key, item.page))
            ranked_pages.append((key, item.page))

    # --- abstention -------------------------------------------------------
    if brief.degraded:
        #  The brief could not reach the model, so it abstained for want of a grader
        #  rather than for want of evidence. Scoring that as a correct abstention
        #  would let a total outage look like perfect judgement — every case would
        #  abstain and every abstention case would pass.
        result.abstention_correct = None
        result.diagnostics.append(f"Abstention not scored: {brief.degraded_reason}")
    else:
        result.abstention_correct = brief.abstained == case.should_abstain
    if case.should_abstain and not brief.abstained and not brief.degraded:
        result.diagnostics.append(
            "Should have abstained: the corpus does not answer this question, but the brief "
            f"produced {len(brief.claims)} claim(s)."
        )
        if case.distractor_documents:
            wrongly_used = retrieved_keys & set(case.distractor_documents)
            if wrongly_used:
                result.diagnostics.append(
                    "Answered from a distractor: " + ", ".join(sorted(wrongly_used))
                    + ". These look relevant and are not."
                )
    elif brief.abstained and not case.should_abstain and not brief.degraded:
        result.diagnostics.append(
            "Abstained on an answerable question. The evidence is in the corpus; retrieval or "
            "grading did not surface it."
        )

    #  An abstention case has no retrieval or extraction target, so those metrics
    #  stay None rather than being scored against an empty expectation.
    if case.should_abstain:
        if not grading_available:
            result.exercised = False
            result.abstention_correct = None
            result.diagnostics = [
                "Skipped: deciding this question is unanswerable is grading work, and the offline "
                "stub grader scores by term overlap — a near-miss passage shares almost every term "
                "with the question. Run with a real provider to exercise abstention."
            ]
        return result

    # --- retrieval --------------------------------------------------------
    if case.expected_documents:
        hit = retrieved_keys & set(case.expected_documents)
        result.doc_recall = len(hit) / len(case.expected_documents)
        missed = set(case.expected_documents) - retrieved_keys
        if missed:
            result.diagnostics.append(f"Missed document(s): {sorted(missed)}")

    if case.expected_pages:
        expected = set(case.expected_pages)
        hit_pages = retrieved_pages & expected
        result.page_recall = len(hit_pages) / len(expected)
        missed_pages = expected - retrieved_pages
        if missed_pages:
            result.diagnostics.append(
                "Missed page(s): " + ", ".join(f"{k} p.{p}" for k, p in sorted(missed_pages))
            )

        #  Where the answer ranked. Including it at position six is not the same as
        #  leading with it: the extractor reads the top of the list most carefully,
        #  and a long context pushes the retrieval instructions out of the window.
        result.reciprocal_rank = 0.0
        for rank, page in enumerate(ranked_pages, start=1):
            if page in expected:
                result.reciprocal_rank = 1.0 / rank
                if rank > len(expected):
                    result.diagnostics.append(
                        f"The first expected page ranked {rank}; {rank - 1} other passage(s) "
                        "were ranked above it."
                    )
                break
        else:
            result.diagnostics.append("No expected page appeared in the retrieved window at all.")

        if ranked_pages:
            result.page_precision = len(
                [page for page in ranked_pages if page in expected]
            ) / len(ranked_pages)
            distractors = [
                f"{k} p.{p}" for k, p in ranked_pages
                if (k, p) not in expected and k in set(case.distractor_documents)
            ]
            if distractors:
                result.diagnostics.append(
                    "Window included known distractor page(s): " + ", ".join(distractors)
                )

    # --- citation validity ------------------------------------------------
    #  Checked against the frozen corpus text, not against what the brief says about
    #  itself: a citation is valid only if the quote is actually on the page named.
    if brief.claims:
        valid = 0
        for claim in brief.claims:
            evidence = claim.evidence[0]
            key = _document_key_for_title(evidence.document_title)
            source = page_text(key, evidence.page) if key and evidence.page else None
            if source and _quote_is_in(evidence.quote, source):
                valid += 1
            else:
                result.unsupported_claims += 1
                result.diagnostics.append(
                    f"Unsupported claim {claim.field_name}={claim.value}: its quote is not on "
                    f"{evidence.citation}."
                )
        result.citation_accuracy = valid / len(brief.claims)

    # --- extraction -------------------------------------------------------
    if case.expected_claims and extraction_available:
        matched, precision, recall, unit_hits = _match_claims(case, brief, result)
        result.extraction_precision = precision
        result.extraction_recall = recall
        result.extraction_f1 = _f1(precision, recall)
        result.unit_accuracy = unit_hits

        #  Context completeness over the claims that matched an expectation: a
        #  correct number with a missing frequency is not a usable value (Sec. 16).
        if matched:
            complete = sum(1 for claim, expected in matched if _context_ok(claim, expected))
            result.context_completeness = complete / len(matched)
            for claim, expected in matched:
                if not _context_ok(claim, expected):
                    missing = sorted(set(expected.required_context) - set(claim.context))
                    result.diagnostics.append(
                        f"{claim.field_name}={claim.value} extracted without required context "
                        f"{missing}, which the source does state."
                    )
    elif case.expected_claims:
        result.diagnostics.append(
            "Extraction not exercised in this run (no extracting provider), so extraction, unit, "
            "and context metrics are unavailable rather than zero."
        )

    # --- what must NOT be there ------------------------------------------
    #  The suite could previously assert only what should be found, so the §10d
    #  fabrications could not become regression cases.
    for forbidden in case.forbidden_claims:
        for claim in brief.claims:
            if claim.field_name != forbidden.field_name:
                continue
            if forbidden.units_dimension is not None and (
                classify_unit(claim.units) != forbidden.units_dimension
            ):
                continue
            if forbidden.any_numeric_value and claim.value is None:
                continue
            result.forbidden_present += 1
            where = claim.evidence[0].citation if claim.evidence else "?"
            result.diagnostics.append(
                f"FORBIDDEN claim present: {claim.field_name}={claim.value} "
                f"{claim.units!r} from {where}"
                + (f" — {forbidden.why}" if forbidden.why else "")
            )

    for field_name in case.forbid_contradiction_fields:
        for contradiction in brief.contradictions:
            if contradiction.field_name == field_name:
                result.forbidden_present += 1
                result.diagnostics.append(
                    f"FORBIDDEN contradiction reported on {field_name}: the corpus does "
                    "not disagree here, so this is a manufactured disagreement."
                )

    # --- contradictions ---------------------------------------------------
    if case.expected_contradiction_fields:
        #  Named differently from the page-set `expected` above: reusing that name
        #  for a set of field names is how a type error hides in plain sight.
        found = {c.field_name for c in brief.contradictions}
        expected_fields = set(case.expected_contradiction_fields)
        result.contradiction_detected = bool(expected_fields & found)
        if not result.contradiction_detected:
            result.diagnostics.append(
                f"Did not report the known disagreement on {sorted(expected_fields)}. Both values "
                "may have been found; the conflict was not named."
            )

    return result


def _quote_is_in(quote: str, source: str) -> bool:
    """Whether a quote appears in its source page, ignoring whitespace shape."""
    def normalise(text: str) -> str:
        return " ".join(text.lower().split())

    needle = normalise(quote)
    if not needle:
        return False
    return needle in normalise(source)


#  Spellings of the same unit. The extractor is told to copy the passage's own units
#  verbatim, and passages write the same unit many ways — "A/cycle", "Angstrom/cycle",
#  "angstrom per cycle", "Å/cycle". Comparing those as strings measures which spelling
#  the source happened to use, not whether the extraction was right. Normalisation
#  here, not in the extractor: rewriting the units would destroy the verbatim record
#  that makes a claim checkable against its page.
_UNIT_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("angstroms", "a"), ("angstrom", "a"), ("\u00e5ngstr\u00f6m", "a"), ("\u00c5", "a"),
    ("degrees celsius", "degc"), ("deg c", "degc"), ("\u00b0c", "degc"),
    ("degrees kelvin", "k"), ("kelvin", "k"),
    ("nanometres", "nm"), ("nanometers", "nm"),
    ("millitorr", "mtorr"), ("torr", "torr"),
    (" per ", "/"),
)


def normalise_units(units: str | None) -> str:
    """A comparable form of a unit string, for scoring only.

    Never written back to a claim: the stored units are what the source said.
    """
    text = (units or "").strip().lower()
    for spelling, canonical in _UNIT_SYNONYMS:
        text = text.replace(spelling.lower(), canonical)
    #  "a / cycle" and "a/cycle" are the same unit; whitespace is not information.
    return "".join(text.split())


#  Gold field names that a *correct* extraction may legitimately file under a different
#  registry key. Each entry is an explicit, reviewed pair with a stated reason: no fuzzy
#  matching, no substrings, no edit distance. `_match_claims` stays strict and consults
#  this first, so the relaxation is visible in one place rather than spread through the
#  matcher.
#
#  These exist because §10e proved two of three benchmark "misses" were not misses:
#  the values were extracted, passed every guard, and appeared in the brief under the
#  keys the extract-v4 prompt instructs the model to use. The gold names were the thing
#  that was wrong, and `extraction_f1 = 0.397` understated true recall as a result.
GOLD_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    #  The prompt mandates temperature_c for any Celsius temperature, and a growth
    #  table's "substrate_temperature 700 degC" is one. The gold name keeps the
    #  substrate-specific intent; the alias accepts the registry key.
    "substrate_temperature": ("temperature_c",),
    #  Likewise for a chamber pressure: pressure_torr is the registry key, and the
    #  oxygen-specific gold name records what the column meant.
    "oxygen_pressure": ("pressure_torr",),
}


def _keys_matching(expected_field: str) -> tuple[str, ...]:
    """The extracted keys that satisfy one gold field name."""
    return (expected_field, *GOLD_KEY_ALIASES.get(expected_field, ()))


def _match_claims(case: BenchmarkCase, brief: ResearchBrief, result: CaseResult):
    """Greedily pair extracted claims with expectations.

    One expectation per claim: two extractions of the same value are not two
    correct answers, and letting them both count would reward a pipeline that
    emits duplicates.
    """
    unmatched = list(case.expected_claims)
    matched: list[tuple] = []
    correct = 0
    unit_correct = 0

    for claim in brief.claims:
        for expected in list(unmatched):
            if claim.field_name not in _keys_matching(expected.field_name):
                continue
            if expected.value is not None:
                if claim.value is None:
                    continue
                #  The gold is written in the source's own units ("oxygen_pressure
                #  100 mTorr") while the claim has already been normalised to the unit
                #  its field name declares (0.1 torr). Both go through the same
                #  conversion, keyed on the *claim's* field name, so the comparison
                #  happens in one unit. Reused rather than reimplemented: two copies of
                #  this arithmetic would eventually disagree.
                gold_value, _gold_units, _ = convert_to_canonical(
                    claim.field_name, expected.value, expected.units
                )
                if gold_value is None:
                    continue
                tolerance = abs(gold_value) * expected.tolerance
                if abs(claim.value - gold_value) > max(tolerance, 1e-12):
                    continue
            unmatched.remove(expected)
            matched.append((claim, expected))
            correct += 1
            _gv, gold_units, _ = convert_to_canonical(
                claim.field_name, expected.value, expected.units
            )
            if expected.units is None or normalise_units(claim.units) == normalise_units(
                gold_units
            ):
                unit_correct += 1
            else:
                result.diagnostics.append(
                    f"{claim.field_name}: units {claim.units!r} do not match the expected "
                    f"{expected.units!r}."
                )
            break

    for expected in unmatched:
        result.diagnostics.append(
            f"Missed claim {expected.field_name}"
            + (f"={expected.value}" if expected.value is not None else "")
        )

    precision = correct / len(brief.claims) if brief.claims else 0.0
    recall = correct / len(case.expected_claims) if case.expected_claims else None
    #  None, not 0.0, when nothing matched: a case that extracted no claim at all has
    #  no units to be right or wrong about, and scoring it zero says the units were
    #  wrong. Surfacing this metric in the run report made the difference visible —
    #  it read 0.7778 while every unit that was actually checked was correct, because
    #  two cases that matched no claims each contributed a hard zero to the mean.
    #  Same rule as everywhere else here: an unexercised metric is n/a.
    unit_accuracy = unit_correct / correct if correct else None
    return matched, precision, recall, unit_accuracy


def _context_ok(claim, expected) -> bool:
    """Whether a claim carries the context its expectation demands."""
    if expected.expect_incomparable:
        #  The source states no context, so a correct extraction is one that knows
        #  the claim is incomparable rather than one that invented context for it.
        return not claim.is_comparable
    return set(expected.required_context).issubset(set(claim.context))


@dataclass
class BenchmarkResult:
    """Every case's result, plus the aggregate."""

    policy_version: str
    case_set: str
    results: list[CaseResult] = field(default_factory=list)
    extraction_available: bool = True
    retrievers: tuple[str, ...] = ()
    git_commit: str = ""
    notes: str = ""

    @property
    def exercised(self) -> list[CaseResult]:
        """Cases the run could actually measure. The aggregate is over these."""
        return [r for r in self.results if r.exercised]

    @property
    def n_cases_skipped(self) -> int:
        return len(self.results) - len(self.exercised)

    def _mean(self, name: str) -> float | None:
        values = [
            getattr(r, name) for r in self.exercised if getattr(r, name, None) is not None
        ]
        return sum(values) / len(values) if values else None

    @property
    def overall_score(self) -> float:
        exercised = self.exercised
        return sum(r.score for r in exercised) / len(exercised) if exercised else 0.0

    @property
    def doc_recall(self) -> float | None:
        return self._mean("doc_recall")

    @property
    def page_recall(self) -> float | None:
        return self._mean("page_recall")

    @property
    def reciprocal_rank(self) -> float | None:
        return self._mean("reciprocal_rank")

    @property
    def page_precision(self) -> float | None:
        return self._mean("page_precision")

    @property
    def citation_accuracy(self) -> float | None:
        return self._mean("citation_accuracy")

    @property
    def extraction_f1(self) -> float | None:
        return self._mean("extraction_f1")

    @property
    def context_completeness(self) -> float | None:
        return self._mean("context_completeness")

    @property
    def unit_accuracy(self) -> float | None:
        """Share of matched claims whose units agree with the expectation.

        Computed per case since the first version of this benchmark and never
        surfaced, which meant the unit normalisation in `normalise_units` could not be
        observed from a run. A metric nobody can read is a metric nobody maintains.
        """
        return self._mean("unit_accuracy")

    @property
    def abstention_f1(self) -> float | None:  # noqa: D401
        """F1 over "should abstain" as the positive class.

        F1 rather than accuracy: three of twelve cases are abstentions, so a
        pipeline that never abstains would score 75% accuracy and be useless at the
        thing this metric exists to measure.
        """
        scored = [r for r in self.exercised if r.abstention_correct is not None]
        if not scored:
            #  None rather than 1.0: "nothing was measured" and "everything was
            #  correct" must not look the same in a results table.
            return None
        tp = sum(1 for r in scored if r.abstained and r.abstention_correct)
        fp = sum(1 for r in scored if r.abstained and not r.abstention_correct)
        fn = sum(1 for r in scored if not r.abstained and not r.abstention_correct)
        if tp + fp + fn == 0:
            return 1.0  # nothing to abstain on and nothing wrongly abstained
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return _f1(precision, recall) or 0.0

    @property
    def unsupported_claim_rate(self) -> float:
        claims = sum(r.n_claims for r in self.exercised)
        unsupported = sum(r.unsupported_claims for r in self.exercised)
        return unsupported / claims if claims else 0.0

    @property
    def latency_ms(self) -> int:
        return sum(r.latency_ms for r in self.results)

    def by_category(self) -> dict[str, float]:
        """Mean score per category — an aggregate hides which capability moved."""
        grouped: dict[str, list[float]] = {}
        for result in self.exercised:
            grouped.setdefault(result.category, []).append(result.score)
        return {k: sum(v) / len(v) for k, v in sorted(grouped.items())}

    def failures(self) -> list[dict]:
        """Cases that scored badly, with why — the actionable output."""
        return [
            {"case_id": r.case_id, "category": r.category, "score": r.score,
             "diagnostics": r.diagnostics}
            for r in sorted(self.results, key=lambda r: r.score)
            if r.score < 0.8 or r.diagnostics
        ]

    def as_dict(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "case_set": self.case_set,
            "git_commit": self.git_commit,
            "extraction_available": self.extraction_available,
            "retrievers": list(self.retrievers),
            "overall_score": self.overall_score,
            "doc_recall": self.doc_recall,
            "page_recall": self.page_recall,
            "reciprocal_rank": self.reciprocal_rank,
            "page_precision": self.page_precision,
            "citation_accuracy": self.citation_accuracy,
            "extraction_f1": self.extraction_f1,
            "context_completeness": self.context_completeness,
            "abstention_f1": self.abstention_f1,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "latency_ms": self.latency_ms,
            "n_cases": len(self.results),
            "n_cases_exercised": len(self.exercised),
            "n_cases_skipped": self.n_cases_skipped,
            "by_category": self.by_category(),
            "cases": [r.as_dict() for r in self.results],
            "failures": self.failures(),
            "notes": self.notes,
        }
