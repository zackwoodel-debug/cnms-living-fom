"""The benchmark: honest metrics, frozen truth data, reproducible runs.

Everything here runs offline in a throwaway database. The properties under test are
mostly about *not lying*: a capability the run could not exercise must not score as
zero or as one, and a policy must not be able to buy a better number with confident
fabrication.
"""

from __future__ import annotations

import re

import pytest

from cnms_fom.research.benchmark import (
    CASE_SETS,
    CASES,
    CORPUS,
    RESULT_COLUMNS,
    BenchmarkResult,
    StubExtractor,
    append_result,
    compare_policies,
    evaluate_case,
    get_case_set,
    read_results,
    run_benchmark,
)
from cnms_fom.research.benchmark.cases import CASES_BY_ID, ExpectedClaim, page_text
from cnms_fom.research.benchmark.evaluate import CaseResult
from cnms_fom.research.benchmark.runner import RunOutcome, git_commit
from cnms_fom.research.contracts import (
    ClaimTier,
    EvidenceItem,
    ExtractedClaim,
    ResearchBrief,
)
from cnms_fom.research.policy import BASELINE, get_policy

# --- the fixture corpus ---------------------------------------------------


def test_the_corpus_is_large_enough_to_make_retrieval_selective():
    """A corpus smaller than the retrieval window measures nothing.

    The first version of this benchmark had eleven pages against a window of six and
    scored every policy at 1.000.
    """
    pages = sum(len(document.pages) for document in CORPUS)
    assert pages > 4 * BASELINE.top_k, f"only {pages} pages against top_k={BASELINE.top_k}"


def test_every_document_is_marked_synthetic():
    """These must be unmistakable if they ever escape a test database."""
    for document in CORPUS:
        assert document.title.startswith("SYNTHETIC")
        assert "SYNTHETIC BENCHMARK DOCUMENT" in document.pages[0].text


def test_document_keys_and_titles_are_unique():
    keys = [document.key for document in CORPUS]
    titles = [document.title for document in CORPUS]
    assert len(keys) == len(set(keys))
    #  Titles must be unique: the evaluator maps a retrieved title back to a key.
    assert len(titles) == len(set(titles))


def test_every_case_expectation_points_at_real_corpus_text():
    """Truth data that disagrees with the corpus scores a correct pipeline badly."""
    for case in CASES:
        for key, page in case.expected_pages:
            assert page_text(key, page), f"{case.case_id}: {key} p.{page} is not in the corpus"
        for key in case.expected_documents:
            assert any(d.key == key for d in CORPUS), f"{case.case_id}: unknown document {key}"
        for key in case.distractor_documents:
            assert any(d.key == key for d in CORPUS), f"{case.case_id}: unknown distractor {key}"


def test_abstention_cases_declare_no_expectations():
    """A case cannot both be unanswerable and have an expected answer."""
    for case in CASES:
        if case.should_abstain:
            assert not case.expected_claims
            assert not case.expected_pages


def test_case_sets_reference_real_cases():
    for name in CASE_SETS:
        for case in get_case_set(name):
            assert case.case_id in CASES_BY_ID


# --- the evaluator's honesty ---------------------------------------------


def _brief(**kwargs) -> ResearchBrief:
    defaults = {"research_question": "q"}
    return ResearchBrief(**{**defaults, **kwargs})


def _evidence(key: str, page: int, quote: str | None = None) -> EvidenceItem:
    from cnms_fom.research.benchmark.cases import DOCUMENTS_BY_KEY

    document = DOCUMENTS_BY_KEY[key]
    return EvidenceItem(
        document_id=None,
        document_title=document.title,
        page=page,
        quote=quote or page_text(key, page)[:120],
        content_sha256=document.content_sha256,
        retrieval_rank=1,
        grade=3,
    )


def test_an_unexercised_capability_is_none_not_zero():
    """Scoring an absent extractor as 0 makes a working pipeline look broken."""
    case = CASES_BY_ID["direct_lookup_gpc"]
    brief = _brief(evidence=[_evidence("hotwall", 2)])
    result = evaluate_case(case, brief, extraction_available=False)

    assert result.extraction_f1 is None
    assert result.context_completeness is None
    assert result.doc_recall == 1.0  # retrieval *was* exercised
    assert any("not exercised" in d for d in result.diagnostics)


def test_an_abstention_case_is_skipped_without_a_real_grader():
    """Separating a near-miss from an answer is grading work the stub cannot do."""
    case = CASES_BY_ID["absent_gaas_on_ge"]
    brief = _brief(abstained=False, evidence=[_evidence("mbe_gaas", 1)])

    skipped = evaluate_case(case, brief, extraction_available=False, grading_available=False)
    assert skipped.exercised is False
    assert skipped.abstention_correct is None
    assert any("stub grader" in d for d in skipped.diagnostics)

    #  With a real grader it is scored, and this brief fails it.
    scored = evaluate_case(case, brief, extraction_available=True, grading_available=True)
    assert scored.exercised is True
    assert scored.abstention_correct is False


def test_skipped_cases_are_excluded_from_the_aggregate():
    result = BenchmarkResult(policy_version="p", case_set="s")
    result.results = [
        CaseResult(case_id="a", category="x", doc_recall=1.0),
        CaseResult(case_id="b", category="y", exercised=False),
    ]
    assert len(result.exercised) == 1
    assert result.n_cases_skipped == 1
    assert result.overall_score == pytest.approx(1.0)
    assert "y" not in result.by_category()


def test_abstention_f1_is_none_when_nothing_was_measured():
    """"Nothing was measured" and "everything was correct" must not look alike."""
    result = BenchmarkResult(policy_version="p", case_set="s")
    result.results = [CaseResult(case_id="a", category="x", exercised=False)]
    assert result.abstention_f1 is None


def test_a_fabricated_quote_is_counted_as_an_unsupported_claim():
    """The worst failure mode with the best disguise, so it gets its own metric."""
    case = CASES_BY_ID["direct_lookup_gpc"]
    fabricated = ExtractedClaim(
        field_name="growth_per_cycle_ang", value=0.98, units="A/cycle",
        evidence=[_evidence("hotwall", 2, quote="we grew the film at 900 degrees in a furnace")],
        context={"temperature_k": 523.0, "chamber": "hot-wall", "precursor": "TDMAH"},
    )
    brief = _brief(evidence=[_evidence("hotwall", 2)], claims=[fabricated])
    result = evaluate_case(case, brief, extraction_available=True)

    assert result.unsupported_claims == 1
    assert result.citation_accuracy == 0.0
    assert any("Unsupported claim" in d for d in result.diagnostics)


def test_fabrication_cannot_buy_a_better_score():
    """The penalty is subtracted, not weighted, so it cannot be averaged away."""
    case = CASES_BY_ID["direct_lookup_gpc"]
    good_quote = "growth per cycle was constant at 0.98 angstrom per cycle"
    context = {"temperature_k": 523.0, "chamber": "hot-wall", "precursor": "TDMAH"}

    honest = _brief(
        evidence=[_evidence("hotwall", 2)],
        claims=[ExtractedClaim(
            field_name="growth_per_cycle_ang", value=0.98, units="A/cycle",
            evidence=[_evidence("hotwall", 2, quote=good_quote)], context=context,
        )],
    )
    inflated = _brief(
        evidence=[_evidence("hotwall", 2)],
        claims=[
            ExtractedClaim(
                field_name="growth_per_cycle_ang", value=0.98, units="A/cycle",
                evidence=[_evidence("hotwall", 2, quote=good_quote)], context=context,
            ),
            ExtractedClaim(
                field_name="rho", value=9.1, units="g/cm3",
                evidence=[_evidence("hotwall", 2, quote="invented supporting text")],
            ),
        ],
    )
    honest_score = evaluate_case(case, honest, extraction_available=True).score
    inflated_score = evaluate_case(case, inflated, extraction_available=True).score
    assert inflated_score < honest_score


def test_reciprocal_rank_rewards_leading_with_the_answer():
    """Including the answer at position six is not the same as ranking it first."""
    case = CASES_BY_ID["direct_lookup_gpc"]

    first = _brief(evidence=[_evidence("hotwall", 2), _evidence("zro2_ald", 2)])
    buried = _brief(evidence=[_evidence("zro2_ald", 2), _evidence("hotwall", 2)])

    assert evaluate_case(case, first, extraction_available=False).reciprocal_rank == 1.0
    assert evaluate_case(case, buried, extraction_available=False).reciprocal_rank == 0.5
    #  Recall cannot tell them apart, which is why recall alone scored every policy
    #  identically before these metrics existed.
    assert (
        evaluate_case(case, first, extraction_available=False).doc_recall
        == evaluate_case(case, buried, extraction_available=False).doc_recall
    )


def test_page_precision_penalises_a_window_full_of_distractors():
    case = CASES_BY_ID["direct_lookup_gpc"]
    focused = _brief(evidence=[_evidence("hotwall", 2)])
    noisy = _brief(evidence=[
        _evidence("hotwall", 2), _evidence("hfsiox", 2), _evidence("zro2_ald", 2),
    ])

    assert evaluate_case(case, focused, extraction_available=False).page_precision == 1.0
    precise = evaluate_case(case, noisy, extraction_available=False)
    assert precise.page_precision == pytest.approx(1 / 3)
    assert any("known distractor" in d for d in precise.diagnostics)


def test_an_incomparable_expectation_is_satisfied_by_recording_it_as_such():
    """The review states 25 with no context; inventing context would be the error."""
    case = CASES_BY_ID["contextless_value"]
    honest = _brief(
        evidence=[_evidence("contextless", 1)],
        claims=[ExtractedClaim(
            field_name="k", value=25.0,
            evidence=[_evidence("contextless", 1,
                                quote="HfO2 offers a relative permittivity of 25")],
            context={},  # nothing stated, nothing invented
        )],
    )
    result = evaluate_case(case, honest, extraction_available=True)
    assert result.context_completeness == 1.0

    invented = _brief(
        evidence=[_evidence("contextless", 1)],
        claims=[ExtractedClaim(
            field_name="k", value=25.0, units="1",
            evidence=[_evidence("contextless", 1,
                                quote="HfO2 offers a relative permittivity of 25")],
            context={"temperature_k": 300.0, "frequency_hz": 1e4},  # not in the source
        )],
    )
    assert evaluate_case(case, invented, extraction_available=True).context_completeness == 0.0


def test_a_value_outside_tolerance_does_not_match():
    case = CASES_BY_ID["density_lookup"]
    expected = case.expected_claims[0]
    assert isinstance(expected, ExpectedClaim)

    wrong = _brief(
        evidence=[_evidence("hotwall", 3)],
        claims=[ExtractedClaim(
            field_name="rho", value=4.0, units="g/cm3",
            evidence=[_evidence("hotwall", 3, quote="mass density of 9.1 g/cm3")],
        )],
    )
    result = evaluate_case(case, wrong, extraction_available=True)
    assert result.extraction_recall == 0.0
    assert any("Missed claim rho" in d for d in result.diagnostics)


def test_a_known_disagreement_must_be_named_not_merely_found():
    case = CASES_BY_ID["cross_paper_disagreement"]
    brief = _brief(evidence=[_evidence("hotwall", 2), _evidence("crossflow", 2)])
    result = evaluate_case(case, brief, extraction_available=True)
    assert result.contradiction_detected is False
    assert any("was not named" in d for d in result.diagnostics)


# --- the runner ----------------------------------------------------------


def test_a_run_is_offline_and_scores_the_cases_it_can():
    outcome = run_benchmark(case_set="baseline", results_path=None)
    result = outcome.result

    assert outcome.status == "keep"
    assert result.extraction_available is False
    #  The fixture corpus has no embeddings, so the run is lexical-only and says so
    #  rather than claiming a retriever it did not have.
    assert result.retrievers == ("lexical",)
    assert "lexical-only" in result.notes
    assert result.n_cases_skipped == 3  # the abstention cases
    assert result.overall_score > 0.5


def test_a_run_leaves_the_truth_data_untouched():
    before = [(c.case_id, c.expected_pages, c.expected_claims) for c in CASES]
    run_benchmark(case_set="baseline", results_path=None)
    after = [(c.case_id, c.expected_pages, c.expected_claims) for c in CASES]
    assert before == after


def test_the_benchmark_distinguishes_policies():
    """Its whole purpose. Grading demonstrably buys score."""
    comparison = compare_policies(["no_grading"], case_set="answerable", results_path=None)
    baseline = comparison["policies"]["baseline"]["score"]
    ungraded = comparison["policies"]["no_grading"]["score"]
    assert ungraded < baseline
    assert comparison["policies"]["no_grading"]["status"] == "discard"


def test_table_weighting_still_earns_its_place():
    """A knob that changes nothing measurable should not be in the policy.

    It used to earn its place on the tabular case. Scoring the document title as well
    as the chunk text then lifted the baseline's tabular score to match it — the two
    fixes address the same cause, a table row carrying none of its document's subject.
    Table weighting now earns its place on *technique disambiguation* instead, where
    the competing passages are prose and the title fix does not separate them.
    """
    from cnms_fom.research.benchmark.runner import run_benchmark as run

    plain = run(policy=get_policy("baseline"), case_set="hard", results_path=None).result
    biased = run(policy=get_policy("table_biased"), case_set="hard", results_path=None).result

    assert biased.overall_score > plain.overall_score
    assert (
        biased.by_category()["technique_disambiguation"]
        > plain.by_category()["technique_disambiguation"]
    )
    #  And it still leads on ranking, which is what a weight acts on.
    assert biased.reciprocal_rank >= plain.reciprocal_rank


def test_the_combined_policy_beats_both_of_its_parts():
    """`focused` exists because the two winners fix different cases."""
    from cnms_fom.research.benchmark.runner import run_benchmark as run

    scores = {
        name: run(policy=get_policy(name), case_set="baseline", results_path=None).result
        for name in ("baseline", "narrow_pool", "table_biased", "focused")
    }
    assert scores["focused"].overall_score > scores["narrow_pool"].overall_score
    assert scores["focused"].overall_score > scores["table_biased"].overall_score
    #  The check that matters: a tighter window must not win by failing the
    #  multi-source case. page_recall is no longer 1.0 for *any* policy — truth-set v3
    #  added a compound question whose two conjuncts sit on two different pages and the
    #  second is not surfaced — so the claim is now made as a comparison rather than an
    #  absolute. Recorded rather than asserted away: that miss is the discriminating
    #  signal the suite previously lacked.
    assert scores["focused"].doc_recall == 1.0
    assert scores["focused"].page_recall >= scores["baseline"].page_recall
    assert (
        scores["focused"].by_category()["cross_paper_disagreement"]
        >= scores["baseline"].by_category()["cross_paper_disagreement"]
    )
    #  And it costs less: half the passages to extract from.
    assert get_policy("focused").top_k < get_policy("baseline").top_k


def test_a_crashed_case_does_not_lose_the_run():
    class Exploding(StubExtractor):
        def send(self, system, messages, *, tools=None, temperature: float = 0.0):
            raise RuntimeError("boom")

    outcome = run_benchmark(case_set="hard", provider=Exploding(), results_path=None)
    #  Every case crashed, but the run completed and recorded why.
    assert outcome.status in ("keep", "discard")
    assert all(
        any("crashed" in d for d in r.diagnostics) or r.score == 0.0
        for r in outcome.result.results
    )


def test_a_verdict_discards_a_run_that_raises_the_unsupported_claim_rate():
    """A gain in the aggregate bought with fabrication is not a gain."""
    from cnms_fom.research.benchmark.runner import _verdict

    baseline = BenchmarkResult(policy_version="base", case_set="s")
    baseline.results = [CaseResult(case_id="a", category="x", doc_recall=0.5, n_claims=2)]

    candidate = BenchmarkResult(policy_version="cand", case_set="s")
    candidate.results = [
        CaseResult(case_id="a", category="x", doc_recall=1.0, n_claims=2, unsupported_claims=1)
    ]
    status, reason = _verdict(candidate, baseline)
    assert status == "discard"
    assert "unsupported claims rose" in reason


def test_a_verdict_discards_a_run_that_loses_abstention():
    from cnms_fom.research.benchmark.runner import _verdict

    def _result(version, abstention_correct):
        result = BenchmarkResult(policy_version=version, case_set="s")
        result.results = [
            CaseResult(case_id="a", category="x", doc_recall=1.0,
                       abstained=True, abstention_correct=abstention_correct),
        ]
        return result

    status, reason = _verdict(_result("cand", False), _result("base", True))
    assert status == "discard"
    assert "abstention_f1 fell" in reason


# --- the results file ----------------------------------------------------


def test_results_are_appended_as_tab_separated_rows(tmp_path):
    path = tmp_path / "results.tsv"
    outcome = run_benchmark(case_set="hard", results_path=path)

    assert outcome.results_path == path
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == list(RESULT_COLUMNS)
    assert len(lines) == 2
    #  Every cell is present and tab-free.
    assert len(lines[1].split("\t")) == len(RESULT_COLUMNS)

    run_benchmark(case_set="hard", results_path=path)
    assert len(path.read_text().splitlines()) == 3  # appended, header written once

    rows = read_results(path)
    assert len(rows) == 2
    assert rows[0]["status"] in ("keep", "discard", "crash")
    #  The row names the code and configuration that produced it.
    assert rows[0]["commit"]
    assert "policy=" in rows[0]["description"]
    assert "extraction=stub" in rows[0]["description"]


def test_an_unavailable_metric_is_written_as_not_applicable(tmp_path):
    path = tmp_path / "results.tsv"
    run_benchmark(case_set="answerable", results_path=path)
    row = read_results(path)[0]
    #  extraction_f1 cannot be computed by a stub run, and must not read as 0.0000.
    assert row["extraction_f1"] == "n/a"


def test_a_dirty_tree_is_marked_in_the_commit_cell():
    """A row from an uncommitted tree does not identify the code that ran."""
    commit = git_commit()
    assert commit
    if commit != "unknown":
        assert commit.endswith("-dirty") or len(commit) >= 7


def test_a_crashed_run_is_still_recorded(tmp_path):
    path = tmp_path / "results.tsv"
    outcome = RunOutcome(
        result=BenchmarkResult(policy_version="p", case_set="s", git_commit="abc1234"),
        status="crash", description="deliberate", error="RuntimeError: boom",
    )
    append_result(path, outcome)
    row = read_results(path)[0]
    assert row["status"] == "crash"
    assert "RuntimeError: boom" in row["description"]


# --- an abstention case has to be scorable -------------------------------
#
# Found on the first real-provider run: two cases abstained correctly and still
# scored 0.000, listed as failures with no diagnostic to explain why.


def test_a_correct_abstention_scores_full_marks():
    """An abstention case computes no other metric, so it must score on this one.

    Previously `SCORE_WEIGHTS` looked up `abstention_f1` on the case — a run-level
    metric no case carries — so `available` came out empty and the case scored a
    hard 0.0 no matter what it did.
    """
    case = CaseResult(
        case_id="absent_sputtering", category="insufficient_evidence",
        abstained=True, abstention_correct=True,
    )
    assert case.score == 1.0


def test_a_failed_abstention_scores_zero():
    case = CaseResult(
        case_id="absent_gaas_on_ge", category="insufficient_evidence",
        abstained=False, abstention_correct=False, n_claims=4,
    )
    assert case.score == 0.0


def test_a_correct_and_a_failed_abstention_are_distinguishable():
    """The property that was broken: the two were identical at 0.0."""
    good = CaseResult(case_id="a", category="insufficient_evidence",
                      abstained=True, abstention_correct=True)
    bad = CaseResult(case_id="b", category="insufficient_evidence",
                     abstained=False, abstention_correct=False)
    assert good.score > bad.score


def test_a_wrong_abstention_on_an_answerable_case_costs_something():
    """Abstaining when the evidence is there is a failure too, not a free pass."""
    answered = CaseResult(case_id="a", category="x", doc_recall=1.0,
                          page_recall=1.0, abstention_correct=True)
    refused = CaseResult(case_id="b", category="x", doc_recall=1.0,
                         page_recall=1.0, abstained=True, abstention_correct=False)
    assert refused.score < answered.score


def test_the_insufficient_evidence_category_reflects_real_performance():
    """Two of three right should not read as 0.0000."""
    result = BenchmarkResult(policy_version="p", case_set="baseline")
    result.results = [
        CaseResult(case_id="a", category="insufficient_evidence",
                   abstained=True, abstention_correct=True),
        CaseResult(case_id="b", category="insufficient_evidence",
                   abstained=True, abstention_correct=True),
        CaseResult(case_id="c", category="insufficient_evidence",
                   abstained=False, abstention_correct=False),
    ]
    assert result.by_category()["insufficient_evidence"] == pytest.approx(2 / 3)
    #  And the run-level F1 agrees with the case-level view instead of contradicting it.
    assert result.abstention_f1 == pytest.approx(0.8)


def test_every_score_weight_resolves_to_a_case_attribute():
    """Guards the class of bug: a weight naming a field no case carries.

    The abstention weight was dead for exactly this reason, and a silent `getattr`
    default is what hid it.
    """
    from cnms_fom.research.benchmark.evaluate import PER_CASE_METRIC, SCORE_WEIGHTS

    blank = CaseResult(case_id="a", category="x")
    for name in SCORE_WEIGHTS:
        attr = PER_CASE_METRIC.get(name, name)
        assert hasattr(blank, attr), f"SCORE_WEIGHTS['{name}'] names no case attribute"


# --- the cache has to survive the throwaway database ----------------------
#
# The corpus database is rebuilt per run by design, and the cache originally lived
# in it — so every run started cold and a policy sweep paid full model cost for
# each policy over the same passages, which is the one workload it was built for.


class _CountingExtractor(StubExtractor):
    """A stub that records how many times it was actually asked."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_calls = 0

    def send(self, system, messages, *, tools=None, temperature: float = 0.0):
        self.n_calls += 1
        return super().send(system, messages, tools=tools, temperature=temperature)


def test_a_second_run_reuses_the_first_run_s_model_calls(tmp_path):
    cache = tmp_path / "cache.db"

    cold = _CountingExtractor()
    run_benchmark(case_set="hard", provider=cold, results_path=None, cache_path=cache)
    assert cold.n_calls > 0, "the first run has to actually call the model"

    warm = _CountingExtractor()
    run_benchmark(case_set="hard", provider=warm, results_path=None, cache_path=cache)
    assert warm.n_calls < cold.n_calls


def test_the_cache_survives_in_the_file_not_the_corpus_database(tmp_path):
    cache = tmp_path / "cache.db"
    run_benchmark(case_set="hard", provider=StubExtractor(), results_path=None,
                  cache_path=cache)

    assert cache.exists()
    import sqlite3

    with sqlite3.connect(cache) as conn:
        tables = {r[0] for r in conn.execute(
            "select name from sqlite_master where type='table'"
        )}
    #  Only the cache. A benchmark cache holding corpus or results would be a second
    #  source of truth for what the fixtures say.
    assert tables == {"llm_cache"}


def test_a_cached_run_produces_the_same_score(tmp_path):
    """The cache must be invisible in the result, or it is not a cache."""
    cache = tmp_path / "cache.db"
    cold = run_benchmark(case_set="hard", provider=StubExtractor(), results_path=None,
                         cache_path=cache)
    warm = run_benchmark(case_set="hard", provider=StubExtractor(), results_path=None,
                         cache_path=cache)

    assert warm.result.overall_score == pytest.approx(cold.result.overall_score)
    assert warm.result.by_category() == pytest.approx(cold.result.by_category())


def test_cache_path_none_disables_it(tmp_path):
    a = _CountingExtractor()
    run_benchmark(case_set="hard", provider=a, results_path=None, cache_path=None)
    b = _CountingExtractor()
    run_benchmark(case_set="hard", provider=b, results_path=None, cache_path=None)
    assert b.n_calls == a.n_calls


def test_an_unwritable_cache_path_does_not_fail_the_run(tmp_path):
    """A cache is an optimisation; losing it must not lose the score."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    outcome = run_benchmark(
        case_set="hard", provider=StubExtractor(), results_path=None,
        cache_path=blocker / "cache.db",
    )
    assert outcome.status in ("keep", "discard")


# --- unit spellings ------------------------------------------------------
#
# The extractor copies the passage's units verbatim, by design, so the scorer has
# to treat spellings of one unit as one unit. Otherwise it measures which spelling
# the source happened to use.


def test_the_same_unit_spelled_differently_compares_equal():
    from cnms_fom.research.benchmark.evaluate import normalise_units

    for spelling in ("A/cycle", "angstrom per cycle", "Angstrom/cycle", "Å/cycle",
                     "a / cycle", " A/CYCLE "):
        assert normalise_units(spelling) == normalise_units("A/cycle"), spelling


def test_genuinely_different_units_still_differ():
    from cnms_fom.research.benchmark.evaluate import normalise_units

    #  The failure this guards: normalising so hard that a length and a rate match.
    assert normalise_units("angstrom") != normalise_units("angstrom per cycle")
    assert normalise_units("nm/min") != normalise_units("nm")
    assert normalise_units("mTorr") != normalise_units("Torr")
    assert normalise_units("degC") != normalise_units("K")


def test_missing_units_normalise_without_raising():
    from cnms_fom.research.benchmark.evaluate import normalise_units

    assert normalise_units(None) == ""
    assert normalise_units("") == ""


def test_unit_accuracy_is_none_when_no_claim_matched():
    """A case that extracted nothing has no units to be right or wrong about.

    Scoring it 0.0 said the units were wrong and dragged the run mean down: the
    metric read 0.7778 while every unit actually checked was correct.
    """
    case = CaseResult(case_id="a", category="x", unit_accuracy=None)
    assert case.unit_accuracy is None

    result = BenchmarkResult(policy_version="p", case_set="baseline")
    result.results = [
        CaseResult(case_id="matched", category="x", unit_accuracy=1.0),
        CaseResult(case_id="matched_nothing", category="x", unit_accuracy=None),
    ]
    #  The mean skips the case with nothing to measure instead of averaging in a zero.
    assert result.unit_accuracy == pytest.approx(1.0)


def test_a_genuine_unit_error_still_lowers_unit_accuracy():
    """The guard must not make the metric unable to report a real failure."""
    result = BenchmarkResult(policy_version="p", case_set="baseline")
    result.results = [
        CaseResult(case_id="good", category="x", unit_accuracy=1.0),
        CaseResult(case_id="bad", category="x", unit_accuracy=0.0),
    ]
    assert result.unit_accuracy == pytest.approx(0.5)


# --- gold-key aliases (truth-set v2) --------------------------------------
#
# §10e proved two of three benchmark "misses" were not misses: the values were
# extracted, passed every guard, and appeared in the brief under the registry keys
# the extract-v4 prompt mandates. The gold field names were wrong.


def test_a_gold_name_accepts_its_reviewed_registry_alias():
    from cnms_fom.research.benchmark.evaluate import _keys_matching

    assert "temperature_c" in _keys_matching("substrate_temperature")
    assert "pressure_torr" in _keys_matching("oxygen_pressure")
    #  The gold name itself must still match.
    assert "substrate_temperature" in _keys_matching("substrate_temperature")


def test_an_unaliased_key_matches_only_itself():
    from cnms_fom.research.benchmark.evaluate import _keys_matching

    assert _keys_matching("growth_per_cycle_ang") == ("growth_per_cycle_ang",)
    assert _keys_matching("rho") == ("rho",)


def test_aliasing_is_not_transitive_or_reversed():
    """`temperature_c` must not start accepting `substrate_temperature`."""
    from cnms_fom.research.benchmark.evaluate import _keys_matching

    assert "substrate_temperature" not in _keys_matching("temperature_c")


def test_a_wrong_dimension_key_still_fails_to_match():
    """The alias table must not become a general key-loosening mechanism."""
    from cnms_fom.research.benchmark.evaluate import _keys_matching

    #  A pressure is not a temperature, however close the gold name sounds.
    assert "pressure_torr" not in _keys_matching("substrate_temperature")
    assert "temperature_c" not in _keys_matching("oxygen_pressure")
    #  And nothing unrelated sneaks in.
    for gold in ("substrate_temperature", "oxygen_pressure"):
        assert "growth_per_cycle_ang" not in _keys_matching(gold)
        assert "rho" not in _keys_matching(gold)


def test_every_alias_target_is_a_real_registry_or_context_field():
    """An alias pointing at a key nothing emits would be dead configuration."""
    from cnms_fom.research.benchmark.evaluate import GOLD_KEY_ALIASES
    from cnms_fom.research.contracts import (
        CLAIM_CONTEXT_FIELDS,
        FIELD_DIMENSION,
        REQUIRED_CONTEXT,
    )

    known = set(CLAIM_CONTEXT_FIELDS) | set(FIELD_DIMENSION) | set(REQUIRED_CONTEXT)
    for gold, aliases in GOLD_KEY_ALIASES.items():
        for alias in aliases:
            assert alias in known, f"{gold} aliases {alias}, which is not a known field"


def test_the_case_set_version_is_recorded():
    """v1 and v2 metrics are different experiments and must be labellable as such."""
    from cnms_fom.research.benchmark.cases import CASE_SET_VERSION

    assert CASE_SET_VERSION
    #  Deliberately not pinned to a number: the point is that a version exists at all, so
    #  two expectation sets are never compared as one experiment. Pinning "v2" made the
    #  bump to v3 fail a test that was asserting nothing useful.
    assert re.match(r"^v\d+-[a-z-]+$", CASE_SET_VERSION), CASE_SET_VERSION


# --- truth-set v3: compound questions and forbidden claims -----------------
#
# The suite had no compound question, which is why three separate changes
# (grade-v3, lexical_relaxed, and next per-conjunct grading) could not be judged on
# it: it saw their cost and none of their benefit. It also had no way to assert that
# a fabrication stays absent, so the §10d failures could not become regressions.


def test_the_suite_has_compound_questions():
    from cnms_fom.research.benchmark.cases import get_case_set

    compound = get_case_set("compound")
    assert len(compound) >= 2
    for case in compound:
        #  A compound question asks for more than one thing; "and" is the cheap proxy
        #  and every one of these was written to have at least two conjuncts.
        assert " and " in case.question, case.case_id


def test_dev_and_holdout_partition_the_suite():
    """Tuning on dev only means dev and holdout must not share a case."""
    from cnms_fom.research.benchmark.cases import CASE_SETS, CASES

    dev, holdout = set(CASE_SETS["dev"]), set(CASE_SETS["holdout"])
    assert not dev & holdout, f"overlap: {sorted(dev & holdout)}"
    assert dev and holdout
    #  Nothing orphaned: a case in neither split is a case nobody looks at.
    assert {c.case_id for c in CASES} == dev | holdout


def test_both_splits_contain_a_compound_case():
    """Otherwise a compound-question change could be tuned without being validated."""
    from cnms_fom.research.benchmark.cases import CASE_SETS, CASES_BY_ID

    for split in ("dev", "holdout"):
        categories = {CASES_BY_ID[cid].category for cid in CASE_SETS[split]}
        assert "compound_question" in categories, split


def test_a_forbidden_claim_zeroes_the_case():
    """A reproduced fabrication is not a partial success."""
    good = CaseResult(case_id="a", category="x", doc_recall=1.0, page_recall=1.0)
    bad = CaseResult(case_id="b", category="x", doc_recall=1.0, page_recall=1.0,
                     forbidden_present=1)
    assert good.score > 0
    assert bad.score == 0.0


def test_a_dose_time_as_growth_per_cycle_is_caught_as_forbidden():
    """The §10d regression, now assertable: purge times filed as growth per cycle."""
    from cnms_fom.research.benchmark.cases import CASES_BY_ID
    from cnms_fom.research.benchmark.evaluate import evaluate_case

    case = CASES_BY_ID["dimensional_negative_cycle_timing"]

    #  This claim cannot be *constructed* any more: the dimensional guard rejects seconds
    #  under an Angstrom-declaring field at __post_init__. That is the first line of
    #  defence, and it is tested in test_research_contracts.py. The forbidden rule is the
    #  second, and exists for a future change that weakens the first — so the shape is
    #  assembled legally and then mutated, which is exactly what such a regression would
    #  look like from the benchmark's side.
    claim = ExtractedClaim(
        field_name="purge_duration_s", value=6.0, units="s", tier=ClaimTier.REPORTED,
        evidence=[EvidenceItem(
            document_id=1,
            document_title="SYNTHETIC ALD of HfO2 on Si(100) from TDMAH and water "
                           "(hot-wall)",
            page=1, quote="a 6 s N2 purge")],
    )
    claim.field_name = "growth_per_cycle_ang"

    brief = ResearchBrief(research_question=case.question, claims=[claim])
    result = evaluate_case(case, brief, extraction_available=True)
    assert result.forbidden_present >= 1
    assert result.score == 0.0
    assert any("FORBIDDEN" in d for d in result.diagnostics)


def test_a_manufactured_disagreement_is_caught_as_forbidden():
    """The mirror of cross_paper_disagreement: only one source states a density."""
    from cnms_fom.research.benchmark.cases import CASES_BY_ID
    from cnms_fom.research.benchmark.evaluate import evaluate_case
    from cnms_fom.research.contracts import Contradiction

    case = CASES_BY_ID["negative_disagreement_density"]
    ev = [EvidenceItem(document_id=1, document_title="t", page=3, quote="9.1 g/cm3")]
    left = ExtractedClaim(field_name="rho", value=9.1, units="g/cm3", evidence=ev,
                          tier=ClaimTier.MEASURED)
    right = ExtractedClaim(field_name="rho", value=8.7, units="g/cm3", evidence=ev,
                           tier=ClaimTier.MEASURED)
    brief = ResearchBrief(
        research_question=case.question, claims=[left, right],
        contradictions=[Contradiction(field_name="rho", left=left, right=right,
                                      basis="test")],
    )
    result = evaluate_case(case, brief, extraction_available=True)
    assert result.forbidden_present >= 1
    assert result.score == 0.0
    assert any("manufactured disagreement" in d for d in result.diagnostics)


def test_a_clean_brief_trips_no_forbidden_rule():
    """The guard must not fire on a correct answer."""
    from cnms_fom.research.benchmark.cases import CASES_BY_ID
    from cnms_fom.research.benchmark.evaluate import evaluate_case

    case = CASES_BY_ID["dimensional_negative_cycle_timing"]
    brief = ResearchBrief(
        research_question=case.question,
        claims=[
            ExtractedClaim(
                field_name="purge_duration_s", value=6.0, units="s",
                tier=ClaimTier.REPORTED,
                evidence=[EvidenceItem(document_id=1, document_title="t", page=1,
                                       quote="a 6 s N2 purge")],
            )
        ],
    )
    result = evaluate_case(case, brief, extraction_available=True)
    assert result.forbidden_present == 0


def test_the_unit_variant_gold_matches_a_canonical_claim():
    """Gold in nm/cycle against a claim in A/cycle: the same number, both ways."""
    from cnms_fom.research.benchmark.cases import CASES_BY_ID
    from cnms_fom.research.benchmark.evaluate import evaluate_case

    case = CASES_BY_ID["unit_variant_gpc"]
    brief = ResearchBrief(
        research_question=case.question,
        claims=[
            ExtractedClaim(
                field_name="growth_per_cycle_ang", value=0.98, units="A/cycle",
                tier=ClaimTier.MEASURED,
                evidence=[EvidenceItem(
                    document_id=1,
                    document_title="SYNTHETIC ALD of HfO2 on Si(100) from TDMAH and "
                                   "water (hot-wall)",
                    page=2, quote="0.98 angstrom per cycle")],
            )
        ],
    )
    result = evaluate_case(case, brief, extraction_available=True)
    assert result.extraction_recall == pytest.approx(1.0), (
        "a normalisation mismatch must not read as a recall failure: "
        f"{result.diagnostics}"
    )
