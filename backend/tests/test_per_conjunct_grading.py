"""Per-conjunct grading: the fix for §10e's measured grade dilution.

Step 0 of Phase 2 measured the cause with a 2x3 matrix. The passage carrying
0.98 A/cycle graded **2** on "What growth per cycle is reported for HfO2 ALD?" and
**1** on the three-part compound question; the control passage carrying 1.42 graded
3 and 2. So the compound form costs exactly one grade point, deterministically, and
only the passage that started at 2 fell below ``MIN_USEFUL_GRADE``.

The fix grades each conjunct separately and keeps the best grade, escalating only for
a passage that would otherwise be dropped.
"""

from __future__ import annotations

import json

from cnms_fom.rag_backend.grading import (
    MIN_USEFUL_GRADE,
    grade_and_rerank,
    split_conjuncts,
)
from cnms_fom.rag_backend.hybrid import FusedHit
from cnms_fom.rag_backend.providers import ChatResult
from cnms_fom.rag_backend.vectorstore import ChunkHit

COMPOUND = (
    "What growth per cycle and film density are reported for HfO2 ALD, "
    "and do the sources agree?"
)


# --- splitting -------------------------------------------------------------


def test_a_compound_question_splits_and_keeps_the_original_first():
    parts = split_conjuncts(COMPOUND)
    assert parts[0] == COMPOUND, "the original must be the query of record"
    assert len(parts) >= 3


def test_a_single_clause_question_is_not_split():
    for question in (
        "What growth per cycle is reported for HfO2 ALD?",
        "Does the literature agree on the growth per cycle for HfO2 ALD?",
        "What substrate temperature should be used for MBE of GaAs on germanium?",
    ):
        assert split_conjuncts(question) == [question]


def test_fragments_carry_the_subject_forward():
    """A fragment with no material or technique is not gradeable."""
    parts = split_conjuncts(COMPOUND)
    for fragment in parts[1:]:
        assert "HfO2" in fragment or "ALD" in fragment, fragment


def test_a_two_word_fragment_is_discarded_rather_than_graded():
    #  "and so on" must not become a grading query.
    assert split_conjuncts("What is the density and so on?") == [
        "What is the density and so on?"
    ]


def test_splitting_is_stable():
    assert split_conjuncts(COMPOUND) == split_conjuncts(COMPOUND)


# --- grading ---------------------------------------------------------------


class _DilutingGrader:
    """Reproduces the measured behaviour: 1 for the compound form, 2 for a conjunct.

    Exactly the §10e matrix, so the test asserts the real mechanism rather than a
    convenient stand-in.
    """

    name = "stub"
    model = "diluting"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def send(self, system, messages, *, tools=None, temperature: float = 0.0):  # noqa: ARG002
        body = " ".join(str(m.get("content", "")) for m in messages)
        self.questions.append(body)
        compound = "do the sources agree" in body
        grade = 1 if compound else 2
        return ChatResult(
            text=json.dumps({"grade": grade, "reason": "compound" if compound else "clause"}),
            model=self.model, provider=self.name,
        )


def _candidate(chunk_id: int = 1) -> FusedHit:
    hit = ChunkHit(
        chunk_id=chunk_id, document_id=1, document_title="hotwall", technique="ald",
        page=2, text="the growth per cycle was constant at 0.98 angstrom per cycle",
        similarity=0.5,
    )
    return FusedHit(hit=hit, rrf_score=0.5, lexical_rank=1)


def test_without_the_policy_the_passage_is_dropped():
    """The status quo, asserted so the fix has something to be measured against."""
    graded, cost = grade_and_rerank(_DilutingGrader(), COMPOUND, [_candidate()])
    assert graded[0].grade == 1
    assert graded[0].useful is False
    assert cost["conjunct_rescued"] == 0


def test_with_the_policy_the_passage_is_rescued():
    graded, cost = grade_and_rerank(
        _DilutingGrader(), COMPOUND, [_candidate()], per_conjunct=True
    )
    assert graded[0].grade >= MIN_USEFUL_GRADE
    assert graded[0].useful is True
    assert cost["conjunct_rescued"] == 1
    #  The reason must say why the grade changed, or the brief cannot be audited.
    assert "conjunct of a compound question" in graded[0].reason


def test_the_escalation_is_monotone():
    """A grade can go up, never down. That is what makes this safe to enable."""
    plain, _ = grade_and_rerank(_DilutingGrader(), COMPOUND, [_candidate()])
    escalated, _ = grade_and_rerank(
        _DilutingGrader(), COMPOUND, [_candidate()], per_conjunct=True
    )
    assert escalated[0].grade >= plain[0].grade


def test_a_passage_already_above_the_threshold_costs_no_extra_calls():
    """Escalation is proportional to the problem: only a would-be-dropped passage."""

    class _Generous(_DilutingGrader):
        def send(self, system, messages, *, tools=None, temperature: float = 0.0):  # noqa: ARG002
            self.questions.append(" ".join(str(m.get("content", "")) for m in messages))
            return ChatResult(text=json.dumps({"grade": 3, "reason": "direct"}),
                              model=self.model, provider=self.name)

    grader = _Generous()
    graded, cost = grade_and_rerank(
        grader, COMPOUND, [_candidate()], per_conjunct=True
    )
    assert graded[0].grade == 3
    assert cost["conjunct_rescued"] == 0
    assert len(grader.questions) == 1, "no conjunct calls for a passage already kept"


def test_a_single_clause_question_costs_no_extra_calls_either():
    grader = _DilutingGrader()
    grade_and_rerank(
        grader, "What growth per cycle is reported for HfO2 ALD?", [_candidate()],
        per_conjunct=True,
    )
    assert len(grader.questions) == 1


def test_an_irrelevant_passage_stays_irrelevant():
    """Rescuing must not promote a passage that no conjunct is about."""

    class _Honest(_DilutingGrader):
        def send(self, system, messages, *, tools=None, temperature: float = 0.0):  # noqa: ARG002
            self.questions.append(" ".join(str(m.get("content", "")) for m in messages))
            return ChatResult(text=json.dumps({"grade": 0, "reason": "irrelevant"}),
                              model=self.model, provider=self.name)

    graded, cost = grade_and_rerank(
        _Honest(), COMPOUND, [_candidate()], per_conjunct=True
    )
    assert graded[0].grade == 0
    assert graded[0].useful is False
    assert cost["conjunct_rescued"] == 0


def test_the_policy_is_off_by_default():
    from cnms_fom.research.policy import BASELINE, get_policy

    assert BASELINE.grade_per_conjunct is False
    assert get_policy("per_conjunct").grade_per_conjunct is True
