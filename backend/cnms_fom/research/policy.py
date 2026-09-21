"""The knobs an autoresearch experiment is allowed to turn.

Every setting that affects *how* evidence is found, graded, extracted, and
interpreted lives here, in one frozen object with a version string.  Two
consequences, both of which the benchmark depends on:

* A brief records the policy that produced it, so a result is attributable. A run
  that cannot name its own configuration is not an experiment, it is an anecdote.
* A policy change is a diff on one object rather than a scattered set of keyword
  arguments, so "what did we change?" has an answer that is not archaeology.

What is deliberately *not* here: anything from the FOM protocol. No weights, no
normalization bounds, no hypothesis signs, no floor. Those are versioned science
(Sec. 5.3) and a retrieval experiment has no business touching them — a policy
sweep that moved a FOM bound would be measuring two things at once and could
attribute the result to neither.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

#  Prompt identifiers. Bumped when the prompt text changes, because an extraction
#  is only reproducible if the prompt that produced it is identifiable — and a
#  silently edited prompt is the most common way a benchmark result stops meaning
#  what it meant.
#  v3 added the registry-key glosses; v4 added temperature_c. Bumped for the second
#  change as well, even though v3 was never shipped, because a cache entry written
#  under a version label must have been produced by that exact prompt — that is the
#  whole guarantee, and "it was only a small edit" is how a cache starts lying.
EXTRACTION_PROMPT_VERSION = "extract-v4"
INTERPRETATION_PROMPT_VERSION = "interpret-v1"


@dataclass(frozen=True)
class ResearchPolicy:
    """One candidate configuration of the evidence pipeline."""

    name: str = "baseline"

    # --- chunking (applies at ingest; recorded so a brief knows the corpus shape)
    chunk_size: int = 1_000
    chunk_overlap: int = 200

    # --- retrieval
    #  How many candidates each retriever returns before fusion. Fusion can only
    #  promote what a retriever surfaced, so a narrow pool makes the second
    #  retriever pointless.
    candidate_depth: int = 12
    #  How many graded passages reach the extractor.
    top_k: int = 6
    use_dense: bool = True
    use_lexical: bool = True
    #  Reciprocal-rank-fusion damping. 60 is the TREC value; exposed because it is
    #  a knob, not because it is expected to move.
    rrf_k: int = 60
    #  Applied to the dense leg before fusion. A lexical hit has no cosine
    #  similarity to threshold.
    min_similarity: float = 0.2

    # --- grading
    grade: bool = True
    #  A passage below this does not reach the extractor. 2 = "contains part of
    #  what the question asks for".
    min_useful_grade: int = 2
    #  How many useful passages count as enough to answer from.
    min_useful_passages: int = 1
    allow_query_rewrite: bool = True

    # --- document and passage preference
    #  Multiplier on the fused score for passages that look like a table or a
    #  figure caption. Process parameters live in tables far more often than in
    #  prose, and a fused ranking that treats them alike buries them.
    table_weight: float = 1.0
    caption_weight: float = 1.0
    #  Restrict retrieval to these corpus partitions. None means all.
    techniques: tuple[str, ...] | None = None

    # --- extraction and interpretation
    extraction_prompt_version: str = EXTRACTION_PROMPT_VERSION
    interpretation_prompt_version: str = INTERPRETATION_PROMPT_VERSION
    #  Claims below this model-reported confidence are recorded but marked, never
    #  dropped: a discarded extraction is invisible, and an extractor that is
    #  systematically underconfident would look like a corpus with no content.
    low_confidence_threshold: float = 0.4
    #  Require complete context before a claim counts as comparable. Off would let
    #  a permittivity with no frequency be set beside one that has it, which
    #  Sec. 16 forbids; exposed only so the benchmark can measure the cost.
    require_complete_context: bool = True

    # --- abstention
    #  Abstain when fewer than this many useful passages survive grading.
    abstain_below_passages: int = 1
    #  Abstain when no retrieved passage is locatable to a page.
    abstain_without_page: bool = True

    # --- models
    #  Each stage may use a different model, because they are different tasks.
    #  Unset means fall back to the configured default for that stage, then to the
    #  chat model. Grading is a 0-3 classification and extraction is structured
    #  output; neither needs the answer model's reasoning, and both run once per
    #  passage, so they dominate the wall clock.
    chat_model: str | None = None
    grader_model: str | None = None
    extraction_model: str | None = None
    provider: str | None = None

    notes: str = ""

    def __post_init__(self) -> None:
        if not (self.name or "").strip():
            raise ValueError("A policy needs a name.")
        if not (0 <= self.min_useful_grade <= 3):
            raise ValueError(f"min_useful_grade must be 0-3, got {self.min_useful_grade}.")
        if self.top_k > self.candidate_depth:
            raise ValueError(
                f"top_k ({self.top_k}) exceeds candidate_depth ({self.candidate_depth}): the "
                "pipeline cannot return more passages than it retrieved."
            )
        if not (self.use_dense or self.use_lexical):
            raise ValueError("A policy must enable at least one retriever.")

    @property
    def version(self) -> str:
        """``name@hash`` — stable, and changes whenever any knob changes."""
        return f"{self.name}@{self.fingerprint()[:12]}"

    def fingerprint(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k not in ("name", "notes")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def evolve(self, **changes) -> ResearchPolicy:
        """A copy with some knobs changed. The unit of a benchmark experiment."""
        unknown = set(changes) - set(asdict(self))
        if unknown:
            raise ValueError(f"Unknown policy fields: {sorted(unknown)}")
        return replace(self, **changes)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["version"] = self.version
        payload["fingerprint"] = self.fingerprint()
        return payload

    def diff(self, other: ResearchPolicy) -> dict[str, tuple]:
        """What differs between two policies, for a benchmark report."""
        mine, theirs = asdict(self), asdict(other)
        return {
            key: (mine[key], theirs[key])
            for key in mine
            if key not in ("name", "notes") and mine[key] != theirs[key]
        }


BASELINE = ResearchPolicy(
    name="baseline",
    notes="The shipped defaults: hybrid retrieval, grading on, one corrective rewrite.",
)

#  Candidate policies worth trying, kept here so a benchmark run names a policy
#  rather than inlining a pile of keyword arguments.
CANDIDATES: dict[str, ResearchPolicy] = {
    "baseline": BASELINE,
    "no_grading": BASELINE.evolve(
        name="no_grading", grade=False,
        notes="Grading off. Cheaper and faster; measures what grading actually buys.",
    ),
    "lexical_only": BASELINE.evolve(
        name="lexical_only", use_dense=False,
        notes="No embeddings. Measures how much of this corpus is reachable by exact tokens.",
    ),
    "dense_only": BASELINE.evolve(
        name="dense_only", use_lexical=False,
        notes="No lexical leg. Measures what rare exact tokens cost when only vectors look.",
    ),
    "strict_grading": BASELINE.evolve(
        name="strict_grading", min_useful_grade=3,
        notes="Only passages that directly answer. Higher precision, more abstentions.",
    ),
    "wide_pool": BASELINE.evolve(
        name="wide_pool", candidate_depth=24, top_k=10,
        notes="Twice the candidate pool. Costs grading calls; may recover buried passages.",
    ),
    "no_rewrite": BASELINE.evolve(
        name="no_rewrite", allow_query_rewrite=False,
        notes="No corrective rewrite. Measures how many misses are vocabulary misses.",
    ),
    "table_biased": BASELINE.evolve(
        name="table_biased", table_weight=1.5, caption_weight=1.3,
        notes="Prefer tabular and caption passages, where process parameters usually live.",
    ),
    "narrow_pool": BASELINE.evolve(
        name="narrow_pool", candidate_depth=6, top_k=3,
        notes="Half the pool and half the passages. Cheaper — grading and extraction both run "
        "once per passage — and measurably more precise on this corpus, though a larger one "
        "would eventually pay for it in recall.",
    ),
    #  The two candidates that beat the baseline improve different cases:
    #  table weighting fixes technique disambiguation, a narrow pool fixes precision.
    #  Neither subsumes the other, so the combination is the obvious next experiment.
    "focused": BASELINE.evolve(
        name="focused", candidate_depth=8, top_k=3, table_weight=1.5, caption_weight=1.3,
        notes="narrow_pool plus table weighting. Half the extraction cost of the baseline, and "
        "the two improvements it combines fix different cases.",
    ),
}


def get_policy(name: str) -> ResearchPolicy:
    if name not in CANDIDATES:
        raise KeyError(f"Unknown policy {name!r}. Available: {sorted(CANDIDATES)}")
    return CANDIDATES[name]
