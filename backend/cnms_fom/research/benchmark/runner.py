"""Running the benchmark: one policy, one case set, a score and a diagnosis.

Each run builds a **throwaway database**, seeds the frozen corpus into it, and runs
every case.  Nothing touches the working database, and nothing modifies the truth
data — a run produces a score, never an updated expectation.

Results are appended to a tab-separated file with a stable column order, so a
history of runs is greppable, diffable, and readable without a tool.  Every row
carries the git commit and the policy fingerprint, because a result that cannot
name the code and configuration that produced it is not reproducible.

The runner is offline by default: a stub extractor exercises retrieval, citation
validity and abstention with no model server.  Pass a real provider to additionally
score extraction, and the row records which capabilities were actually exercised so
a retrieval-only run is never silently compared with a full one.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cnms_fom.research.benchmark.cases import BenchmarkCase, get_case_set, seed_corpus
from cnms_fom.research.benchmark.evaluate import BenchmarkResult, evaluate_case
from cnms_fom.research.brief import generate_brief
from cnms_fom.research.policy import BASELINE, ResearchPolicy

logger = logging.getLogger(__name__)

#  Tab-separated, with this column order. Literal tabs so the file is readable and
#  greppable; the header is written once when the file is created.
RESULT_COLUMNS: tuple[str, ...] = (
    "commit",
    "overall_score",
    "doc_recall",
    "citation_accuracy",
    "extraction_f1",
    "abstention_f1",
    "unsupported_claim_rate",
    "latency_ms",
    "status",
    "description",
)

DEFAULT_RESULTS_PATH = Path("data/exports/autoresearch_results.tsv")
#  Not under data/exports: this is a rebuildable cache, not a result. Deleting it
#  costs wall-clock on the next run and nothing else.
DEFAULT_CACHE_PATH = Path("data/cache/benchmark_llm_cache.db")

#  Resolved at call time, not bound as a default argument, so the constant above is
#  the single place the path is decided and a test can redirect it. As a default
#  argument it would be captured at import and no monkeypatch could reach it — which
#  is how the suite ended up writing a real cache file into the repo.
_USE_DEFAULT_CACHE: object = object()


def git_commit() -> str:
    """The commit a run was made at, or ``unknown``.

    Recorded rather than assumed: comparing two scores from different code is the
    easiest way to reach a wrong conclusion about a policy.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        commit = result.stdout.strip()
        if not commit:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        #  A dirty tree means the row does not identify the code that ran. Marked
        #  rather than silently recorded as the commit.
        return f"{commit}-dirty" if dirty.stdout.strip() else commit
    except (OSError, subprocess.SubprocessError):
        return "unknown"


#  Fraction of a question's distinctive terms a passage must contain to be graded
#  useful by the stub grader. Chosen so the fixture corpus's answerable questions
#  clear it and its unanswerable ones do not; it is a fixture constant, not a tuned
#  parameter, and the real grader is what a policy sweep actually varies.
STUB_GRADE_COVERAGE = 0.45


class StubExtractor:
    """A deterministic stand-in for the model, so the benchmark runs with no server.

    It does two things. It **grades by term overlap**, which is crude but real: a
    passage sharing few of the question's distinctive terms is graded 1 and does not
    reach the answer, so abstention is exercised rather than assumed. And it
    **extracts nothing**, so extraction metrics come back unavailable rather than
    zero and a retrieval-only score is never mistaken for a complete one.

    With this provider the grader is a fixture, not the thing under test: it makes
    the retrieval, citation and abstention plumbing measurable in CI. Pass a real
    provider to measure the real grader and the real extractor.
    """

    name = "stub"
    model = "stub-extractor"
    #  Declares what the docstring above already says: this provider does not extract.
    #  Load-bearing. `_should_abstain` treats "passages retrieved but no claims in any of
    #  them" as grounds to abstain, which is right for a real extractor and wrong for a
    #  fixture that returns an empty list by design — without this flag every answerable
    #  case abstained offline and the whole suite read ~0.67.
    extracts = False

    def __init__(self, coverage_threshold: float = STUB_GRADE_COVERAGE) -> None:
        self.coverage_threshold = coverage_threshold
        self.calls: list[str] = []

    def send(self, system, messages, *, tools=None, temperature: float = 0.0) -> object:  # noqa: ARG002
        import json

        from cnms_fom.rag_backend.providers import ChatResult

        body = " ".join(str(m.get("content", "")) for m in messages)

        if "You grade whether a retrieved passage" in system:
            self.calls.append("grade")
            grade, reason = self._grade(body)
            return ChatResult(
                text=json.dumps({"grade": grade, "reason": reason}),
                model=self.model, provider=self.name,
            )
        if "You extract quantitative claims" in system:
            self.calls.append("extract")
            return ChatResult(text=json.dumps({"claims": []}),
                              model=self.model, provider=self.name)
        if "You rewrite a failed search query" in system:
            #  No rewrite: a stub that invented a better query would be measuring
            #  its own cleverness rather than the pipeline's.
            self.calls.append("rewrite")
            return ChatResult(text="{}", model=self.model, provider=self.name)

        self.calls.append("other")
        return ChatResult(text="{}", model=self.model, provider=self.name)

    def _grade(self, body: str) -> tuple[int, str]:
        """Grade by how much of the question's vocabulary the passage contains."""
        from cnms_fom.rag_backend.hybrid import tokenise

        question, _, passage = body.partition("Passage")
        terms = set(tokenise(question))
        if not terms:
            return 1, "stub: no distinctive terms in the question"
        present = {term for term in terms if term in passage.lower()}
        coverage = len(present) / len(terms)
        if coverage >= self.coverage_threshold:
            return 3, f"stub: {coverage:.0%} term coverage"
        return 1, f"stub: only {coverage:.0%} term coverage, below {self.coverage_threshold:.0%}"


@dataclass
class RunOutcome:
    """A finished run: its result, its status, and where it was written."""

    result: BenchmarkResult
    status: str  # "keep" | "discard" | "crash"
    description: str
    results_path: Path | None = None
    error: str = ""


def run_benchmark(
    *,
    policy: ResearchPolicy | None = None,
    case_set: str = "baseline",
    provider=None,
    database_url: str | None = None,
    description: str = "",
    baseline: BenchmarkResult | None = None,
    results_path: Path | str | None = DEFAULT_RESULTS_PATH,
    embed_corpus: bool = False,
    cache_path: Path | str | None | object = _USE_DEFAULT_CACHE,
) -> RunOutcome:
    """Run one policy over one case set in a throwaway database.

    ``provider=None`` uses :class:`StubExtractor`, which needs no model server.
    ``baseline`` enables a keep/discard verdict; without one the status is ``keep``
    if the run completed, because there is nothing to compare against.

    ``embed_corpus`` embeds the fixture chunks so a dense policy is comparable with a
    lexical one. It needs a reachable embedder, which is why it is off by default.

    ``cache_path`` is a database that survives the run and holds nothing but the
    per-passage model-call cache. The corpus database is deliberately thrown away, so
    without this the cache started cold on every run and a policy sweep paid full
    model cost for each policy over the same passages — which is the one workload the
    cache was built for. Safe to share because the cache is content-addressed: an
    entry is keyed on the passage text (and the question, for grading) plus the model
    and prompt version, so a changed prompt or model misses rather than lies.
    ``cache_path=None`` disables it and restores the cold-cache behaviour.
    """
    #  Narrowed to a concrete type here so the sentinel does not leak into _cache_db's
    #  signature; the sentinel exists only to distinguish "not passed" from "no cache".
    resolved_cache_path: Path | str | None = (
        DEFAULT_CACHE_PATH
        if cache_path is _USE_DEFAULT_CACHE
        else cast("Path | str | None", cache_path)
    )
    policy = policy or BASELINE
    cases = get_case_set(case_set)
    #  `extracting` must mean "a provider that actually extracts", not "a provider object
    #  exists". StubExtractor returns an empty claim list by design, so a run with it
    #  scored every expected claim as missed and reported extraction_f1 = 0 as though it
    #  had been measured — contradicting the stub's own promise that extraction metrics
    #  come back unavailable. Passing the stub explicitly is now identical to passing
    #  nothing, which is the truth.
    extracting = provider is not None and getattr(provider, "extracts", True)
    provider = provider or StubExtractor()

    result = BenchmarkResult(
        policy_version=policy.version,
        case_set=case_set,
        extraction_available=extracting,
        git_commit=git_commit(),
        notes=description or policy.notes,
    )

    try:
        with _throwaway_db(database_url) as db, _cache_db(resolved_cache_path) as cache:
            seed_corpus(db, embed=embed_corpus)
            db.commit()

            #  The fixture corpus carries no embeddings, so a policy asking for the
            #  dense leg would get a silent lexical-only run. Detected and recorded:
            #  comparing a two-retriever score with a one-retriever score is the
            #  easiest way to reach a wrong conclusion about a policy.
            policy, result.retrievers, availability_note = _resolve_retrievers(db, policy)
            if availability_note:
                result.notes = f"{result.notes} [{availability_note}]".strip()

            for case in cases:
                result.results.append(_run_case(
                    db, case, policy=policy, provider=provider, extracting=extracting,
                    cache_db=cache,
                ))
                if cache is not None:
                    #  Per case, so an interrupted sweep keeps what it already paid for.
                    #  Defensive: two benchmarks sharing the cache file can collide on
                    #  a SQLite write lock, and losing a cache write must cost time
                    #  rather than the run.
                    try:
                        cache.commit()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not write the model-call cache: %s", exc)
                        cache.rollback()
    except Exception as exc:  # noqa: BLE001 - a crashed run is a recordable outcome
        logger.exception("Benchmark run crashed")
        outcome = RunOutcome(
            result=result, status="crash",
            description=description or policy.name, error=f"{type(exc).__name__}: {exc}",
        )
        if results_path:
            outcome.results_path = append_result(results_path, outcome)
        return outcome

    status, verdict = _verdict(result, baseline)
    outcome = RunOutcome(
        result=result, status=status,
        description=(description or policy.name) + (f" — {verdict}" if verdict else ""),
    )
    if results_path:
        outcome.results_path = append_result(results_path, outcome)
    return outcome


def _resolve_retrievers(db, policy: ResearchPolicy) -> tuple[ResearchPolicy, tuple[str, ...], str]:
    """Which retrievers are actually usable here, and a note when one is not."""
    from sqlalchemy import func

    from cnms_fom.db.models import DocumentChunk

    embedded = (
        db.query(func.count(DocumentChunk.id))
        .filter(DocumentChunk.embedding.isnot(None))
        .scalar()
        or 0
    )

    note = ""
    if policy.use_dense and not embedded:
        policy = policy.evolve(use_dense=False)
        note = (
            "dense retrieval requested but no chunk carries an embedding, so this run is "
            "lexical-only; its score is not comparable with a run that had both retrievers"
        )

    retrievers = tuple(
        name for name, on in (("dense", policy.use_dense), ("lexical", policy.use_lexical)) if on
    )
    return policy, retrievers, note


def _run_case(db, case: BenchmarkCase, *, policy, provider, extracting: bool, cache_db=None):
    """One case, scored. A crash in one case must not lose the run."""
    case_policy = policy
    if case.techniques and policy.techniques is None:
        #  The case declares its own corpus scope. Applied only when the policy has
        #  not set one, so a policy sweeping the technique filter stays in control.
        case_policy = policy.evolve(techniques=case.techniques)

    started = time.monotonic()
    try:
        brief = generate_brief(
            db, case.question, provider=provider, policy=case_policy, interpret=False,
            include_cards=False, cache_db=cache_db,
        )
    except Exception as exc:  # noqa: BLE001
        from cnms_fom.research.benchmark.evaluate import CaseResult

        logger.warning("Case %s crashed: %s", case.case_id, exc)
        return CaseResult(
            case_id=case.case_id, category=case.category,
            diagnostics=[f"Case crashed: {type(exc).__name__}: {exc}"],
        )

    evaluated = evaluate_case(
        case, brief, extraction_available=extracting, grading_available=extracting
    )
    evaluated.latency_ms = int((time.monotonic() - started) * 1000)
    return evaluated


def _verdict(result: BenchmarkResult, baseline: BenchmarkResult | None) -> tuple[str, str]:
    """Keep or discard, against a baseline if one was given."""
    if baseline is None:
        return "keep", ""

    delta = result.overall_score - baseline.overall_score
    regressions: list[str] = []

    #  A gain in the aggregate that comes with a rise in unsupported claims is not a
    #  gain. Checked separately from the score so it cannot be averaged away.
    if result.unsupported_claim_rate > baseline.unsupported_claim_rate + 1e-9:
        regressions.append(
            f"unsupported claims rose {baseline.unsupported_claim_rate:.1%} -> "
            f"{result.unsupported_claim_rate:.1%}"
        )
    for name in ("abstention_f1", "citation_accuracy"):
        mine, theirs = getattr(result, name), getattr(baseline, name)
        if mine is not None and theirs is not None and mine < theirs - 0.05:
            regressions.append(f"{name} fell {theirs:.2f} -> {mine:.2f}")

    if regressions:
        return "discard", "; ".join(regressions)
    if delta > 0.005:
        return "keep", f"score {baseline.overall_score:.3f} -> {result.overall_score:.3f}"
    if delta < -0.005:
        return "discard", f"score fell {baseline.overall_score:.3f} -> {result.overall_score:.3f}"
    return "keep", f"score unchanged at {result.overall_score:.3f}"


class _cache_db:  # noqa: N801 - a context manager, named for how it reads
    """A persistent SQLite database holding only the model-call cache.

    Separate from the throwaway corpus database on purpose. The corpus is rebuilt per
    run so a policy is always scored against the same frozen fixtures; the cache is
    the opposite — its whole value is surviving. Content addressing is what makes the
    two safe to combine: an entry is keyed on a hash of the passage text plus the
    model and prompt version, so nothing stale is ever served under a new prompt.

    A failure to open the cache is logged and downgraded to "no cache", because a
    benchmark that cannot write a cache file should still produce a score.
    """

    def __init__(self, path: Path | str | None) -> None:
        self._path = Path(path) if path is not None else None
        self._engine = None
        self._session = None

    def __enter__(self):
        if self._path is None:
            return None

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from cnms_fom.db import models  # noqa: F401 - registers the mappers
        from cnms_fom.db.models import LlmCacheEntry

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._engine = create_engine(f"sqlite+pysqlite:///{self._path}", future=True)
            #  Only the cache table: this database must never hold corpus or results.
            LlmCacheEntry.__table__.create(self._engine, checkfirst=True)
            self._session = sessionmaker(
                bind=self._engine, autoflush=False, future=True
            )()
        except Exception as exc:  # noqa: BLE001 - a cache is an optimisation
            logger.warning(
                "Could not open the benchmark model-call cache at %s (%s); "
                "running without it, which costs time and changes no score.",
                self._path, exc,
            )
            self._session = None
        return self._session

    def __exit__(self, *exc_info) -> None:
        if self._session is not None:
            try:
                self._session.commit()
            except Exception:  # noqa: BLE001
                self._session.rollback()
            self._session.close()
        if self._engine is not None:
            self._engine.dispose()


class _throwaway_db:  # noqa: N801 - a context manager, named for how it reads
    """A fresh in-file SQLite database, discarded when the run ends.

    In-file rather than in-memory so a crashed run can be inspected, and separate
    from the working database so a benchmark can never corrupt real data.
    """

    def __init__(self, database_url: str | None = None) -> None:
        self._url = database_url
        self._directory = None
        self._engine = None
        self._session = None

    def __enter__(self):
        import tempfile

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from cnms_fom.db import models  # noqa: F401 - registers the mappers
        from cnms_fom.db.base import Base

        url = self._url
        if url is None:
            self._directory = tempfile.TemporaryDirectory(prefix="cnms-benchmark-")
            url = f"sqlite+pysqlite:///{Path(self._directory.name) / 'benchmark.db'}"

        self._engine = create_engine(url, future=True)
        Base.metadata.create_all(self._engine)
        self._session = sessionmaker(bind=self._engine, autoflush=False, future=True)()
        return self._session

    def __exit__(self, *exc_info) -> None:
        if self._session is not None:
            self._session.close()
        if self._engine is not None:
            self._engine.dispose()
        if self._directory is not None:
            self._directory.cleanup()


def append_result(path: Path | str, outcome: RunOutcome) -> Path:
    """Append one row to the tab-separated results file, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = outcome.result

    def cell(value) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        #  Tabs and newlines would break the format; a description is free text.
        return str(value).replace("\t", " ").replace("\n", " ")

    row = [
        cell(result.git_commit),
        cell(result.overall_score),
        cell(result.doc_recall),
        cell(result.citation_accuracy),
        cell(result.extraction_f1),
        cell(result.abstention_f1),
        cell(result.unsupported_claim_rate),
        cell(result.latency_ms),
        cell(outcome.status),
        cell(
            f"{outcome.description} [policy={result.policy_version} cases={result.case_set} "
            f"retrievers={'+'.join(result.retrievers) or 'none'} "
            f"extraction={'yes' if result.extraction_available else 'stub'}]"
            + (f" ERROR: {outcome.error}" if outcome.error else "")
        ),
    ]

    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write("\t".join(RESULT_COLUMNS) + "\n")
        handle.write("\t".join(row) + "\n")
    return path


def read_results(path: Path | str = DEFAULT_RESULTS_PATH) -> list[dict]:
    """Every recorded run, newest last."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    return [
        dict(zip(header, line.split("\t"), strict=False))
        for line in lines[1:]
        if line.strip()
    ]


def compare_policies(
    names: list[str],
    *,
    case_set: str = "baseline",
    provider=None,
    results_path=None,
    embed_corpus: bool = False,
    cache_path: Path | str | None | object = _USE_DEFAULT_CACHE,
) -> dict:
    """Run several policies against the same cases and rank them.

    The baseline runs first and every other policy is judged against it, so a
    sweep answers "is this better than what we ship?" rather than "which of these
    is least bad?".

    Every policy shares one ``cache_path``, which is what makes a sweep affordable:
    extraction is keyed on the passage alone, so a passage any policy has already
    extracted is free for the rest, and policies differ in retrieval depth rather
    than in the questions they ask.
    """
    from cnms_fom.research.policy import get_policy

    resolved_cache_path: Path | str | None = (
        DEFAULT_CACHE_PATH
        if cache_path is _USE_DEFAULT_CACHE
        else cast("Path | str | None", cache_path)
    )
    ordered = ["baseline"] + [name for name in names if name != "baseline"]
    outcomes: dict[str, RunOutcome] = {}
    baseline_result: BenchmarkResult | None = None

    for name in ordered:
        outcome = run_benchmark(
            policy=get_policy(name), case_set=case_set, provider=provider,
            baseline=baseline_result, results_path=results_path,
            embed_corpus=embed_corpus, cache_path=resolved_cache_path,
        )
        outcomes[name] = outcome
        if name == "baseline":
            baseline_result = outcome.result

    return {
        "case_set": case_set,
        "baseline_score": baseline_result.overall_score if baseline_result else None,
        "policies": {
            name: {
                "score": outcome.result.overall_score,
                "status": outcome.status,
                "description": outcome.description,
                "doc_recall": outcome.result.doc_recall,
                "abstention_f1": outcome.result.abstention_f1,
                "unsupported_claim_rate": outcome.result.unsupported_claim_rate,
                "by_category": outcome.result.by_category(),
            }
            for name, outcome in outcomes.items()
        },
        "ranking": sorted(
            outcomes, key=lambda n: outcomes[n].result.overall_score, reverse=True
        ),
    }
