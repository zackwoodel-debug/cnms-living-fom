"""The autoresearch benchmark: fixed cases, honest metrics, reproducible runs.

Offline by default. The corpus is defined in :mod:`cases` and seeded into a
throwaway database, so a score compares policies rather than corpora and two runs a
month apart are comparable. Nothing a run does modifies the truth data.

The metric that matters most is not the headline score: it is
``unsupported_claim_rate``, the fraction of extracted claims whose quote is not
actually on the page they cite. It is subtracted from the score rather than
weighted into it, so a policy cannot buy a better number with confident fabrication.
"""

from .cases import CASE_SETS, CASES, CORPUS, BenchmarkCase, get_case_set, seed_corpus  # noqa: F401
from .evaluate import BenchmarkResult, CaseResult, evaluate_case  # noqa: F401
from .runner import (  # noqa: F401
    DEFAULT_RESULTS_PATH,
    RESULT_COLUMNS,
    RunOutcome,
    StubExtractor,
    append_result,
    compare_policies,
    read_results,
    run_benchmark,
)
