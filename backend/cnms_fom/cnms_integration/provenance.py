"""Run provenance stamping (FOM_PROOF Sec. 16, reproducibility checklist).

An analysis is reproducible only if you can say exactly what produced it. This
module builds the stamp that goes onto every ``AnalysisRun``: code version, git
commit, random seed, permutation count, the frozen eligible set, and the
fingerprint of the pre-registered hypothesis table.

That last one is what makes pre-registration real. If someone edits an expected
sign after seeing the results, the fingerprint on the new run differs from the
one on the old, and the two runs no longer claim to have tested the same thing.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

from cnms_fom import __version__
from cnms_fom.config import get_settings
from cnms_fom.fom_engine.hypotheses import registry_fingerprint


def git_sha(short: bool = True) -> str | None:
    """Current commit, or ``None`` outside a git checkout."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        if short:
            args.append("HEAD")
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def working_tree_dirty() -> bool | None:
    """Whether uncommitted changes exist — a dirty tree makes ``git_sha`` a lie."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


@dataclass
class RunProvenance:
    """Everything needed to re-run an analysis and get the same numbers."""

    kind: str
    code_version: str = __version__
    git_sha: str | None = field(default_factory=git_sha)
    dirty_working_tree: bool | None = field(default_factory=working_tree_dirty)
    hypothesis_fingerprint: str = field(default_factory=registry_fingerprint)
    random_seed: int | None = None
    permutation_b: int | None = None
    eligible_material_ids: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        settings = get_settings()
        if self.random_seed is None:
            self.random_seed = settings.random_seed
        if self.permutation_b is None:
            self.permutation_b = settings.permutation_b

    def warnings(self) -> list[str]:
        """Reproducibility problems worth surfacing before a result is released."""
        issues: list[str] = []
        if self.git_sha is None:
            issues.append(
                "Not a git checkout: the code version cannot be pinned. Sec. 16 item 11 "
                "requires transformations and weights to be versioned."
            )
        if self.dirty_working_tree:
            issues.append(
                f"Working tree is dirty at {self.git_sha}: the recorded commit does not describe "
                "the code that ran. Commit before producing a reportable result."
            )
        if self.permutation_b and self.permutation_b < 10_000:
            issues.append(
                f"B = {self.permutation_b} is below the 10,000 of Eq. (42). Acceptable only for "
                "exploratory work, and the constraint must be documented in the report."
            )
        if not self.eligible_material_ids:
            issues.append(
                "No eligible material set recorded. Sec. 16 item 10: freeze the eligible set "
                "before the final calculation."
            )
        return issues

    def as_run_kwargs(self) -> dict:
        """Column values for ``db.models.AnalysisRun``."""
        return {
            "kind": self.kind,
            "params": {
                **self.params,
                "hypothesis_fingerprint": self.hypothesis_fingerprint,
                "dirty_working_tree": self.dirty_working_tree,
                "provenance_warnings": self.warnings(),
            },
            "eligible_material_ids": self.eligible_material_ids,
            "random_seed": self.random_seed,
            "permutation_b": self.permutation_b,
            "code_version": self.code_version,
            "git_sha": self.git_sha,
            "started_at": self.created_at,
        }
