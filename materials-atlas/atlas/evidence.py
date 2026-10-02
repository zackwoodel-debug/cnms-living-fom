"""Evidence rubrics: published rules, pure functions, every grade with its reason.

Each function here maps release facts to a verdict and a sentence saying why.
None of them reads a file or a database, so each branch is tested on built
cases and on named real materials (``tests/test_evidence.py``), and the
vault's ``Methods/`` pages are written from the same constants.

A rubric never upgrades on missing information. Where a rule cannot decide,
the answer is ``unknown`` (or the lower grade), never the favourable option.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from cnms_fom.materials_release import DENSITY_KINDS, density_kind

from .authors import GroupResult

#  The release's own constants (materials-db scripts/release_validation.py).
#  Restated, not imported, so that a change there is a visible change here.
AMBIENT_C: tuple[float, float] = (15.0, 30.0)
REF_WL_NM = 633.0
CONSENSUS_CLASSES = ("single_source", "excellent", "warning", "suspicious")
VALIDATION_CLASSES = ("excellent", "warning", "suspicious")

# ---------------------------------------------------------------------------
# 6.1 Grade of n at 633 nm
# ---------------------------------------------------------------------------

GRADES: tuple[str, ...] = ("A", "B", "C", "D", "T", "S", "none")

#  The rule behind each grade, as published in Methods/Evidence grades.
GRADE_RULES: dict[str, str] = {
    "A": "consensus of two or more measured ambient datasets, classed excellent (spread < 2% of the "
    "median), from two or more independent author groups",
    "B": "consensus of two or more measured ambient datasets classed excellent whose independence is "
    "one group or unknown; or a consensus classed warning (spread 2-10%)",
    "C": "a single measured ambient dataset covers 633 nm (consensus single_source)",
    "D": "no measured dataset covers 633 nm at ambient temperature; at least one model fit of the "
    "dielectric function does",
    "T": "no dataset covers 633 nm at ambient temperature; datasets cover it only at other stated "
    "temperatures",
    "S": "consensus classed suspicious: the measured ambient datasets disagree (spread >= 10%)",
    #  (Applied before the rows above.) A voter whose own label names a temperature
    #  outside 15-30 degC while its rows store none is excluded, and the entry is
    #  graded on the voters that remain: at most B with two or more, C with one.
    "none": "no dataset covers 633 nm",
}


@dataclass(frozen=True)
class Grade:
    grade: str
    reason: str


def is_ambient(temperature_c: float | None) -> bool:
    """The release's ambient window: 15-30 degC, or no temperature stated."""
    return temperature_c is None or AMBIENT_C[0] <= temperature_c <= AMBIENT_C[1]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def grade_n633(
    *,
    consensus_class: str | None,
    n_voting: int = 0,
    n_voting_sources: int = 0,
    groups: GroupResult | None = None,
    ambient_model_fits: Sequence[str] = (),
    other_temperatures: Sequence[tuple[str, float | None]] = (),
    contested: Sequence[tuple[str, float]] = (),
) -> Grade:
    """Grade one n-at-633-nm entry (one material, phase or variant, and axis).

    ``consensus_class`` and ``n_voting`` are the stored ``consensus_properties``
    row (None when there is none). ``groups`` is the author-group result over
    the voting datasets' sources. ``ambient_model_fits`` lists model-fit
    datasets covering 633 nm at ambient temperature; ``other_temperatures``
    lists (label, degC) of datasets covering it only at another temperature.

    ``contested`` lists voters (label, degC) whose own label names a temperature
    outside the ambient window while their rows store none. The release counted
    them as ambient because the stored temperature is missing; that is a grade
    resting on missing information, so they are excluded here and the entry is
    graded on the voters that remain.
    """
    if consensus_class is not None and consensus_class not in CONSENSUS_CLASSES:
        raise ValueError(f"unknown consensus class {consensus_class!r}")
    if contested:
        if consensus_class is None:
            raise ValueError("contested voters need a consensus")
        remaining = n_voting - len(contested)
        why = "; ".join(f"{label} names {fmt_number(t)} degC but stores no temperature" for label, t in contested)
        if remaining >= 2:
            return Grade("B", f"{_plural(remaining, 'measured ambient dataset')} remain after excluding "
                              f"{why}; their agreement cannot be re-classed from stored values")
        if remaining == 1:
            return Grade("C", f"one measured ambient dataset remains after excluding {why}")
        return Grade("T", f"no voter remains after excluding {why}")

    who = (
        f"{_plural(n_voting, 'measured ambient dataset')} from {_plural(n_voting_sources, 'source')}"
        if n_voting_sources
        else _plural(n_voting, "measured ambient dataset")
    )

    if consensus_class == "suspicious":
        return Grade("S", f"{who} give n at 633 nm and disagree: the release classes their spread "
                          "suspicious (>= 10% of the median)")
    if consensus_class == "excellent":
        if n_voting < 2:
            raise ValueError("an excellent consensus needs two or more voting datasets")
        if groups is not None and groups.status == "determined" and (groups.n_groups or 0) >= 2:
            return Grade("A", f"{who} agree (spread classed excellent, < 2% of the median) and come "
                              f"from {groups.n_groups} independent author groups")
        if groups is not None and groups.status == "determined":
            why = "all of them from one author group"
        else:
            why = "independence unknown" + (f": {groups.reason}" if groups is not None else "")
        return Grade("B", f"{who} agree (spread classed excellent, < 2% of the median), {why}")
    if consensus_class == "warning":
        return Grade("B", f"{who} agree only within the warning class (spread 2-10% of the median)")
    if consensus_class == "single_source":
        if n_voting != 1:
            raise ValueError("a single_source consensus has exactly one voting dataset")
        return Grade("C", "one measured ambient dataset covers 633 nm; there is nothing to check it against")

    if ambient_model_fits:
        reason = (
            "no measured dataset covers 633 nm at ambient temperature; only model fits do ("
            + ", ".join(ambient_model_fits) + ")"
        )
        if other_temperatures:
            reason += "; " + _other_temperature_phrase(other_temperatures) + " also"
        return Grade("D", reason)
    if other_temperatures:
        return Grade("T", "no dataset covers 633 nm at ambient temperature (15-30 degC or unstated); "
                          + _other_temperature_phrase(other_temperatures))
    return Grade("none", "no dataset covers 633 nm")


def _other_temperature_phrase(entries: Sequence[tuple[str, float | None]]) -> str:
    """'2 datasets at -7 degC cover it' / '1 dataset at 599.85 degC covers it'."""
    temps = sorted({t for _, t in entries if t is not None})
    shown = ", ".join(f"{fmt_number(t)} degC" for t in temps)
    verb = "covers it" if len(entries) == 1 else "cover it"
    return f"{_plural(len(entries), 'dataset')} at {shown} {verb}"


_TAG_TEMPERATURE = re.compile(r"-(\d+(?:\.\d+)?)(C|K)$")
#  273 K is stored as -0.15 degC where a source wrote "0 degC"; a smaller gap than
#  this is that rounding, not a contradiction.
TAG_TEMPERATURE_TOLERANCE_C = 0.2


def tag_temperature_c(tag: str) -> float | None:
    """The temperature a source tag's series suffix names ('Wu1993-25.1C', 'Kerl1995-293K'), in degC."""
    match = _TAG_TEMPERATURE.search(tag)
    if not match:
        return None
    value = float(match.group(1))
    return value if match.group(2) == "C" else value - 273.15


# ---------------------------------------------------------------------------
# 6.2 Density kind
# ---------------------------------------------------------------------------

DENSITY_KIND_NAMES: dict[str, str] = {**DENSITY_KINDS, "absent": "no density in this release"}


@dataclass(frozen=True)
class DensityEvidence:
    kind: str
    reason: str


def density_evidence(origin: str | None, *, present: bool, absent_reason: str | None = None) -> DensityEvidence:
    """Kind of a material's density, from its stored label only (or why there is none)."""
    if not present:
        return DensityEvidence(
            "absent",
            absent_reason or "the release records no density and gives no reason for this material",
        )
    kind = density_kind(origin)
    return DensityEvidence(kind.kind, kind.reason)


# ---------------------------------------------------------------------------
# 6.3 Measured vs model fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetKind:
    kind: str  # "measured" | "model_fit"
    reason: str


def dataset_kind(model_fit_reason: str | None) -> DatasetKind:
    """materials-db's dataset_kind.py verdict, with its quoted evidence."""
    if model_fit_reason:
        return DatasetKind("model_fit", model_fit_reason)
    return DatasetKind(
        "measured",
        "measured: the page carries no model-fit marker (no calculation script, no 'fit of ... model' "
        "comment, no model as the paper's subject)",
    )


# ---------------------------------------------------------------------------
# 6.4 Independence: see authors.author_groups. Summarised here for a note.
# ---------------------------------------------------------------------------


def independence_value(groups: GroupResult | None) -> int | str:
    """2, 1, ... or 'unknown' — the frontmatter value of independent_groups."""
    if groups is None or groups.status != "determined" or groups.n_groups is None:
        return "unknown"
    return groups.n_groups


# ---------------------------------------------------------------------------
# 6.5 k status
# ---------------------------------------------------------------------------

K_STATUSES = ("measured", "partially_measured", "not_measured")


def k_status(n_rows: int, n_with_k: int) -> str:
    """Per dataset, counted from stored rows: k on every row, on some, or on none.

    A NULL k is "the source gives none". It is never 0 (materials-db's ModalFit
    exporter writes 0.0 for it as a transparent approximation, counted and
    printed; the atlas never does).
    """
    if n_rows <= 0 or not 0 <= n_with_k <= n_rows:
        raise ValueError(f"bad k counts: {n_with_k} of {n_rows}")
    if n_with_k == n_rows:
        return "measured"
    return "not_measured" if n_with_k == 0 else "partially_measured"


def material_k_status(statuses: Sequence[str]) -> str:
    """Across a material's datasets: measured only if every dataset is."""
    if not statuses:
        raise ValueError("a material with no datasets has no k status")
    if all(s == "measured" for s in statuses):
        return "measured"
    if all(s == "not_measured" for s in statuses):
        return "not_measured"
    return "partially_measured"


# ---------------------------------------------------------------------------
# dataset_validation pairs -> typed links
# ---------------------------------------------------------------------------

VALIDATION_RELATION = {
    "excellent": "agrees_with",
    "warning": "partly_agrees_with",
    "suspicious": "disagrees_with",
}


def validation_relation(classification: str) -> str:
    try:
        return VALIDATION_RELATION[classification]
    except KeyError as exc:
        raise ValueError(f"unknown validation class {classification!r}") from exc


# ---------------------------------------------------------------------------
# Exact number formatting (shared with the templates and the renderer)
# ---------------------------------------------------------------------------


def fmt_number(value: float | int) -> str:
    """The stored value, exactly: shortest round-trip form, no trailing '.0'.

    Never rounds. 1.457012124643149 stays 1.457012124643149; 210.0 is 210.
    """
    if isinstance(value, bool):
        raise TypeError("a boolean is not a number here")
    if isinstance(value, int):
        return str(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{value!r} is not a finite stored value")
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text
