"""The only prose the atlas writes: fixed sentences filled with release facts.

Section 5 of materials-atlas/CLAUDE.md. Every function returns one sentence (or
a list of one-sentence limits), every branch is unit-tested, and every branch
that cannot be filled from a fact says so instead of filling it. Numbers are
printed exactly as stored (``evidence.fmt_number``), never rounded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .authors import GroupResult
from .evidence import DENSITY_KIND_NAMES, fmt_number

#  Adjective for each density kind, as it reads before "density".
_DENSITY_ADJECTIVE = {
    "literature": "literature",
    "calculated_dft": "DFT-calculated",
    "bulk_approximation": "bulk-approximation",
    "calculated_from_lattice": "lattice-derived",
}


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


# ---------------------------------------------------------------------------
# 5.1 Summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DensityFact:
    value: float
    label: str
    kind: str


def summary_sentence(
    *,
    name: str,
    formula: str | None,
    material_id: int,
    n_datasets: int,
    wavelength_min_nm: float | None,
    wavelength_max_nm: float | None,
    n_model_fits: int,
    density: DensityFact | None,
    density_absent_reason: str | None = None,
) -> str:
    """'{name} ({formula}, id {id}) has {n} optical dataset(s) covering {min}-{max} nm ... and {density}.'"""
    identity = f"{formula}, id {material_id}" if formula else f"no formula in this release, id {material_id}"
    if n_datasets == 0:
        optics = "has no optical dataset"
    else:
        if wavelength_min_nm is None or wavelength_max_nm is None:
            raise ValueError("a material with datasets has a wavelength range")
        optics = (
            f"has {_count(n_datasets, 'optical dataset')} covering "
            f"{fmt_number(wavelength_min_nm)}-{fmt_number(wavelength_max_nm)} nm"
        )
        if n_model_fits and n_datasets == 1:
            optics += ", which is a model fit"
        elif n_model_fits == 1:
            optics += ", 1 of them a model fit"
        elif n_model_fits:
            optics += f", {n_model_fits} of them model fits"
    if density is not None:
        adjective = _DENSITY_ADJECTIVE.get(density.kind)
        if adjective is None:
            raise ValueError(f"no wording for density kind {density.kind!r}")
        dens = f"a {adjective} density of {fmt_number(density.value)} g/cm3 ({density.label})"
    else:
        dens = f"no density in this release ({density_absent_reason or 'no reason recorded'})"
    return f"{name} ({identity}) {optics} and {dens}."


# ---------------------------------------------------------------------------
# 5.2 Agreement (one n-at-633-nm entry)
# ---------------------------------------------------------------------------


def agreement_sentence(
    *,
    grade: str,
    consensus_class: str | None,
    n_voting: int = 0,
    n_sources: int = 0,
    n_model_fits: int = 0,
    n_other_temperature: int = 0,
) -> str:
    """How far the sources that give n at 633 nm agree, in the release's own classes."""
    who = f"{_count(n_voting, 'measured dataset')} from {_count(n_sources, 'source')}"
    if consensus_class == "single_source":
        return "Only one measured dataset gives n at 633 nm, so its agreement cannot be checked."
    if consensus_class == "excellent":
        return f"{who} give n at 633 nm and agree within 2% of their median (excellent)."
    if consensus_class == "warning":
        return f"{who} give n at 633 nm and agree only within 10% of their median (warning)."
    if consensus_class == "suspicious":
        return f"{who} give n at 633 nm and disagree by 10% of their median or more (suspicious)."
    if consensus_class is not None:
        raise ValueError(f"unknown consensus class {consensus_class!r}")
    if grade == "D":
        return (
            f"Only model fits cover 633 nm ({_count(n_model_fits, 'dataset')}); model fits never vote, "
            "so there is no consensus."
        )
    if grade == "T":
        return (
            "No dataset covers 633 nm at ambient temperature; "
            f"{_count(n_other_temperature, 'dataset covers', 'datasets cover')} it only at other stated temperatures."
        )
    if grade == "none":
        return "No dataset covers 633 nm."
    raise ValueError(f"no agreement wording for grade {grade!r} without a consensus class")


# ---------------------------------------------------------------------------
# 5.3 Independence
# ---------------------------------------------------------------------------


def independence_sentence(groups: GroupResult | None) -> str | None:
    """None when there is only one voting dataset (nothing to be independent of)."""
    if groups is None:
        return None
    if groups.status != "determined" or groups.n_groups is None:
        return f"Independence cannot be determined: {groups.reason}."
    if groups.n_groups >= 2:
        return f"They come from {groups.n_groups} distinct author groups."
    return (
        "All agreeing sources are connected by shared authors, so the agreement is within one "
        "research group."
    )


# ---------------------------------------------------------------------------
# 5.4 Limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LimitFacts:
    """What a material's record lacks or qualifies, as release facts."""

    k_status: str
    density_kind: str
    density_absent_reason: str | None = None
    has_sld: bool = False
    has_dielectric: bool = False
    n_model_fits: int = 0
    unstated_temperature_labels: Sequence[str] = ()
    phase_unstated: bool = False
    dielectric_withheld_reason: str | None = None


def limit_sentences(facts: LimitFacts) -> list[str]:
    """One sentence per limit, in a fixed order. Always ends with the reference-data limit."""
    out: list[str] = []
    if facts.k_status == "not_measured":
        out.append("k is not measured in any dataset; this does not mean k = 0.")
    elif facts.k_status == "partially_measured":
        out.append(
            "k is given in only some datasets or over part of their range; where it is missing it is "
            "unknown, not 0."
        )
    elif facts.k_status != "measured":
        raise ValueError(f"unknown k status {facts.k_status!r}")

    if facts.density_kind == "absent":
        out.append(
            "This release has no density for this material"
            + (f" ({facts.density_absent_reason})" if facts.density_absent_reason else "")
            + ", so it has no SLD either."
        )
    elif facts.density_kind == "calculated_dft":
        out.append("The density is a DFT calculation (Materials Project), not a measurement.")
    elif facts.density_kind == "bulk_approximation":
        out.append("The density is a bulk value standing in for a film, not a measurement of a film.")
    elif facts.density_kind == "calculated_from_lattice":
        out.append("The density is computed from experimental lattice parameters, not measured directly.")
    elif facts.density_kind not in DENSITY_KIND_NAMES:
        raise ValueError(f"unknown density kind {facts.density_kind!r}")

    if facts.has_sld:
        out.append("The X-ray and neutron SLDs are calculated with the periodictable package, not measured.")
    if facts.has_dielectric:
        out.append(
            "The dielectric constants are 0 K DFPT calculations (Materials Project), not measurements; "
            "the release reports that they mostly run high against measured values."
        )
    elif facts.dielectric_withheld_reason:
        out.append(f"The release stores no calculated dielectric constant here: {facts.dielectric_withheld_reason}.")
    if facts.n_model_fits:
        out.append(
            f"{_count(facts.n_model_fits, 'dataset is a model fit', 'datasets are model fits')} of the "
            "dielectric function, not measured data."
        )
    for label in facts.unstated_temperature_labels:
        out.append(
            f"The source of {label} states no temperature; the release's consensus rule counts it as ambient."
        )
    if facts.phase_unstated:
        out.append("The crystal phase of the optical samples is not stated in this release.")
    out.append(
        "These are reference values from materials-db, not CNMS measurements; they are not used in "
        "FOM scores or by the optimizer."
    )
    return out
