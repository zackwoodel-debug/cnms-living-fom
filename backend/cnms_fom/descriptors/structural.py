"""Structural descriptor computation from a pymatgen ``Structure``.

Implements the descriptors of FOM_PROOF Table 2 that are derivable from the
atomic structure alone.  The four that are *not* — Z*_RMS, omega_TO,min, S_osc,
A_eps — come from DFPT or spectroscopy and are ingested as values with their own
provenance; see :func:`missing_from_structure`.

pymatgen and matminer are imported lazily so the API starts without the heavy
``descriptors`` extra installed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from cnms_fom.db.enums import ProvenanceTier

from .registry import STRUCTURAL_DESCRIPTORS

#  Computed from DFPT/spectroscopy, not from geometry.
NON_GEOMETRIC = ("Z_RMS_star", "omega_TO_min", "S_osc", "A_eps")


@dataclass
class DescriptorResult:
    """A computed descriptor plus everything needed to store it defensibly."""

    key: str
    value: float | None
    units: str
    method: str
    provenance_tier: ProvenanceTier
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "descriptor_key": self.key,
            "value": self.value,
            "units": self.units,
            "method": self.method,
            "provenance_tier": self.provenance_tier.value,
            "note": self.note,
        }


def _require_pymatgen():
    try:
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen.core import Element, Structure
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Structural descriptors need the 'descriptors' extra: "
            "pip install -e '.[descriptors]'"
        ) from exc
    return Structure, Element, CrystalNN


def structure_from_cif(cif: str):
    """Parse a CIF string into a pymatgen ``Structure``."""
    Structure, _, _ = _require_pymatgen()
    return Structure.from_str(cif, fmt="cif")


def _anion_species(structure, Element) -> set[str]:
    """Identify anions by electronegativity.

    A simple, stated rule: the most electronegative element present is the anion
    family, plus anything within 0.2 Pauling units of it.  For the oxide corpus
    this resolves to oxygen.  It is declared here rather than hidden in a
    heuristic so that a nitride or chalcogenide corpus can override it.
    """
    symbols = {str(sp.symbol) for sp in structure.composition.elements}
    chis = {s: Element(s).X for s in symbols if Element(s).X is not None}
    if not chis:
        return set()
    chi_max = max(chis.values())
    return {s for s, chi in chis.items() if chi >= chi_max - 0.2}


def compute_geometric_descriptors(
    structure: Any, *, cn_cutoff_tolerance: float = 0.0
) -> list[DescriptorResult]:
    """Compute the geometry-derived entries of Table 2.

    Returns V_fu, rho, CN, d_M-O, sigma_d, delta_chi.  Values that cannot be
    computed come back as ``None`` — FOM_PROOF Eq. (4): missing stays missing.
    """
    _, Element, CrystalNN = _require_pymatgen()
    results: list[DescriptorResult] = []

    comp = structure.composition
    _, factor = comp.get_reduced_composition_and_factor()
    z = max(1, int(round(factor)))

    results.append(
        DescriptorResult(
            key="V_fu",
            value=float(structure.volume) / z,
            units=STRUCTURAL_DESCRIPTORS["V_fu"].units,
            method=f"pymatgen Structure.volume / Z (Z={z})",
            provenance_tier=ProvenanceTier.CALCULATED,
        )
    )
    results.append(
        DescriptorResult(
            key="rho",
            value=float(structure.density),
            units=STRUCTURAL_DESCRIPTORS["rho"].units,
            method="pymatgen Structure.density",
            provenance_tier=ProvenanceTier.CALCULATED,
        )
    )

    anions = _anion_species(structure, Element)
    cns: list[float] = []
    bond_lengths: list[float] = []
    cnn_note: str | None = None
    try:
        cnn = CrystalNN()
        for index, site in enumerate(structure):
            symbol = str(site.specie.symbol) if hasattr(site, "specie") else None
            if symbol is None or symbol in anions:
                continue  # cation sites only
            neighbours = cnn.get_nn_info(structure, index)
            anion_neighbours = [
                nn for nn in neighbours if str(nn["site"].specie.symbol) in anions
            ]
            if not anion_neighbours:
                continue
            cns.append(float(len(anion_neighbours)))
            bond_lengths.extend(
                float(site.distance(nn["site"])) for nn in anion_neighbours
            )
    except Exception as exc:  # noqa: BLE001 - CrystalNN fails on pathological cells
        cnn_note = f"CrystalNN failed: {exc}"

    results.append(
        DescriptorResult(
            key="CN",
            value=float(statistics.fmean(cns)) if cns else None,
            units=STRUCTURAL_DESCRIPTORS["CN"].units,
            method="pymatgen CrystalNN, cation sites, anion neighbours only",
            provenance_tier=ProvenanceTier.CALCULATED,
            note=cnn_note,
        )
    )
    results.append(
        DescriptorResult(
            key="d_M_O",
            value=float(statistics.fmean(bond_lengths)) if bond_lengths else None,
            units=STRUCTURAL_DESCRIPTORS["d_M_O"].units,
            method="mean cation-anion CrystalNN bond length",
            provenance_tier=ProvenanceTier.CALCULATED,
            note=cnn_note,
        )
    )
    results.append(
        DescriptorResult(
            key="sigma_d",
            value=(
                float(statistics.stdev(bond_lengths)) if len(bond_lengths) > 1 else None
            ),
            units=STRUCTURAL_DESCRIPTORS["sigma_d"].units,
            method="sample stdev of cation-anion bond lengths",
            provenance_tier=ProvenanceTier.CALCULATED,
            note=cnn_note,
        )
    )

    # delta_chi: composition-weighted anion electronegativity minus cation.
    cation_chis: list[tuple[float, float]] = []
    anion_chis: list[tuple[float, float]] = []
    for element, amount in comp.get_el_amt_dict().items():
        chi = Element(element).X
        if chi is None:
            continue
        (anion_chis if element in anions else cation_chis).append((chi, float(amount)))

    def _weighted(pairs: list[tuple[float, float]]) -> float | None:
        total = sum(w for _, w in pairs)
        return sum(chi * w for chi, w in pairs) / total if total else None

    chi_cat, chi_an = _weighted(cation_chis), _weighted(anion_chis)
    results.append(
        DescriptorResult(
            key="delta_chi",
            value=(chi_an - chi_cat) if (chi_an is not None and chi_cat is not None) else None,
            units=STRUCTURAL_DESCRIPTORS["delta_chi"].units,
            method="composition-weighted Pauling electronegativity difference (pymatgen Element.X)",
            provenance_tier=ProvenanceTier.CALCULATED,
        )
    )
    return results


def missing_from_structure() -> tuple[str, ...]:
    """Descriptors that geometry cannot supply.

    These must be ingested from DFPT or spectroscopy with their own method and
    XC-functional metadata (Sec. 16 items 8-9).  Returning them explicitly means
    a caller can check completeness rather than silently scoring a partial row.
    """
    return NON_GEOMETRIC


def matminer_features(structure: Any, featurizer_names: list[str] | None = None) -> dict:
    """Optional matminer featurisation for exploratory work.

    Kept deliberately separate from :func:`compute_geometric_descriptors`.
    Matminer produces hundreds of features; FOM_PROOF Sec. 9.2 warns that a
    wide, collinear feature set invites exactly the confounding the protocol is
    built to avoid.  Use these for screening and hypothesis generation, never as
    a substitute for the declared Table 2 descriptors in an official analysis.

    TODO(FOM_PROOF): if a matminer feature is promoted to an official
    descriptor, give it an entry in ``registry.STRUCTURAL_DESCRIPTORS`` with a
    physical interpretation and a pre-registered sign in ``fom_engine.hypotheses``.
    """
    try:
        from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "matminer features need the 'descriptors' extra: pip install -e '.[descriptors]'"
        ) from exc

    available = {
        "density": DensityFeatures(),
        "symmetry": GlobalSymmetryFeatures(),
    }
    selected = featurizer_names or list(available)
    out: dict[str, float] = {}
    for name in selected:
        featurizer = available.get(name)
        if featurizer is None:
            continue
        labels = featurizer.feature_labels()
        values = featurizer.featurize(structure)
        out.update({f"{name}.{lab}": val for lab, val in zip(labels, values, strict=True)})
    return out
