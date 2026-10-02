"""The rubrics on named real materials, from the real release (skipped when it is absent)."""

from __future__ import annotations

from collections import Counter

import pytest


def entries(records, material_id, quantity="n"):
    m = records.materials[material_id]
    return {e.scope: e for e in (m.n633 if quantity == "n" else m.k633)}


def test_counts_match_the_manifest(records):
    tables = records.manifest["counts"]["tables"]
    assert len(records.materials) == tables["materials"] == 2129
    assert len(records.sources) == tables["sources"] == 417
    assert records.n_datasets == records.manifest["counts"]["datasets"] == 2605
    assert len(records.families) == 20


def test_every_stored_n_consensus_row_is_reproduced_and_graded(records):
    graded = Counter(e.grade.grade for m in records.materials.values() for e in m.n633 if e.basis == "consensus")
    #  2,148 stored n_633nm rows; the build would have stopped on any it could not reproduce.
    assert sum(graded.values()) == 2148
    assert all(e.grade and e.grade.reason for m in records.materials.values() for e in m.n633)


def test_single_source_is_c(records):
    fused = entries(records, 38)["structure:amorphous"]
    assert (fused.grade.grade, fused.classification, fused.value) == ("C", "single_source", 1.457012124643149)


def test_model_fit_only_is_d(records):
    [cdte] = entries(records, 316).values()
    assert cdte.grade.grade == "D" and cdte.basis == "primary_dataset_point"
    assert cdte.model_fits == ["Adachi1993"]


def test_independent_agreement_is_a_and_one_group_is_b(records):
    water = entries(records, 232)
    assert water["liquid"].grade.grade == "A" and water["liquid"].groups.n_groups == 5
    #  BaGa4S7: Badikov 2010 and Kato 2021 share V. Badikov and V. Petrov.
    bgs = entries(records, 217)
    assert {e.grade.grade for e in bgs.values()} == {"B"}
    assert all(e.groups.n_groups == 1 for e in bgs.values())


def test_handbook_voter_makes_independence_unknown(records):
    #  Thallium(I) bromide: Palik's handbook (no readable authors) votes with Schroter 1931.
    [tlbr] = entries(records, 208).values()
    assert tlbr.groups.status == "unknown" and tlbr.grade.grade == "B"


def test_the_suspicious_consensus_row_is_k_of_as2se3(records):
    [k] = entries(records, 223, "k").values()
    assert k.classification == "suspicious"
    suspicious = [e for m in records.materials.values() for e in [*m.n633, *m.k633] if e.classification == "suspicious"]
    assert len(suspicious) == 1  # v0.22.0 has one, and it is a k row: no n entry is graded S


def test_only_other_temperatures_is_t(records):
    argon = entries(records, 367)
    assert argon["liquid"].grade.grade == "T" and argon["solid"].grade.grade == "T"


def test_a_contested_voter_is_excluded(records):
    e_ray = entries(records, 219)["e-ray"]
    assert e_ray.contested == [("Boyd1971-120C | e-ray", 120.0)]
    assert e_ray.grade.grade == "C"
    assert any("mixes temperatures" in n for n in records.materials[219].notices)


def test_tio2_has_only_rutile_entries(records):
    assert set(entries(records, 42)) == {"rutile | o-ray", "rutile | e-ray"}


@pytest.mark.parametrize(
    ("material_id", "kind"),
    [(38, "literature"), (77, "calculated_dft"), (133, "calculated_from_lattice"), (1840, "literature")],
)
def test_density_kinds(records, material_id, kind):
    assert records.materials[material_id].density_evidence.kind == kind


def test_k_status_from_stored_rows(records):
    assert records.materials[38].k_status == "not_measured"
    assert records.materials[232].k_status == "partially_measured"


def test_the_release_inconsistencies_are_reported(records):
    assert any("Gadolinium(III) fluoride" in n and "no density row" in n for n in records.notices)
