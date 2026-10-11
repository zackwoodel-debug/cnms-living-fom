"""The mapping from materials-db's provenance contract into CNMS (docs: materials-db docs/specs/provenance/SPEC-prov-cnms-map.md).

A literature claim never silently becomes a CNMS measurement; nothing reviewed becomes a fit bound; preview writes nothing."""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import sys
from pathlib import Path

import pytest

from cnms_fom.db.enums import ProvenanceTier
from cnms_fom.materials_release import provenance as pm

REQUIRED = ("temperature",)


def rec(provenance="measured", review="unreviewed", conflict=False, states=None, **kw):
    return pm.EvidenceInput(dataset_id=kw.pop("dataset_id", 1), provenance=provenance, review=review, review_conflict=conflict,
                            states=states if states is not None else {"temperature": "stated"}, required=REQUIRED,
                            basis=kw.pop("basis", "test"), **kw)


def test_mapping_table_row_by_row():
    assert pm.map_evidence(rec("calculated")).tier is ProvenanceTier.CALCULATED
    assert pm.map_evidence(rec("modeled")).tier is ProvenanceTier.MODELED
    for word, label in (("measured", "literature_measured"), ("fitted", "literature_fitted"), ("reported", "literature_reported")):
        r = pm.map_evidence(rec(word))
        assert r.tier is None and r.destination == "reference" and r.label == label
    unknown = pm.map_evidence(rec("unknown"))
    assert unknown.tier is ProvenanceTier.UNAVAILABLE and unknown.destination == "blocked"


def test_no_literature_record_can_come_out_measured():
    for word in pm.PROVENANCE_WORDS:
        for review in pm.REVIEW_WORDS:
            assert pm.map_evidence(rec(word, review)).tier is not ProvenanceTier.MEASURED


@pytest.mark.parametrize("review", ["rejected", "needs_clarification"])
def test_a_blocking_review_blocks_whatever_the_provenance(review):
    for word in ("calculated", "measured", "modeled"):
        r = pm.map_evidence(rec(word, review))
        assert r.destination == "blocked" and any("review" in x for x in r.reasons)


def test_reviewers_who_disagree_block_until_resolved():
    r = pm.map_evidence(rec("calculated", "conflict", conflict=True))
    assert r.destination == "blocked" and any("disagree" in x for x in r.reasons)


def test_a_missing_required_condition_blocks_and_names_it():
    for states in ({"temperature": "not_recorded"}, {"temperature": "not_reported"}, {}):
        r = pm.map_evidence(rec("calculated", states=states))
        assert r.destination == "blocked" and any("temperature" in x for x in r.reasons)


def test_approval_changes_the_label_only():
    plain, reviewed = pm.map_evidence(rec("calculated")), pm.map_evidence(rec("calculated", "approved"))
    assert (plain.tier, plain.destination) == (reviewed.tier, reviewed.destination)
    assert plain.reviewed is False and reviewed.reviewed is True
    lit = pm.map_evidence(rec("measured", "approved"))
    assert lit.tier is None and lit.reviewed is True


def test_unreviewed_is_allowed_through_but_marked():
    r = pm.map_evidence(rec("calculated"))
    assert r.destination == "reference" and r.reviewed is False


def test_unknown_words_are_refused_not_defaulted():
    for bad in (rec("verified"), rec("measured", "maybe")):
        with pytest.raises(ValueError):
            pm.map_evidence(bad)


def test_the_result_has_no_bounds_constraint_or_verified_field():
    fields = {f.name for f in dataclasses.fields(pm.MappingResult)}
    assert fields == {"tier", "destination", "label", "reviewed", "reasons", "dry_run"}
    public = {n for n in dir(pm) if not n.startswith("_")}
    assert not [n for n in public if "bound" in n.lower() or "verified" in n.lower() or "constraint" in n.lower()]


def test_the_fit_code_never_imports_the_mapper():
    pkg = Path(inspect.getfile(importlib.import_module("cnms_fom.modalfit"))).parent
    offenders = [p.name for p in pkg.glob("*.py") if "materials_release" in p.read_text() and "provenance" in p.read_text()]
    assert offenders == []


def test_preview_is_pure_and_dry():
    out = pm.preview([rec("calculated"), rec("measured"), rec("unknown")])
    assert [o.destination for o in out] == ["reference", "reference", "blocked"]
    assert all(o.dry_run is True for o in out)


def test_preview_cannot_write(monkeypatch):
    """Any attempt to open a database session inside preview fails the test."""
    import sqlalchemy.orm

    def boom(*a, **k):
        raise AssertionError("preview touched a session")

    monkeypatch.setattr(sqlalchemy.orm.Session, "__init__", boom)
    pm.preview([rec("calculated")])


def test_the_importers_current_tier_is_reported_where_it_differs():
    rows = pm.compare_with_importer([rec("measured", dataset_id=1), rec("calculated", dataset_id=2), rec("modeled", dataset_id=3),
                                     rec("reported", dataset_id=4), rec("unknown", dataset_id=5)])
    by = {r["dataset_id"]: r for r in rows}
    assert by[1]["importer_tier"] == "measured" and by[1]["mapper_tier"] is None and by[1]["differs"] is True
    assert by[2]["differs"] is False and by[3]["differs"] is False
    assert by[4]["importer_tier"] == "measured" and by[4]["differs"] is True
    assert by[5]["mapper_tier"] == "unavailable"


def test_importer_default_matches_what_the_comparison_assumes():
    from cnms_fom.ingest.materials_db import DEFAULT_TIER
    assert pm.IMPORTER_DEFAULT_TIER is DEFAULT_TIER


def test_vocabulary_copy_matches_materials_db_when_it_is_present():
    src = Path.home() / "code" / "materials-db" / "src"
    if not (src / "materials_db" / "provenance" / "vocab.py").exists():
        pytest.skip("materials-db checkout not found next to this repository")
    sys.path.insert(0, str(src))
    try:
        vocab = importlib.import_module("materials_db.provenance.vocab")
    finally:
        sys.path.remove(str(src))
    assert tuple(pm.PROVENANCE_WORDS) == vocab.PROVENANCE
    assert tuple(pm.REVIEW_WORDS) == vocab.REVIEW
    assert tuple(pm.FIELD_STATES) == vocab.FIELD_STATE
