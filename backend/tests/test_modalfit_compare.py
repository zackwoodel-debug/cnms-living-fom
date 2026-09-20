"""Cross-technique comparison, and the refusal to average.

The scientific point of the ModalFit integration is that one quantity gets
determined twice by unrelated physics. These tests pin down the three outcomes
that matters: agreement, disagreement, and *silence* — a technique that cannot
determine a parameter is not in disagreement with one that can, and collapsing
those two is how a non-result becomes a finding.
"""

from __future__ import annotations

import copy

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.models import FitRecord  # noqa: F401 - registers mappers
from cnms_fom.modalfit.compare import (
    compare_parameter,
    cross_technique_report,
    describe_fit,
)
from cnms_fom.modalfit.records import import_fit

SAMPLE = "HFO2-PILOT-07"


def _export(*, techniques, thickness, chi2=1.5, stack_id="v1", algorithm="L-BFGS-B",
            vary=True, uncertainty=None, bounds=(50.0, 200.0), blocks=None):
    thickness_param = {"value": thickness, "min": bounds[0], "max": bounds[1], "vary": vary}
    if uncertainty is not None:
        thickness_param["uncertainty"] = uncertainty

    film = {
        "role": "layer",
        "label": "hfo2_film",
        "material": "HfO2",
        "structural": {
            "thickness": thickness_param,
            "roughness": {"value": 4.0, "min": 0.0, "max": 20.0, "vary": True},
        },
    }
    film.update(copy.deepcopy(blocks or {"xray": {"sld_real": {"value": 40.0, "vary": True}}}))

    return {
        "stack_id": stack_id,
        "sample_id": SAMPLE,
        "fit": {"techniques": techniques, "algorithm": algorithm, "chi2": chi2},
        "stack": [
            {"role": "ambient", "label": "air"},
            film,
            {"role": "substrate", "label": "silicon", "material": "Si"},
        ],
    }


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'compare.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


def test_no_fits_reports_no_fits(db):
    result = compare_parameter(db, "NOT-A-SAMPLE")
    assert result["verdict"] == "no_fits"
    assert result["determinations"] == []


def test_one_determination_says_there_is_nothing_to_check_it_against(db):
    import_fit(db, _export(techniques=["XRR"], thickness=103.4))
    db.commit()

    result = compare_parameter(db, SAMPLE, parameter="thickness")
    assert result["verdict"] == "single_determination"
    assert "Nothing to cross-check" in result["summary"]


def test_agreeing_techniques_are_consistent(db):
    import_fit(db, _export(techniques=["XRR"], thickness=103.4, stack_id="v1"))
    import_fit(
        db,
        _export(
            techniques=["SE"],
            thickness=105.0,
            stack_id="v2",
            blocks={"optical": {"n": {"value": 2.05, "vary": True}}},
        ),
    )
    db.commit()

    result = compare_parameter(db, SAMPLE, parameter="thickness")
    assert result["verdict"] == "consistent"
    assert {d["label"] for d in result["determinations"]} == {"XRR", "SE"}


def test_disagreeing_techniques_are_reported_not_averaged(db):
    import_fit(db, _export(techniques=["XRR"], thickness=103.4, stack_id="v1"))
    import_fit(
        db,
        _export(
            techniques=["SE"],
            thickness=152.0,
            stack_id="v2",
            blocks={"optical": {"n": {"value": 2.05, "vary": True}}},
        ),
    )
    db.commit()

    result = compare_parameter(db, SAMPLE, parameter="thickness")
    assert result["verdict"] == "disagreement"
    assert "Do not average them" in result["summary"]
    #  Both determinations survive; no merged value anywhere in the payload.
    assert sorted(d["value"] for d in result["determinations"]) == [103.4, 152.0]
    assert "mean" not in result and "average" not in result
    assert result["spread"] == pytest.approx(48.6)


def test_uncertainties_are_used_when_two_fits_report_them(db):
    """With posteriors, the test is spread against combined sigma, not a flat 10%."""
    import_fit(
        db,
        _export(
            techniques=["XRR"], thickness=100.0, stack_id="v1",
            algorithm="DREAM (emcee)", uncertainty=0.5,
        ),
    )
    import_fit(
        db,
        _export(
            techniques=["NR"], thickness=108.0, stack_id="v2",
            algorithm="DREAM (emcee)", uncertainty=0.5,
            blocks={"neutron": {"sld_real": {"value": 5.0, "vary": True}}},
        ),
    )
    db.commit()

    result = compare_parameter(db, SAMPLE, parameter="thickness")
    #  8 Å apart on ~0.7 Å combined sigma: a disagreement even though the
    #  relative spread (7.7%) is under the flat review threshold.
    assert result["relative_spread"] < 0.10
    assert result["verdict"] == "disagreement"
    assert "combined 1σ" in result["summary"]


def test_a_technique_that_cannot_determine_a_parameter_is_silent_not_disagreeing(db):
    """QCM does not measure roughness; its carried-through value is not a rival claim."""
    import_fit(
        db,
        _export(
            techniques=["QCM"], thickness=90.0, stack_id="qcm",
            blocks={"viscoelastic": {"density": {"value": 9.0, "vary": True}}},
        ),
    )
    db.commit()

    result = compare_parameter(db, SAMPLE, parameter="roughness")
    assert result["determinations"] == []
    assert result["silent_fits"] and result["silent_fits"][0]["techniques"] == ["QCM"]
    assert result["verdict"] == "no_determination"


def test_a_fixed_parameter_is_not_a_determination(db):
    import_fit(db, _export(techniques=["XRR"], thickness=103.4, stack_id="v1"))
    import_fit(
        db,
        _export(
            techniques=["SE"], thickness=152.0, stack_id="v2", vary=False,
            blocks={"optical": {"n": {"value": 2.05, "vary": True}}},
        ),
    )
    db.commit()

    result = compare_parameter(db, SAMPLE, parameter="thickness")
    #  Both values are visible, but only one was refined — so there is nothing to
    #  cross-check, and the SE value carries its caveat.
    assert result["verdict"] == "single_determination"
    se = next(d for d in result["determinations"] if d["label"] == "SE")
    assert se["was_free"] is False
    assert any("held fixed" in c for c in se["caveats"])


def test_caveats_carry_the_dq_zero_limitation(db):
    import_fit(db, _export(techniques=["XRR"], thickness=103.4))
    db.commit()
    result = compare_parameter(db, SAMPLE, parameter="thickness")
    assert any("dq=0" in c for c in result["determinations"][0]["caveats"])


def test_a_clamped_determination_is_flagged(db):
    import_fit(db, _export(techniques=["XRR"], thickness=200.0, stack_id="v1"))
    db.commit()
    result = compare_parameter(db, SAMPLE, parameter="thickness")
    determination = result["determinations"][0]
    assert determination["at_bound"] is True
    assert any("clamped" in c for c in determination["caveats"])


def test_multiple_film_layers_are_skipped_rather_than_guessed(db):
    payload = _export(techniques=["XRR"], thickness=103.4)
    payload["stack"].insert(
        2,
        {
            "role": "layer",
            "label": "interlayer",
            "structural": {"thickness": {"value": 12.0, "vary": True}},
            "xray": {"sld_real": {"value": 20.0, "vary": True}},
        },
    )
    import_fit(db, payload)
    db.commit()

    unresolved = compare_parameter(db, SAMPLE, parameter="thickness")
    assert unresolved["unresolved_layer_fits"]
    assert "skipped rather than guessed" in unresolved["layer_hint"]

    #  Naming the layer resolves it.
    named = compare_parameter(db, SAMPLE, parameter="thickness", layer_label="interlayer")
    assert [d["value"] for d in named["determinations"]] == [12.0]


def test_unknown_parameter_is_rejected_with_the_valid_list(db):
    result = compare_parameter(db, SAMPLE, parameter="bandgap")
    assert "not a cross-technique comparable parameter" in result["error"]
    assert "thickness" in result["error"]


def test_report_rolls_up_every_disagreement(db):
    import_fit(db, _export(techniques=["XRR"], thickness=103.4, stack_id="v1"))
    import_fit(
        db,
        _export(
            techniques=["SE"], thickness=152.0, stack_id="v2",
            blocks={"optical": {"n": {"value": 2.05, "vary": True}}},
        ),
    )
    db.commit()

    report = cross_technique_report(db, SAMPLE)
    assert "thickness" in report["disagreements"]
    assert "thickness" in report["comparisons"]
    assert "disagree across techniques" in report["summary"]


def test_describe_fit_states_the_caveats_in_the_body(db):
    """The prose a model reads must not hide what qualifies each number."""
    result = import_fit(db, _export(techniques=["XRR"], thickness=103.4))
    db.commit()

    prose = describe_fit(db.get(FitRecord, result["fit_record_id"]))
    assert "hfo2_film" in prose
    assert "varied:" in prose
    assert "dq=0" in prose
    assert "reports no posterior" in prose


def test_describe_fit_says_so_when_chi_squared_is_missing(db):
    payload = _export(techniques=["XRR"], thickness=103.4)
    payload["fit"].pop("chi2")
    result = import_fit(db, payload)
    db.commit()
    assert "NOT RECORDED" in describe_fit(db.get(FitRecord, result["fit_record_id"]))
