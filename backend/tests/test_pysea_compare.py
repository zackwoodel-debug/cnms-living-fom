"""Cross-platform comparison. The property under test is that it never averages.

Two instruments measuring one film is the whole point of the integration. When
they agree, the number is worth trusting. When they disagree, the disagreement is
the finding, and collapsing it into a mean destroys the only information the
second measurement added.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db import models  # noqa: F401 - registers the mappers
from cnms_fom.db.base import Base
from cnms_fom.db.enums import ProvenanceTier, SpecimenForm
from cnms_fom.db.models import FitRecord, Material, PropertyValue
from cnms_fom.pysea.compare import compare_across_platforms, cross_platform_disagreements
from cnms_fom.pysea.promote import promote
from cnms_fom.pysea.records import import_pysea_record
from tests.pysea_fixtures import EXPERIMENTAL, SIMULATION, fixture_envelope

SAMPLE = "HFO2-SI-042"


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'compare.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def material(db):
    record = Material(
        formula="HfO2", formula_reduced="HfO2", polymorph="monoclinic",
        specimen_form=SpecimenForm.CRYSTALLINE_FILM,
    )
    db.add(record)
    db.flush()
    return record


def _modalfit_value(db, material, value, uncertainty, *, sample_id=SAMPLE):
    """A promoted ModalFit determination of eps_inf for the same sample."""
    fit = FitRecord(
        sample_id=sample_id, stack_id="stack-1", content_sha256=f"hash-{value}",
        techniques=["SE"], chi2_total=1.4,
    )
    db.add(fit)
    db.flush()
    db.add(
        PropertyValue(
            material_id=material.id, property_key="eps_inf", value=value,
            units="dimensionless", uncertainty=uncertainty,
            method="ModalFit SE refinement", software="ModalFit",
            provenance_tier=ProvenanceTier.MEASURED,
            source_locator=f"fit_record:{fit.id}",
        )
    )
    db.flush()
    return fit


def test_no_determination_when_nothing_measured_it(db):
    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    assert result["verdict"] == "no_determination"
    assert result["determinations"] == []


def test_a_single_pysea_determination_is_not_cross_checked(db):
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    db.commit()

    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    assert result["verdict"] == "single_determination"
    assert result["n_determinations"] == 1
    assert "falsifiable" in result["summary"]


def test_two_platforms_agreeing_are_consistent(db, material):
    """4.18 ± 0.09 against 4.21 ± 0.10: inside the combined uncertainty."""
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    _modalfit_value(db, material, 4.21, 0.10)
    db.commit()

    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    assert result["verdict"] == "consistent"
    assert set(result["platforms"]) == {"pySEA", "ModalFit"}
    assert result["n_determinations"] == 2


def test_two_platforms_disagreeing_say_so_and_keep_both(db, material):
    """The demo: 4.18 ± 0.09 against 5.90 ± 0.08. Both survive; neither is merged."""
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    _modalfit_value(db, material, 5.90, 0.08)
    db.commit()

    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    assert result["verdict"] == "disagreement"
    assert result["n_determinations"] == 2

    values = sorted(d["value"] for d in result["determinations"])
    assert values == [4.18, 5.90]
    assert "Do not average them" in result["summary"]
    #  And no merged value is offered anywhere in the payload.
    assert "mean" not in result
    assert "preferred_value" not in result


def test_each_determination_carries_its_own_method_and_tier(db, material):
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    _modalfit_value(db, material, 5.90, 0.08)
    db.commit()

    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    by_platform = {d["platform"]: d for d in result["determinations"]}

    assert "collection 2.4 mrad" in by_platform["pySEA"]["method"]
    assert by_platform["pySEA"]["provenance_tier"] == ProvenanceTier.MEASURED.value
    assert "ModalFit" in by_platform["ModalFit"]["method"]


def test_a_simulation_is_labelled_modeled_in_the_comparison(db):
    """The experiment and its simulation, side by side, tiers intact."""
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    import_pysea_record(db, fixture_envelope(SIMULATION))
    db.commit()

    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    tiers = sorted(d["provenance_tier"] for d in result["determinations"])
    assert tiers == [ProvenanceTier.MEASURED.value, ProvenanceTier.MODELED.value]
    assert result["verdict"] == "consistent"


def test_mismatched_units_are_a_disagreement_not_a_conversion(db, material):
    """Converting here would assert an equivalence nobody recorded."""
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    fit = FitRecord(
        sample_id=SAMPLE, stack_id="s", content_sha256="h-units",
        techniques=["SE"], chi2_total=1.0,
    )
    db.add(fit)
    db.flush()
    db.add(
        PropertyValue(
            material_id=material.id, property_key="eps_inf", value=4.2,
            units="relative", uncertainty=0.1, method="ModalFit",
            provenance_tier=ProvenanceTier.MEASURED,
            source_locator=f"fit_record:{fit.id}",
        )
    )
    db.commit()

    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    assert result["verdict"] == "disagreement"
    assert "more than one unit" in result["summary"]


def test_an_unpromoted_scalar_still_appears(db, material):
    """Hiding a refused determination would let a disagreement go unseen."""
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    _modalfit_value(db, material, 5.90, 0.08)
    db.commit()

    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    pysea = next(d for d in result["determinations"] if d["platform"] == "pySEA")
    assert "unexamined" in pysea["label"]
    assert result["verdict"] == "disagreement"


def test_promotion_status_shows_in_the_label_after_promoting(db, material):
    result = import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    db.commit()
    promote(db, result["id"], material_id=material.id, dry_run=False)
    db.commit()

    compared = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    pysea = next(d for d in compared["determinations"] if d["platform"] == "pySEA")
    assert "promoted" in pysea["label"]


def test_the_overview_names_which_quantities_disagree(db, material):
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    _modalfit_value(db, material, 5.90, 0.08)
    db.commit()

    overview = cross_platform_disagreements(db, SAMPLE)
    assert "eps_inf" in overview["disagreements"]


def test_a_different_sample_is_not_compared(db, material):
    """Two films are two films, however similar their numbers."""
    import_pysea_record(db, fixture_envelope(EXPERIMENTAL))
    _modalfit_value(db, material, 5.90, 0.08, sample_id="SOME-OTHER-FILM")
    db.commit()

    result = compare_across_platforms(db, SAMPLE, quantity="eps_inf")
    assert result["n_determinations"] == 1
    assert result["verdict"] == "single_determination"
