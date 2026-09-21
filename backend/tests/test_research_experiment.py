"""Experiment summaries: assembled from records, labelled, and side-effect free."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import (
    CardCategory,
    ScoreStatus,
    SpecimenForm,
    StatementKind,
)
from cnms_fom.db.models import (
    BoObservation,
    BoRun,
    BoSuggestion,
    Experiment,
    FomDefinition,
    FomScore,
    Material,
)
from cnms_fom.research.experiment import collect_outcome, propose_summary_card, summarise
from tests.fakes import SummaryProvider

SAMPLE = "HFO2-PILOT-07"
RECIPE = {"substrate_temp_c": 250.0, "purge_s": 6.0}


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'exp.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def scenario(db):
    """A campaign, a suggestion, an experiment, an observation and a score."""
    definition = FomDefinition(
        name="logic", version=1, application="logic", weights={"k": 1.0},
        normalization={}, floor_eps=1e-3, approved=False,
    )
    material = Material(formula="HfO2", formula_reduced="HfO2", polymorph="monoclinic",
                        specimen_form=SpecimenForm.CRYSTALLINE_FILM)
    db.add_all([definition, material])
    db.flush()

    run = BoRun(
        name="hfo2_logic", fom_definition_id=definition.id,
        search_space={"parameters": [
            {"name": "substrate_temp_c", "kind": "continuous", "lower": 150.0, "upper": 400.0},
            {"name": "purge_s", "kind": "continuous", "lower": 1.0, "upper": 20.0},
        ]},
        constraints={"bounds": {}, "allowed_choices": {}, "notes": []},
    )
    db.add(run)
    db.flush()

    db.add(BoSuggestion(bo_run_id=run.id, parameters=RECIPE, acquisition_value=0.05,
                        predicted_mean=-0.8, predicted_std=0.1, status="completed"))
    experiment = Experiment(sample_id=SAMPLE, recipe=RECIPE, status="finished")
    db.add(experiment)
    db.flush()

    score = FomScore(material_id=material.id, fom_definition_id=definition.id,
                     value=0.3, status=ScoreStatus.SCORED)
    db.add(score)
    db.flush()
    db.add(BoObservation(bo_run_id=run.id, experiment_id=experiment.id, parameters=RECIPE,
                         objective_value=-1.2, is_feasible=True, fom_score_id=score.id))
    db.commit()
    return {"run": run, "experiment": experiment, "material": material, "score": score}


# --- the deterministic half ----------------------------------------------


def test_the_outcome_is_assembled_from_records(db, scenario):
    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    assert outcome.recipe == RECIPE
    assert outcome.objective_value == pytest.approx(-1.2)
    assert outcome.predicted_mean == pytest.approx(-0.8)
    assert outcome.fom_definition == "logic v1"
    assert outcome.fom_status == "scored"


def test_the_prediction_miss_is_computed_and_flagged(db, scenario):
    """Predicted -0.8 ± 0.1, got -1.2: four sigma out."""
    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    assert outcome.prediction_error == pytest.approx(-0.4)
    assert outcome.prediction_was_within_uncertainty is False
    warning = next(w for w in outcome.warnings if "missed the optimizer" in w)
    assert "two standard deviations" in warning
    assert "ln F" in warning


def test_an_unapproved_objective_makes_the_number_not_a_result(db, scenario):
    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    assert any("UNAPPROVED" in w and "not a result" in w for w in outcome.warnings)


def test_a_modeled_score_is_illustrative(db, scenario):
    scenario["score"].uses_modeled_inputs = True
    scenario["score"].status = ScoreStatus.ILLUSTRATIVE
    db.commit()

    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    assert outcome.result_tier == "modeled"
    assert any("ILLUSTRATIVE" in w and "Sec. 2.3" in w for w in outcome.warnings)


def test_a_not_scored_result_is_a_missing_measurement_not_a_low_score(db, scenario):
    scenario["score"].status = ScoreStatus.NOT_SCORED
    scenario["score"].value = None
    scenario["score"].missing_inputs = ["Eg", "Ebd"]
    db.commit()

    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    warning = next(w for w in outcome.warnings if "NOT SCORED" in w)
    assert "not a low score" in warning


def test_an_infeasible_point_is_information_not_a_failure(db, scenario):
    db.query(BoObservation).one().is_feasible = False
    db.commit()

    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    warning = next(w for w in outcome.warnings if "infeasible" in w)
    assert "bounds the feasible region" in warning
    assert "not a failed measurement" in warning


def test_an_unmatched_prediction_says_so_rather_than_borrowing_one(db, scenario):
    """There is no foreign key from an observation to its suggestion."""
    db.query(BoSuggestion).one().parameters = {"substrate_temp_c": 999.0}
    db.commit()

    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    assert outcome.predicted_mean is None
    assert any("could not be recovered" in w for w in outcome.warnings)


def test_fit_and_plausibility_warnings_are_collected(db, scenario):
    """A fit whose SLD contradicts its own density."""
    from cnms_fom.modalfit.records import import_fit

    import_fit(db, {
        "stack_id": "xrr", "sample_id": SAMPLE,
        "fit": {"techniques": ["XRR"], "algorithm": "L-BFGS-B", "chi2": 1.8},
        "stack": [
            {"role": "ambient", "label": "air"},
            {"role": "layer", "label": "hfo2_film", "material": "HfO2",
             "structural": {"thickness": {"value": 250.0, "min": 50.0, "max": 250.0,
                                          "vary": True}},
             "xray": {"sld_real": {"value": 64.6, "min": 55.0, "max": 75.0, "vary": True}},
             "molecular": {"formula": "HfO2",
                           "density": {"value": 4.0, "min": 3.0, "max": 10.0, "vary": True}}},
            {"role": "substrate", "label": "silicon", "material": "Si"},
        ],
    })
    db.commit()

    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    assert any("disagrees with itself" in w for w in outcome.warnings)
    assert any("clamped" in w for w in outcome.fit_warnings)


def test_collecting_an_outcome_writes_nothing(db, scenario):
    from cnms_fom.db.models import PropertyValue

    before = (db.query(BoObservation).count(), db.query(FomScore).count(),
              db.query(PropertyValue).count())
    collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    assert (db.query(BoObservation).count(), db.query(FomScore).count(),
            db.query(PropertyValue).count()) == before


# --- the narrative -------------------------------------------------------


def test_statements_are_labelled_and_the_next_question_is_captured(db, scenario):
    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    summary = summarise(db, outcome, provider=SummaryProvider())

    assert [s.kind for s in summary.statements] == [
        StatementKind.EVIDENCE, StatementKind.INTERPRETATION, StatementKind.PROPOSAL
    ]
    assert summary.next_question
    assert summary.open_questions
    assert "updated nothing" in summary.as_dict()["disclaimer"]


def test_a_refused_or_malformed_narrative_leaves_the_outcome_intact(db, scenario):
    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    refused = summarise(db, outcome, provider=SummaryProvider(refuse=True))
    assert refused.statements == []
    assert any("declined" in w for w in outcome.warnings)

    outcome2 = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    malformed = summarise(db, outcome2, provider=SummaryProvider(malformed=True))
    assert malformed.statements == []
    assert any("not parseable" in w for w in outcome2.warnings)


def test_an_evidence_statement_is_wired_to_the_record_it_rests_on(db, scenario):
    """Its warrant is the outcome record, not a document page — and the locator says so."""
    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    provider = SummaryProvider(statements=[
        {"kind": "evidence", "text": "The run returned ln F = -1.20."}
    ])
    summary = summarise(db, outcome, provider=provider)

    statement = summary.statements[0]
    assert statement.kind is StatementKind.EVIDENCE
    assert statement.evidence
    warrant = statement.evidence[0]
    #  Distinguishable from a literature citation at a glance.
    assert warrant.document_title == f"experiment:{scenario['experiment'].id}"
    assert warrant.retrieval_method == "record"
    assert "not a literature citation" in warrant.grade_reason
    assert "objective (ln F) -1.2" in warrant.quote


def test_summarising_without_a_provider_returns_the_outcome_alone(db, scenario):
    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    summary = summarise(db, outcome, provider=None)
    assert summary.statements == []
    assert summary.outcome.objective_value == pytest.approx(-1.2)


def test_summarising_writes_nothing(db, scenario):
    from cnms_fom.db.models import KnowledgeCard, PropertyValue

    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    summarise(db, outcome, provider=SummaryProvider())
    assert db.query(PropertyValue).count() == 0
    assert db.query(KnowledgeCard).count() == 0  # a card needs an explicit call


# --- the proposed card ---------------------------------------------------


def test_the_summary_card_is_proposed_and_carries_the_labels(db, scenario):
    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    summary = summarise(db, outcome, provider=SummaryProvider())
    slug = propose_summary_card(db, summary)
    db.commit()

    from cnms_fom.knowledge.cards import read_card

    card = read_card(db, slug)
    assert card["status"] == "proposed"
    assert card["citable"] is False
    assert card["category"] == CardCategory.EXPERIMENT_SUMMARY.value
    #  A reviewer can see which sentences were evidence and which were a reading.
    assert "## Evidence" in card["body"]
    assert "## Interpretation" in card["body"]
    assert "## Proposals" in card["body"]
    #  And the caveats travel with it.
    assert "## Caveats" in card["body"]
    assert "UNAPPROVED" in card["body"]


def test_the_summary_card_is_never_a_side_effect(db, scenario):
    from cnms_fom.db.models import KnowledgeCard

    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    summarise(db, outcome, provider=SummaryProvider())
    db.commit()
    assert db.query(KnowledgeCard).count() == 0

    propose_summary_card(db, summarise(db, outcome, provider=SummaryProvider()))
    db.commit()
    assert db.query(KnowledgeCard).count() == 1


def test_a_proposed_summary_card_cannot_supply_bo_context(db, scenario):
    """It is an experiment_summary, which the bridge does not read even once reviewed."""
    from cnms_fom.research.bo_context import validate_supporting_cards

    outcome = collect_outcome(db, experiment_id=scenario["experiment"].id, sample_id=SAMPLE)
    slug = propose_summary_card(db, summarise(db, outcome, provider=SummaryProvider()))
    db.commit()

    _, problems = validate_supporting_cards(db, [slug])
    assert any("not reviewed" in p for p in problems)
