"""The campaign snapshot and its warnings.

Deterministic by design: every warning here is computed from stored rows with no
model in the loop, because these are the observations a reader most needs and a
language model is least reliable at making. No Ollama, no BoTorch, no Postgres.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.models import BoObservation, BoRun, BoSuggestion, FomDefinition
from cnms_fom.research.campaign import (
    BOUNDARY_FRACTION,
    STALL_THRESHOLD,
    CampaignSnapshot,
    snapshot,
)

SPACE = {
    "parameters": [
        {"name": "substrate_temp_c", "kind": "continuous", "lower": 150.0, "upper": 400.0},
        {"name": "purge_s", "kind": "continuous", "lower": 1.0, "upper": 20.0},
        {"name": "substrate", "kind": "categorical", "choices": ["Si(100)", "Ge"]},
    ]
}


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'campaign.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    yield session
    session.close()
    engine.dispose()


def _run(db, **kwargs) -> BoRun:
    defaults = {
        "name": "hfo2_logic",
        "search_space": SPACE,
        "constraints": {"bounds": {}, "allowed_choices": {}, "notes": []},
        "acquisition": "qLogEI",
        "objective_sense": "max",
    }
    run = BoRun(**{**defaults, **kwargs})
    db.add(run)
    db.flush()
    return run


def _observe(db, run, value, *, feasible=True, temp=250.0):
    db.add(BoObservation(
        bo_run_id=run.id,
        parameters={"substrate_temp_c": temp, "purge_s": 6.0, "substrate": "Si(100)"},
        objective_value=value,
        is_feasible=feasible,
    ))
    db.flush()


def _suggest(db, run, *, temp=250.0, std=0.1, purge=6.0):
    db.add(BoSuggestion(
        bo_run_id=run.id,
        parameters={"substrate_temp_c": temp, "purge_s": purge, "substrate": "Si(100)"},
        acquisition_value=0.02,
        predicted_mean=-1.0,
        predicted_std=std,
        status="proposed",
    ))
    db.flush()


# --- basics ---------------------------------------------------------------


def test_unknown_campaign_raises(db):
    with pytest.raises(LookupError, match="No BO campaign"):
        snapshot(db, 999)


def test_an_empty_campaign_warns_that_the_first_batch_is_not_an_opinion(db):
    run = _run(db)
    db.commit()
    snap = snapshot(db, run.id)
    assert snap.n_observations == 0
    assert any("Sobol" in w for w in snap.warnings)


def test_best_so_far_and_stall_count(db):
    run = _run(db)
    for value in (-2.0, -1.2, -1.5, -1.8, -1.6):
        _observe(db, run, value)
    db.commit()

    snap = snapshot(db, run.id)
    assert snap.best_objective == pytest.approx(-1.2)
    #  Best was observation 2 of 5, so three evaluations have passed since.
    assert snap.evaluations_since_best_improved == 3
    assert snap.is_stalled is False


def test_a_stalled_campaign_is_flagged(db):
    run = _run(db)
    _observe(db, run, -1.2)
    for _ in range(STALL_THRESHOLD):
        _observe(db, run, -2.0)
    db.commit()

    snap = snapshot(db, run.id)
    assert snap.is_stalled is True
    assert any("has not improved" in w for w in snap.warnings)


def test_a_stall_with_an_uncertain_surrogate_is_not_called_convergence(db):
    run = _run(db)
    _observe(db, run, -3.0)
    _observe(db, run, -1.0)  # range is 2.0
    for _ in range(STALL_THRESHOLD):
        _observe(db, run, -2.5)
    _suggest(db, run, std=1.5)  # way above 25% of the range
    db.commit()

    snap = snapshot(db, run.id)
    assert snap.surrogate_is_uncertain is True
    stall = next(w for w in snap.warnings if "has not improved" in w)
    assert "not convergence" in stall


def test_a_stall_with_a_confident_surrogate_says_so_but_defers_to_the_boundary_check(db):
    run = _run(db)
    _observe(db, run, -3.0)
    _observe(db, run, -1.0)
    for _ in range(STALL_THRESHOLD):
        _observe(db, run, -2.5)
    _suggest(db, run, std=0.01, temp=275.0)
    db.commit()

    snap = snapshot(db, run.id)
    assert snap.surrogate_is_uncertain is False
    stall = next(w for w in snap.warnings if "has not improved" in w)
    assert "consistent with convergence" in stall
    assert "boundary warning" in stall


# --- infeasibility -------------------------------------------------------


def test_a_mostly_infeasible_campaign_is_a_constraint_problem(db):
    run = _run(db)
    for _ in range(3):
        _observe(db, run, None, feasible=False)
    _observe(db, run, -1.0)
    db.commit()

    snap = snapshot(db, run.id)
    assert snap.n_infeasible == 3
    warning = next(w for w in snap.warnings if "infeasible" in w)
    assert "constraint problem, not a search problem" in warning


def test_infeasible_observations_do_not_set_the_best(db):
    run = _run(db)
    _observe(db, run, -5.0, feasible=False)
    _observe(db, run, -2.0)
    db.commit()
    assert snapshot(db, run.id).best_objective == pytest.approx(-2.0)


# --- boundary detection --------------------------------------------------


def test_every_suggestion_on_a_bound_is_flagged(db):
    """A search space whose optimum lies outside its bounds looks converged."""
    run = _run(db)
    _observe(db, run, -1.0)
    for _ in range(3):
        _suggest(db, run, temp=400.0)  # the upper bound
    db.commit()

    snap = snapshot(db, run.id)
    assert "substrate_temp_c" in snap.boundary_parameters
    assert any("sits on a bound" in w for w in snap.warnings)
    assert any("somebody typed" in w for w in snap.warnings)


def test_one_suggestion_on_a_bound_is_the_optimizer_probing_not_a_warning(db):
    """*All*, not *any* — probing an edge is what an optimizer is for."""
    run = _run(db)
    _observe(db, run, -1.0)
    _suggest(db, run, temp=400.0)
    _suggest(db, run, temp=250.0)
    db.commit()

    snap = snapshot(db, run.id)
    assert snap.boundary_parameters == []
    assert not any("sits on a bound" in w for w in snap.warnings)


def test_boundary_detection_uses_a_relative_tolerance(db):
    run = _run(db)
    _observe(db, run, -1.0)
    #  Range is 250 C, so the tolerance is 5 C. 397 is inside it, 380 is not.
    _suggest(db, run, temp=397.0)
    db.commit()
    assert "substrate_temp_c" in snapshot(db, run.id).boundary_parameters

    for suggestion in db.query(BoSuggestion).all():
        db.delete(suggestion)
    _suggest(db, run, temp=380.0)
    db.commit()
    assert "substrate_temp_c" not in snapshot(db, run.id).boundary_parameters
    assert BOUNDARY_FRACTION == 0.02


def test_categorical_parameters_are_not_boundary_checked(db):
    run = _run(db)
    _observe(db, run, -1.0)
    _suggest(db, run)
    db.commit()
    assert "substrate" not in snapshot(db, run.id).boundary_parameters


# --- the objective -------------------------------------------------------


def test_an_unapproved_objective_says_the_ranking_is_not_a_result(db):
    definition = FomDefinition(
        name="logic", version=1, application="logic",
        weights={"k": 0.5, "Eg": 0.5}, normalization={}, floor_eps=1e-3, approved=False,
    )
    db.add(definition)
    db.flush()
    run = _run(db, fom_definition_id=definition.id)
    _observe(db, run, -1.0)
    db.commit()

    snap = snapshot(db, run.id)
    assert snap.fom_approved is False
    warning = next(w for w in snap.warnings if "UNAPPROVED" in w)
    assert "not a result" in warning
    assert "Sec. 6.2" in warning


def test_an_approved_but_unfrozen_objective_warns_about_shifting_bounds(db):
    definition = FomDefinition(
        name="logic", version=1, application="logic",
        weights={"k": 1.0}, normalization={}, floor_eps=1e-3,
        approved=True, approved_by="Z. Woodel", frozen=False,
    )
    db.add(definition)
    db.flush()
    run = _run(db, fom_definition_id=definition.id)
    _observe(db, run, -1.0)
    db.commit()

    warning = next(w for w in snapshot(db, run.id).warnings if "not frozen" in w)
    assert "Sec. 5.3" in warning


def test_a_campaign_with_no_objective_at_all_is_flagged(db):
    run = _run(db)
    _observe(db, run, -1.0)
    db.commit()
    assert any("no FOM definition" in w for w in snapshot(db, run.id).warnings)


# --- the fingerprint -----------------------------------------------------


def test_the_fingerprint_tracks_configuration_not_observations(db):
    """It is what the context bridge records before and after an apply."""
    run = _run(db)
    db.commit()
    before = snapshot(db, run.id).fingerprint()

    _observe(db, run, -1.0)
    db.commit()
    assert snapshot(db, run.id).fingerprint() == before, "observations must not move it"

    run.constraints = {"bounds": {"substrate_temp_c": [200.0, 300.0]},
                       "allowed_choices": {}, "notes": []}
    db.commit()
    assert snapshot(db, run.id).fingerprint() != before, "a constraint change must move it"


def test_the_snapshot_states_the_ln_f_scale_trap(db):
    run = _run(db)
    db.commit()
    payload = snapshot(db, run.id).as_dict()
    assert "ln F" in payload["objective_scale_note"]
    assert "factor of two" in payload["objective_scale_note"]


def test_a_snapshot_writes_nothing(db):
    """It is a read. Asserted because the whole loop depends on it."""
    run = _run(db)
    _observe(db, run, -1.0)
    _suggest(db, run)
    db.commit()

    before = (db.query(BoRun).count(), db.query(BoObservation).count(),
              db.query(BoSuggestion).count())
    snapshot(db, run.id)
    db.rollback()
    after = (db.query(BoRun).count(), db.query(BoObservation).count(),
             db.query(BoSuggestion).count())
    assert before == after
    assert isinstance(snapshot(db, run.id), CampaignSnapshot)
