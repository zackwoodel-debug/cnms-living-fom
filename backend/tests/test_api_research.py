"""API tests for /research.

No Ollama, no Postgres, no network: the brief endpoints take a scripted provider
through a dependency override, and the benchmark endpoint runs offline by design.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base, get_db
from cnms_fom.db.enums import CardCategory, SynthesisTechnique
from cnms_fom.db.models import (
    BoObservation,
    BoRun,
    Document,
    DocumentChunk,
    Experiment,
    FomDefinition,
)
from cnms_fom.main import app
from tests.fakes import ResearchProvider

HOT_WALL = (
    "Between 200 and 300 degC the growth per cycle was constant at 0.98 angstrom per cycle "
    "using TDMAH and water in a hot-wall reactor at 1.5 Torr."
)
CLAIM = {
    "field": "growth_per_cycle_ang", "value": 0.98, "units": "A/cycle", "tier": "measured",
    "context": {"technique": "ald", "temperature_k": 523.0, "precursor": "TDMAH",
                "chamber": "hot-wall"},
    "quote": "growth per cycle was constant at 0.98 angstrom per cycle", "confidence": 0.9,
}
SPACE = {
    "parameters": [
        {"name": "substrate_temp_c", "kind": "continuous", "lower": 150.0, "upper": 400.0},
        {"name": "substrate", "kind": "categorical", "choices": ["Si(100)", "Ge"]},
    ]
}


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'api.db'}", future=True)
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    engine.dispose()


@pytest.fixture
def client(session_factory, monkeypatch):
    """A client whose provider is scripted and whose retrieval is lexical-only."""
    from cnms_fom.research import policy as policy_module

    #  Every policy is made lexical-only for the test run, so no embedder is needed
    #  anywhere behind the API. use_lexical is forced on as well: `dense_only` would
    #  otherwise end up with no retriever at all, which the policy rightly refuses.
    lexical = {
        name: candidate.evolve(use_dense=False, use_lexical=True)
        for name, candidate in policy_module.CANDIDATES.items()
    }
    monkeypatch.setattr(policy_module, "CANDIDATES", lexical)
    monkeypatch.setattr("cnms_fom.routers.research.CANDIDATES", lexical)
    monkeypatch.setattr(
        "cnms_fom.research.policy.get_policy", lambda name: lexical[name]
    )
    monkeypatch.setattr(
        "cnms_fom.routers.research.get_policy", lambda name: lexical[name]
    )
    monkeypatch.setattr(
        "cnms_fom.routers.research._provider",
        lambda name, model: ResearchProvider(claims=[CLAIM]),
    )

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def corpus(session_factory):
    db = session_factory()
    document = Document(
        title="ALD of HfO2 hot-wall", filename="hotwall.pdf", content_sha256="hash1",
        technique=SynthesisTechnique.ALD, doi="10.0000/hotwall", n_pages=1,
    )
    db.add(document)
    db.flush()
    db.add(DocumentChunk(document_id=document.id, chunk_index=0, page=1, text=HOT_WALL))
    db.commit()
    db.close()


@pytest.fixture
def campaign(session_factory):
    db = session_factory()
    definition = FomDefinition(
        name="logic", version=1, application="logic", weights={"k": 1.0},
        normalization={}, floor_eps=1e-3, approved=False,
    )
    db.add(definition)
    db.flush()
    run = BoRun(
        name="hfo2_logic", fom_definition_id=definition.id, search_space=SPACE,
        constraints={"bounds": {}, "allowed_choices": {}, "notes": []},
    )
    db.add(run)
    db.flush()
    db.add(BoObservation(bo_run_id=run.id, parameters={"substrate_temp_c": 250.0},
                         objective_value=-1.2, is_feasible=True))
    db.commit()
    run_id = run.id
    db.close()
    return run_id


@pytest.fixture
def card(session_factory):
    from cnms_fom.knowledge.cards import review_card, upsert_card

    db = session_factory()
    upsert_card(
        db, slug="concepts/ald-window-hfo2", title="ALD window for HfO2",
        body="GPC saturates at 0.98 A/cycle between 200 and 300 C [1].",
        category=CardCategory.PROCESS_WINDOW,
        sources=[{"kind": "document", "document_id": 1, "page": 1}],
    )
    review_card(db, "concepts/ald-window-hfo2", reviewed_by="Z. Woodel")
    db.commit()
    db.close()
    return "concepts/ald-window-hfo2"


# --- briefs ---------------------------------------------------------------


def test_a_campaign_brief_carries_the_computed_warnings(client, corpus, campaign):
    response = client.post(
        f"/research/campaigns/{campaign}/brief",
        json={"research_question": "What is the growth per cycle for HfO2 ALD?",
              "techniques": ["ald"]},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["brief_id"]
    assert body["bo_run_id"] == campaign
    assert body["claims"]
    assert body["claims"][0]["is_measurement"] is False
    assert any("UNAPPROVED" in w for w in body["warnings"])
    assert "not a measurement" in body["disclaimer"]
    assert body["policy_version"]
    assert body["fingerprint"]


def test_a_brief_on_an_unknown_campaign_is_404(client, corpus):
    response = client.post(
        "/research/campaigns/999/brief", json={"research_question": "anything at all"}
    )
    assert response.status_code == 404


def test_an_unknown_policy_is_422(client, corpus):
    response = client.post(
        "/research/brief",
        json={"research_question": "growth per cycle for HfO2?", "policy": "nonsense"},
    )
    assert response.status_code == 422
    assert "Unknown policy" in response.json()["detail"]


def test_a_brief_writes_nothing_scientific(client, corpus, campaign, session_factory):
    from cnms_fom.db.models import DescriptorValue, FomScore, PropertyValue

    client.post(
        f"/research/campaigns/{campaign}/brief",
        json={"research_question": "growth per cycle for HfO2?", "techniques": ["ald"]},
    )
    db = session_factory()
    assert db.query(PropertyValue).count() == 0
    assert db.query(DescriptorValue).count() == 0
    assert db.query(FomScore).count() == 0
    db.close()


def test_a_stored_brief_round_trips_and_reviews(client, corpus):
    created = client.post(
        "/research/brief",
        json={"research_question": "growth per cycle for HfO2 ALD?", "techniques": ["ald"]},
    ).json()
    brief_id = created["brief_id"]

    fetched = client.get(f"/research/briefs/{brief_id}").json()
    assert fetched["research_question"] == created["research_question"]
    assert fetched["claims"]

    listing = client.get("/research/briefs").json()
    assert any(row["brief_id"] == brief_id for row in listing)

    refused = client.post(f"/research/briefs/{brief_id}/review", json={"reviewed_by": " "})
    assert refused.status_code == 422  # pydantic min_length

    reviewed = client.post(
        f"/research/briefs/{brief_id}/review", json={"reviewed_by": "Z. Woodel"}
    ).json()
    assert reviewed["status"] == "reviewed"
    assert "No property value" in reviewed["note"]


def test_an_unknown_brief_is_404(client):
    assert client.get("/research/briefs/999").status_code == 404


def test_claims_are_listed_per_field_without_aggregation(client, corpus):
    client.post(
        "/research/brief",
        json={"research_question": "growth per cycle for HfO2 ALD?", "techniques": ["ald"]},
    )
    body = client.get("/research/claims/growth_per_cycle_ang").json()
    assert body["n_claims"] >= 1
    assert all(claim["is_measurement"] is False for claim in body["claims"])
    assert "not measurements" in body["note"]


# --- campaign context -----------------------------------------------------


def test_the_campaign_snapshot_is_readable(client, campaign):
    body = client.get(f"/research/campaigns/{campaign}/snapshot").json()
    assert body["bo_run_id"] == campaign
    assert body["fingerprint"]
    assert "ln F" in body["objective_scale_note"]
    assert any("UNAPPROVED" in w for w in body["warnings"])


def _narrowing(card_slug):
    return {
        "recommended_bounds": [{
            "parameter": "substrate_temp_c", "lower": 200.0, "upper": 300.0,
            "rationale": "the reported ALD window",
        }],
        "supporting_card_slugs": [card_slug],
        "rationale": "narrow to the window",
    }


def test_propose_review_apply_is_the_only_path_to_the_optimizer(
    client, campaign, card, session_factory
):
    proposed = client.post(
        f"/research/campaigns/{campaign}/context/propose", json=_narrowing(card)
    )
    assert proposed.status_code == 201
    proposal_id = proposed.json()["proposal_id"]
    assert proposed.json()["status"] == "proposed"

    #  Applying before review is refused.
    early = client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/apply",
        json={"applied_by": "Z. Woodel"},
    )
    assert early.status_code == 409
    assert "only a REVIEWED proposal" in early.json()["detail"]

    db = session_factory()
    assert db.get(BoRun, campaign).constraints["bounds"] == {}
    db.close()

    reviewed = client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/review",
        json={"reviewed_by": "Z. Woodel"},
    ).json()
    assert reviewed["status"] == "reviewed"

    applied = client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/apply",
        json={"applied_by": "Z. Woodel"},
    ).json()
    assert applied["fingerprint_changed"] is True
    assert applied["constraints_after"]["bounds"]["substrate_temp_c"] == [200.0, 300.0]
    assert "Only BoRun.constraints changed" in applied["note"]


def test_a_widening_proposal_is_409(client, campaign, card):
    response = client.post(
        f"/research/campaigns/{campaign}/context/propose",
        json={
            "recommended_bounds": [{
                "parameter": "substrate_temp_c", "lower": 100.0, "upper": 500.0,
                "rationale": "a paper grew films at 450 C",
            }],
            "supporting_card_slugs": [card],
        },
    )
    assert response.status_code == 409
    assert "never widen" in response.json()["detail"]


def test_a_bound_without_a_reviewed_card_is_409(client, campaign):
    response = client.post(
        f"/research/campaigns/{campaign}/context/propose", json=_narrowing(None)
        | {"supporting_card_slugs": []}
    )
    assert response.status_code == 409
    assert "reviewed, sourced card" in response.json()["detail"]


def test_an_inverted_bound_is_422_at_the_schema(client, campaign, card):
    response = client.post(
        f"/research/campaigns/{campaign}/context/propose",
        json={
            "recommended_bounds": [{
                "parameter": "substrate_temp_c", "lower": 300.0, "upper": 200.0,
                "rationale": "inverted",
            }],
            "supporting_card_slugs": [card],
        },
    )
    assert response.status_code == 422
    assert "must exceed" in response.json()["detail"]


def test_review_requires_a_named_person(client, campaign, card):
    proposal_id = client.post(
        f"/research/campaigns/{campaign}/context/propose", json=_narrowing(card)
    ).json()["proposal_id"]
    response = client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/review", json={"reviewed_by": ""}
    )
    assert response.status_code == 422  # pydantic min_length


def test_an_applied_proposal_can_be_reverted(client, campaign, card, session_factory):
    proposal_id = client.post(
        f"/research/campaigns/{campaign}/context/propose", json=_narrowing(card)
    ).json()["proposal_id"]
    client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/review",
        json={"reviewed_by": "Z. Woodel"},
    )
    client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/apply",
        json={"applied_by": "Z. Woodel"},
    )
    reverted = client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/revert",
        json={"applied_by": "Z. Woodel"},
    ).json()
    assert reverted["constraints_after"]["bounds"] == {}

    db = session_factory()
    assert db.get(BoRun, campaign).constraints["bounds"] == {}
    db.close()


def test_the_context_log_records_who_did_what(client, campaign, card):
    proposal_id = client.post(
        f"/research/campaigns/{campaign}/context/propose", json=_narrowing(card)
    ).json()["proposal_id"]
    client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/review",
        json={"reviewed_by": "Z. Woodel"},
    )
    client.post(
        f"/research/campaigns/{campaign}/context/{proposal_id}/apply",
        json={"applied_by": "Z. Woodel"},
    )

    log = client.get(f"/research/campaigns/{campaign}/context").json()
    assert len(log) == 1
    assert log[0]["status"] == "applied"
    assert log[0]["applied_by"] == "Z. Woodel"
    assert log[0]["campaign_fingerprint_before"] != log[0]["campaign_fingerprint_after"]


# --- experiment summary ---------------------------------------------------


def test_an_experiment_summary_is_records_plus_labels(client, session_factory, campaign):
    db = session_factory()
    experiment = Experiment(sample_id="HFO2-PILOT-07", recipe={"substrate_temp_c": 250.0},
                            status="finished")
    db.add(experiment)
    db.flush()
    db.add(BoObservation(bo_run_id=campaign, experiment_id=experiment.id,
                         parameters={"substrate_temp_c": 250.0}, objective_value=-1.2,
                         is_feasible=True))
    db.commit()
    experiment_id = experiment.id
    db.close()

    body = client.post(
        f"/research/experiments/{experiment_id}/summary",
        json={"bo_run_id": campaign, "sample_id": "HFO2-PILOT-07"},
    ).json()
    assert body["outcome"]["result"]["objective_value_ln_f"] == pytest.approx(-1.2)
    assert "ln F" in body["outcome"]["scale_note"]
    assert "updated nothing" in body["disclaimer"]


def test_an_unknown_experiment_is_404(client):
    assert client.post("/research/experiments/999/summary", json={}).status_code == 404


# --- benchmark ------------------------------------------------------------


def test_the_benchmark_runs_offline_through_the_api(client):
    body = client.post(
        "/research/benchmarks/run",
        json={"policy": "baseline", "case_set": "hard", "persist_results": False},
    ).json()

    assert body["overall_score"] > 0.0
    assert body["extraction_available"] is False
    assert body["retrievers"] == ["lexical"]
    #  An unavailable metric is null, not zero.
    assert body["extraction_f1"] is None
    assert body["n_cases_skipped"] >= 1
    assert body["by_category"]


def test_an_unknown_case_set_is_422(client):
    response = client.post(
        "/research/benchmarks/run", json={"case_set": "nonsense", "persist_results": False}
    )
    assert response.status_code == 422


def test_policies_are_listed_with_their_diffs(client):
    body = client.get("/research/policies").json()
    assert "baseline" in body["policies"]
    assert body["policies"]["no_grading"]["diff_from_baseline"]["grade"] == [True, False]


def test_an_unknown_benchmark_id_is_404(client):
    assert client.get("/research/benchmarks/nonsense/results").status_code == 404
    ok = client.get("/research/benchmarks/hard/results")
    assert ok.status_code == 200
    assert ok.json()["n_cases"] == 6
