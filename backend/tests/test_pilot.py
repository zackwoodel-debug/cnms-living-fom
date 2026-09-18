"""The HfO2-on-Si pilot loop.

Split into three groups:

* **Physics** — the XRR forward model and the property relations, checked
  against known values and against their own internal consistency.
* **Provenance** — that a simulated score stays ILLUSTRATIVE and a measured one
  does not. This is the property that makes the pilot safe to run.
* **Loop** — suggest → simulate → observe actually closes, and the campaign
  state moves.

No BoTorch needed: the cold-start path uses scipy's Sobol, which is exactly the
regime a fresh campaign is in.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base, get_db
from cnms_fom.db.models import BoObservation, FomDefinition, PropertyValue  # noqa: F401
from cnms_fom.fom_engine.definitions import all_draft_foms, to_definition_kwargs
from cnms_fom.main import app
from cnms_fom.pilot.properties import (
    CONSTANTS,
    equivalent_oxide_thickness,
    series_effective_k,
    simulate_properties,
)
from cnms_fom.pilot.simulate import critical_q, fringe_spacing_to_thickness, simulate_xrr
from cnms_fom.pilot.stack import Stack, StackLayer, xray_sld_from_composition

SEED = 20260823


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("formula", "expected", "tolerance"),
    [("Si", 20.0, 1.0), ("SiO2", 18.8, 1.0), ("HfO2", 68.0, 3.0)],
)
def test_xray_sld_matches_literature(formula, expected, tolerance):
    """Computed from composition and density; checked against tabulated values."""
    assert xray_sld_from_composition(formula) == pytest.approx(expected, abs=tolerance)


def test_critical_q_for_silicon():
    """q_c = 4 sqrt(pi SLD). Silicon is the standard calibration point."""
    assert critical_q(xray_sld_from_composition("Si")) == pytest.approx(0.0317, abs=0.001)


def _stack(thickness: float, roughness: float) -> Stack:
    return Stack(
        layers=[
            StackLayer("air", 0.0, 0.0, 0.0),
            StackLayer("HfO2", thickness, roughness, xray_sld_from_composition("HfO2")),
            StackLayer("SiO2", 10.0, 2.0, xray_sld_from_composition("SiO2")),
            StackLayer("Si", 0.0, 2.0, xray_sld_from_composition("Si")),
        ]
    )


def test_total_external_reflection_below_critical_angle():
    result = simulate_xrr(_stack(200.0, 2.0))
    below = result.reflectivity[result.q < 0.9 * critical_q(xray_sld_from_composition("Si"))]
    assert np.all(below > 0.99)


def test_reflectivity_decays_above_the_critical_edge():
    result = simulate_xrr(_stack(200.0, 2.0))

    def at(q: float) -> float:
        return float(result.reflectivity[int(np.argmin(np.abs(result.q - q)))])

    assert at(0.10) < 0.1
    assert at(0.20) < at(0.10)


def test_roughness_damps_the_high_q_tail():
    """Névot-Croce: a rougher interface reflects less at high q."""

    def tail(roughness: float) -> float:
        result = simulate_xrr(_stack(200.0, roughness))
        return float(result.reflectivity[int(np.argmin(np.abs(result.q - 0.25)))])

    assert tail(8.0) < tail(3.0) < tail(0.0)


def test_curve_encodes_the_thickness_that_went_in():
    """Closure check: the fringes must carry the thickness the stack was built with.

    Loose tolerance on purpose — a film-on-interlayer stack beats two fringe
    frequencies together, so a single mean spacing lands between them. This
    catches a broken forward model, not a mis-calibrated one.
    """
    for thickness in (100.0, 200.0, 300.0):
        recovered = fringe_spacing_to_thickness(simulate_xrr(_stack(thickness, 2.0)))
        assert recovered is not None
        assert recovered == pytest.approx(thickness + 10.0, rel=0.20)


def test_series_capacitance_is_bounded_by_its_layers():
    """A series stack cannot beat its best layer or undercut its worst."""
    k_eff = series_effective_k(100.0, 25.0, 10.0, 3.9)
    assert 3.9 < k_eff < 25.0


def test_thicker_film_approaches_the_film_permittivity():
    """The interlayer matters less as the high-k film grows."""
    thin = series_effective_k(50.0, 25.0, 10.0, 3.9)
    thick = series_effective_k(1000.0, 25.0, 10.0, 3.9)
    assert thin < thick < 25.0
    assert thick == pytest.approx(25.0, rel=0.1)


def test_eot_cannot_go_below_the_interfacial_layer():
    """The SiO2 interlayer sets the floor — the reason EOT scaling stalled."""
    for thickness in (10.0, 50.0, 400.0):
        assert equivalent_oxide_thickness(thickness, 25.0, 10.0) > 10.0


def test_roughness_derates_breakdown():
    smooth = simulate_properties({"thickness_ang": 100, "roughness_ang": 1})
    rough = simulate_properties({"thickness_ang": 100, "roughness_ang": 9})
    assert rough.properties["Ebd"] < smooth.properties["Ebd"]
    assert rough.properties["tan_delta"] > smooth.properties["tan_delta"]


def test_doping_trades_permittivity_against_bandgap():
    """The trade-off that gives the pilot an interior optimum."""
    undoped = simulate_properties({"thickness_ang": 150, "roughness_ang": 2, "dopant_fraction": 0.0})
    heavy = simulate_properties({"thickness_ang": 150, "roughness_ang": 2, "dopant_fraction": 0.30})
    assert heavy.properties["k"] < undoped.properties["k"]
    assert heavy.properties["Eg"] > undoped.properties["Eg"]
    assert heavy.properties["Ebd"] > undoped.properties["Ebd"]


def test_light_doping_raises_permittivity_via_phase_stabilisation():
    """HEURISTIC, but it is the shape the model is meant to have."""
    undoped = simulate_properties({"thickness_ang": 150, "roughness_ang": 2, "dopant_fraction": 0.0})
    light = simulate_properties(
        {"thickness_ang": 150, "roughness_ang": 2, "dopant_fraction": CONSTANTS.heuristic_phase_peak_x}
    )
    assert light.properties["k"] > undoped.properties["k"]


def test_simulated_properties_declare_which_relations_are_heuristics():
    """A reader must be able to tell derived physics from a plausible curve."""
    result = simulate_properties({"thickness_ang": 100, "roughness_ang": 2})
    assert "k" in result.derived_physics
    assert {"Ebd", "tan_delta"} <= set(result.heuristics)
    assert any("MODELED" in note for note in result.notes)


def test_zero_thickness_is_rejected():
    with pytest.raises(ValueError, match="thickness_ang must be positive"):
        simulate_properties({"thickness_ang": 0.0, "roughness_ang": 2.0})


# ---------------------------------------------------------------------------
# API and loop
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'pilot.db'}", future=True)
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    with TestingSession() as db:
        for spec in all_draft_foms().values():
            db.add(FomDefinition(**to_definition_kwargs(spec)))
        db.commit()

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


@pytest.fixture
def pilot_run(client):
    response = client.post("/pilot/hfo2_logic_run", json={"random_seed": SEED})
    assert response.status_code == 201, response.text
    return response.json()


def test_pilot_run_has_the_expected_search_space(pilot_run):
    names = {p["name"] for p in pilot_run["search_space"]["parameters"]}
    assert names == {"thickness_ang", "roughness_ang", "dopant_fraction"}
    assert pilot_run["objective_sense"] == "max"
    assert pilot_run["acquisition"] == "qLogEI"


def test_unknown_instrument_is_rejected(client):
    response = client.post("/pilot/hfo2_logic_run", json={"instrument_id": "NOT-A-TOOL"})
    assert response.status_code == 404


def test_iteration_produces_illustrative_scores(client, pilot_run):
    """The central provenance guarantee: a simulated loop cannot produce a ranking."""
    response = client.post(
        f"/pilot/hfo2_logic_run/{pilot_run['id']}/iterate",
        json={"q": 2, "persist": True, "seed": SEED},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["evaluations"]) == 2
    for evaluation in body["evaluations"]:
        assert evaluation["fom_status"] == "illustrative"
        assert evaluation["fom_value"] is not None
        #  ln F is the BO objective (bo_engine.surrogate).
        assert evaluation["objective_value"] == pytest.approx(
            np.log(evaluation["fom_value"])
        )
    assert "ILLUSTRATIVE" in body["disclaimer"]


def test_iteration_logs_observations_so_the_loop_closes(client, pilot_run):
    run_id = pilot_run["id"]
    first = client.post(
        f"/pilot/hfo2_logic_run/{run_id}/iterate", json={"q": 2, "persist": True, "seed": 1}
    ).json()
    second = client.post(
        f"/pilot/hfo2_logic_run/{run_id}/iterate", json={"q": 2, "persist": True, "seed": 2}
    ).json()

    assert first["n_observations_after"] == 2
    assert second["n_observations_after"] == 4
    assert client.get("/bo/runs").json()[0]["n_observations"] == 4


def test_dry_run_leaves_the_campaign_untouched(client, pilot_run):
    run_id = pilot_run["id"]
    body = client.post(
        f"/pilot/hfo2_logic_run/{run_id}/iterate",
        json={"q": 1, "persist": False, "seed": SEED},
    ).json()
    assert body["n_observations_after"] == 0
    assert body["evaluations"][0]["observation_id"] is None


def test_simulated_properties_are_stored_as_modeled(client, pilot_run):
    """So they can never be mistaken for measurements later."""
    client.post(
        f"/pilot/hfo2_logic_run/{pilot_run['id']}/iterate",
        json={"q": 1, "persist": True, "seed": SEED},
    )
    material = client.get("/materials").json()[0]
    detail = client.get(f"/materials/{material['id']}").json()
    assert detail["properties"]
    assert all(p["provenance_tier"] == "modeled" for p in detail["properties"])


def test_two_recipes_are_two_contexts_not_two_materials(client, pilot_run):
    """Eq. (3): processing is context, not identity.

    Both recipes are the same composition, polymorph and specimen form, so they
    share one Material and differ by ``processing_route`` — which is part of the
    context digest, so the uniqueness constraint does not collide.
    """
    run_id = pilot_run["id"]
    client.post(f"/pilot/hfo2_logic_run/{run_id}/iterate", json={"q": 2, "persist": True, "seed": 3})

    materials = client.get("/materials").json()
    assert len(materials) == 1

    detail = client.get(f"/materials/{materials[0]['id']}").json()
    assert len(detail["properties"]) >= 12  # 6 properties x 2 recipes


def test_stack_export_layers_and_sld_sources(client):
    response = client.get(
        "/pilot/hfo2_logic_run/stack", params={"thickness_ang": 150, "roughness_ang": 3}
    )
    assert response.status_code == 200
    stack = response.json()["stack"]

    names = [layer["name"] for layer in stack["layers"]]
    assert names == ["air", "HfO2", "SiO2", "Si"]
    assert stack["layers"][1]["thickness_ang"] == 150
    #  Every SLD says where it came from.
    assert all(layer["sld_source"] for layer in stack["layers"])


def test_ingest_experiment_yields_a_real_score(client, pilot_run):
    """Measured inputs, so the score is `scored` — not illustrative."""
    response = client.post(
        f"/pilot/hfo2_logic_run/{pilot_run['id']}/ingest_experiment",
        json={
            "recipe": {"thickness_ang": 120.0, "roughness_ang": 2.5},
            "external_experiment_id": "CNMS-2026-0042",
            "measurements": [
                {"property_key": "k", "value": 21.4, "temperature_k": 300, "frequency_hz": 1e4,
                 "tensor_component": "zz", "method": "CV"},
                {"property_key": "Eg", "value": 5.8, "method": "ellipsometry"},
                {"property_key": "dEc", "value": 1.6, "substrate": "Si(001)",
                 "interface": "HfO2/SiO2/Si", "method": "XPS"},
                {"property_key": "Ebd", "value": 4.4, "thickness_nm": 12.0, "electrode": "TiN",
                 "temperature_k": 300, "area_cm2": 1e-4, "failure_criterion": "1 mA/cm^2",
                 "method": "ramped voltage"},
            ],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["fom_status"] == "scored"
    assert body["properties_stored"] == 4
    assert body["objective_value"] is not None
    assert body["observation_id"] is not None


def test_ingest_experiment_refuses_a_value_missing_its_context(client, pilot_run):
    """Sec. 16: a breakdown field without thickness or electrode is not comparable."""
    response = client.post(
        f"/pilot/hfo2_logic_run/{pilot_run['id']}/ingest_experiment",
        json={
            "recipe": {"thickness_ang": 120.0, "roughness_ang": 2.5},
            "measurements": [{"property_key": "Ebd", "value": 4.4, "method": "ramped voltage"}],
        },
    )
    assert response.status_code == 422
    assert "thickness_nm" in str(response.json()["detail"])


def test_incomplete_experiment_is_not_scored_and_not_optimised_on(client, pilot_run):
    """No manufactured objective: an unscoreable run is logged infeasible."""
    response = client.post(
        f"/pilot/hfo2_logic_run/{pilot_run['id']}/ingest_experiment",
        json={
            "recipe": {"thickness_ang": 200.0, "roughness_ang": 4.0},
            "measurements": [
                {"property_key": "k", "value": 19.0, "temperature_k": 300, "frequency_hz": 1e4,
                 "tensor_component": "zz", "method": "CV"}
            ],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["fom_status"] == "not_scored"
    assert set(body["missing_inputs"]) == {"Eg", "dEc", "Ebd"}
    assert body["objective_value"] is None


def test_iteration_is_deterministic_under_a_fixed_seed(client, pilot_run):
    """CI depends on this."""
    run_id = pilot_run["id"]
    first = client.post(
        f"/pilot/hfo2_logic_run/{run_id}/iterate", json={"q": 2, "persist": False, "seed": 7}
    ).json()
    second = client.post(
        f"/pilot/hfo2_logic_run/{run_id}/iterate", json={"q": 2, "persist": False, "seed": 7}
    ).json()
    assert [e["recipe"] for e in first["evaluations"]] == [
        e["recipe"] for e in second["evaluations"]
    ]
