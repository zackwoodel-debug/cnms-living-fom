"""The research assistant: the tool surface, the loop, and its guardrails.

The guardrail under test more than any other: an answer with no retrieval behind
it is replaced by an explicit data gap. Everything else in this pipeline is
retrieval quality; that one is the difference between an assistant and a chatbot
that has read about materials science.
"""

from __future__ import annotations

import copy

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cnms_fom.db.base import Base
from cnms_fom.db.enums import ProvenanceTier, SpecimenForm, SynthesisTechnique
from cnms_fom.db.models import ChatMessage, Document, DocumentChunk, Material, PropertyValue
from cnms_fom.modalfit.records import import_fit
from cnms_fom.rag_backend import memory
from cnms_fom.rag_backend.agent import NO_EVIDENCE_ANSWER, ask, available_models
from cnms_fom.rag_backend.tools import TOOLS, run_tool, tool_specs
from tests.fakes import ScriptedProvider, refusal_reply, text_reply, tool_reply

SAMPLE = "HFO2-PILOT-07"

EXPORT = {
    "stack_id": "v1",
    "sample_id": SAMPLE,
    "fit": {"techniques": ["XRR"], "algorithm": "L-BFGS-B", "chi2": 1.84},
    "stack": [
        {"role": "ambient", "label": "air"},
        {
            "role": "layer",
            "label": "hfo2_film",
            "material": "HfO2",
            "structural": {"thickness": {"value": 103.4, "min": 50.0, "max": 200.0, "vary": True}},
            "xray": {"sld_real": {"value": 64.6, "min": 55.0, "max": 75.0, "vary": True}},
        },
        {"role": "substrate", "label": "silicon", "material": "Si"},
    ],
}


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'assistant.db'}", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()

    document = Document(
        title="ALD of HfO2 on Si",
        filename="ald.pdf",
        content_sha256="hash-ald",
        technique=SynthesisTechnique.ALD,
        doi="10.0000/ald-hfo2",
        n_pages=1,
    )
    session.add(document)
    session.flush()
    session.add(
        DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            page=7,
            text=(
                "The growth per cycle saturated at 0.98 A/cycle between 200 and 300 C in a "
                "Beneq TFS-200, defining the ALD window for this precursor combination."
            ),
        )
    )

    material = Material(
        formula="HfO2",
        formula_reduced="HfO2",
        polymorph="monoclinic",
        specimen_form=SpecimenForm.CRYSTALLINE_FILM,
    )
    session.add(material)
    session.flush()
    session.add(
        PropertyValue(
            material_id=material.id,
            property_key="k",
            value=25.0,
            units="1",
            temperature_k=300.0,
            frequency_hz=10_000.0,
            method="parallel-plate capacitance",
            provenance_tier=ProvenanceTier.MEASURED,
            doi="10.0000/dielectric",
        )
    )

    import_fit(session, copy.deepcopy(EXPORT))
    session.commit()
    yield session
    session.close()
    engine.dispose()


# --- the tool surface ------------------------------------------------------


def test_every_tool_has_a_schema_and_a_stable_order():
    """Tool definitions head a cached prompt prefix; reordering invalidates it."""
    from cnms_fom.rag_backend.tools import write_tool_names

    names = [spec.name for spec in tool_specs()]
    assert names == sorted(set(TOOLS) - set(write_tool_names()))
    assert names == [spec.name for spec in tool_specs()]
    assert [s.name for s in tool_specs(allow_writes=True)] == sorted(TOOLS)
    for spec in tool_specs(allow_writes=True):
        assert spec.description.strip()
        assert spec.input_schema["type"] == "object"


def test_tool_schemas_are_strict_for_anthropic():
    """A hallucinated argument name should fail at the schema, not at the SQL."""
    for spec in tool_specs():
        rendered = spec.as_anthropic()
        assert rendered["strict"] is True
        assert rendered["input_schema"]["additionalProperties"] is False


def test_no_tool_reaches_an_analysis_table():
    """Sec. 15.2 is enforced by not building the path, not by asking nicely.

    The only writers are the two knowledge-card tools, and a card is quarantined
    by construction — see test_only_knowledge_cards_are_writable_and_only_on_request.
    """
    from cnms_fom.rag_backend.tools import write_tool_names

    forbidden = ("promote", "property", "material", "score", "experiment", "ingest")
    assert not [name for name in TOOLS if any(word in name for word in forbidden)
                and not name.startswith("lookup_")]
    assert set(write_tool_names()) == {"write_card", "link_cards"}


def test_unknown_tool_returns_an_error_the_model_can_read(db):
    result = run_tool(db, "delete_everything", {})
    assert "No tool named" in result["error"]
    assert "search_corpus" in result["available_tools"]


def test_bad_arguments_return_the_expected_schema(db):
    result = run_tool(db, "list_sample_fits", {"wrong_arg": 1})
    assert "Bad arguments" in result["error"]
    assert "sample_id" in result["expected_schema"]["properties"]


def test_list_sample_fits_returns_the_prose_with_caveats(db):
    result = run_tool(db, "list_sample_fits", {"sample_id": SAMPLE})
    assert result["n_fits"] == 1
    description = result["fits"][0]["description"]
    assert "hfo2_film" in description
    assert "dq=0" in description


def test_list_sample_fits_says_so_when_there_are_none(db):
    result = run_tool(db, "list_sample_fits", {"sample_id": "NOPE"})
    assert result["fits"] == []
    assert "No ModalFit refinements" in result["note"]


def test_property_lookup_carries_the_measurement_context(db):
    result = run_tool(db, "lookup_property_values", {"formula": "hfo2"})
    assert result["n_values"] == 1
    value = result["values"][0]
    assert value["property_key"] == "k"
    assert value["context"]["temperature_k"] == 300.0
    assert value["context"]["frequency_hz"] == 10_000.0
    assert value["provenance_tier"] == "measured"
    assert value["source"]["doi"] == "10.0000/dielectric"
    assert "not alternatives to be averaged" in result["note"]


def test_descriptor_dictionary_reports_an_unknown_key_with_the_valid_ones(db):
    result = run_tool(db, "descriptor_dictionary", {"key": "bandgap"})
    assert result["found"] is False
    assert "Eg" in result["available_properties"]

    assert run_tool(db, "descriptor_dictionary", {"key": "Eg"})["found"] is True


def test_invalid_technique_filter_is_reported_not_silently_dropped(db):
    result = run_tool(db, "search_corpus", {"query": "ALD window", "techniques": ["xrd"]})
    assert result["ignored_technique_filters"] == ["xrd"]
    assert "ald" in result["valid_techniques"]


def test_corpus_coverage_explains_what_absence_means(db):
    result = run_tool(db, "corpus_coverage", {})
    assert result["total_documents"] == 1
    assert "not a reason to substitute general knowledge" in result["note"]


# --- the loop --------------------------------------------------------------


def test_the_loop_chains_tools_and_records_every_step(db):
    provider = ScriptedProvider(
        [
            tool_reply("list_sample_fits", {"sample_id": SAMPLE}),
            tool_reply("compare_fit_techniques", {"sample_id": SAMPLE, "parameter": "thickness"}),
            text_reply("XRR gives 103.4 A on fit record 1 [fit_record:1]."),
        ]
    )
    answer = ask(db, "How thick is the film?", provider=provider)

    assert answer.tools_used == ["list_sample_fits", "compare_fit_techniques"]
    assert len(answer.steps) == 2
    assert answer.steps[0].result["n_fits"] == 1
    assert answer.insufficient_context is False
    assert provider.exhausted


def test_an_answer_with_no_tool_call_is_replaced_by_a_data_gap(db):
    """The single most important guardrail in the package."""
    provider = ScriptedProvider([text_reply("HfO2 has a dielectric constant of about 25.")])
    answer = ask(db, "What is the dielectric constant of HfO2?", provider=provider)

    assert answer.answer == NO_EVIDENCE_ANSWER
    assert answer.insufficient_context is True
    assert "25" not in answer.answer
    assert answer.steps == []


def test_require_evidence_can_be_turned_off_for_debugging(db):
    provider = ScriptedProvider([text_reply("ungrounded")])
    answer = ask(db, "anything", provider=provider, require_evidence=False)
    assert answer.answer == "ungrounded"


def test_a_provider_refusal_is_not_recorded_as_a_data_gap(db):
    provider = ScriptedProvider([refusal_reply("cyber")])
    answer = ask(db, "How thick is the film?", provider=provider)

    assert answer.refused_by_provider is True
    assert answer.refusal_category == "cyber"
    assert "declined to answer" in answer.answer
    assert "corpus and the stored records are unchanged" in answer.answer


def test_hitting_the_step_limit_forces_a_final_answer(db):
    provider = ScriptedProvider(
        [
            tool_reply("list_sample_fits", {"sample_id": SAMPLE}),
            tool_reply("list_sample_fits", {"sample_id": SAMPLE}),
            text_reply("[DATA GAP: explicitly unresolved] Ran out of budget."),
        ]
    )
    answer = ask(db, "How thick is the film?", provider=provider, max_steps=2)

    assert answer.hit_step_limit is True
    assert answer.insufficient_context is True
    #  The final call is made with no tools, so the model has to commit.
    assert provider.calls[-1]["tools"] == []


def test_empty_retrieval_marks_the_answer_insufficient_even_if_the_model_does_not(db):
    """A model that searched, found nothing, and answered anyway is caught here."""
    provider = ScriptedProvider(
        [
            tool_reply("lookup_property_values", {"formula": "GaAs"}),
            text_reply("GaAs has a dielectric constant of 12.9."),
        ]
    )
    answer = ask(db, "What is the dielectric constant of GaAs?", provider=provider)

    assert answer.steps[0].result["n_values"] == 0
    assert answer.insufficient_context is True


def test_sample_scope_is_stated_to_the_model(db):
    provider = ScriptedProvider(
        [tool_reply("list_sample_fits", {"sample_id": SAMPLE}), text_reply("103.4 A [fit_record:1].")]
    )
    ask(db, "How thick is this film?", sample_id=SAMPLE, provider=provider)

    primed = " ".join(str(m["content"]) for m in provider.calls[0]["messages"])
    assert "conversation scope" in primed
    assert SAMPLE in primed


def test_technique_scope_is_added_to_the_system_prompt(db):
    provider = ScriptedProvider(
        [tool_reply("search_corpus", {"query": "ALD window"}), text_reply("0.98 A/cycle [1].")]
    )
    ask(
        db,
        "What is the ALD window?",
        techniques=[SynthesisTechnique.ALD],
        provider=provider,
    )
    assert "only the ald partition" in provider.calls[0]["system"]


def test_citations_are_collected_and_deduplicated(db):
    provider = ScriptedProvider(
        [
            tool_reply("list_sample_fits", {"sample_id": SAMPLE}),
            tool_reply("compare_fit_techniques", {"sample_id": SAMPLE, "parameter": "thickness"}),
            text_reply("103.4 A [fit_record:1]."),
        ]
    )
    answer = ask(db, "How thick?", provider=provider)

    fit_citations = [c for c in answer.citations if c["kind"] == "modalfit_fit"]
    assert len(fit_citations) == 1  # the same fit reached by two tools is one source
    assert fit_citations[0]["citation"] == "fit_record:1"


def test_a_failing_tool_is_reported_to_the_model_rather_than_ending_the_turn(db):
    provider = ScriptedProvider(
        [
            tool_reply("compare_fit_techniques", {"sample_id": SAMPLE, "parameter": "nonsense"}),
            tool_reply("compare_fit_techniques", {"sample_id": SAMPLE, "parameter": "thickness"}),
            text_reply("103.4 A [fit_record:1]."),
        ]
    )
    answer = ask(db, "How thick?", provider=provider)
    assert "not a cross-technique comparable parameter" in answer.steps[0].result["error"]
    assert len(answer.steps) == 2


def test_available_models_reports_the_provider_and_the_tools():
    config = available_models()
    assert config["provider"] == "ollama"
    assert "search_corpus" in config["tools"]
    assert "off this machine" in config["anthropic"]["note"]


# --- memory ----------------------------------------------------------------


def test_a_turn_stores_the_evidence_and_replays_only_the_text(db):
    session = memory.get_or_create_session(db, sample_id=SAMPLE)
    provider = ScriptedProvider(
        [tool_reply("list_sample_fits", {"sample_id": SAMPLE}), text_reply("103.4 A [fit_record:1].")]
    )
    answer = ask(db, "How thick is the film?", provider=provider)
    memory.record_turn(db, session, "How thick is the film?", answer)
    db.commit()

    stored = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.turn_index)
        .all()
    )
    assert [m.role.value for m in stored] == ["user", "assistant"]
    #  Full tool results are kept: a summarised audit trail is not one.
    assert stored[1].tool_calls[0]["tool"] == "list_sample_fits"
    assert stored[1].tool_calls[0]["result"]["n_fits"] == 1
    assert stored[1].evidence

    #  History replays text only, so tool results do not refill the context.
    history = memory.load_history(db, session)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert all("tool_calls" not in m for m in history)


def test_the_first_question_becomes_the_conversation_title(db):
    session = memory.get_or_create_session(db)
    provider = ScriptedProvider(
        [tool_reply("corpus_coverage", {}), text_reply("One ALD document is indexed.")]
    )
    answer = ask(db, "What is indexed?", provider=provider)
    memory.record_turn(db, session, "What is indexed?", answer)
    db.commit()
    assert session.title == "What is indexed?"


def test_session_scope_is_filled_but_never_overwritten(db):
    memory.get_or_create_session(db, "key-1", sample_id=SAMPLE)
    #  Re-pointing a conversation at another sample would reattribute every
    #  answer already in it.
    again = memory.get_or_create_session(db, "key-1", sample_id="OTHER-SAMPLE")
    assert again.sample_id == SAMPLE

    unscoped = memory.get_or_create_session(db, "key-2")
    assert memory.get_or_create_session(db, "key-2", sample_id=SAMPLE).sample_id == SAMPLE
    assert unscoped.session_key == "key-2"


def test_history_never_opens_on_an_assistant_turn(db):
    """A window cut mid-exchange is malformed for every provider."""
    session = memory.get_or_create_session(db)
    provider = ScriptedProvider(
        [
            tool_reply("corpus_coverage", {}),
            text_reply("first"),
            tool_reply("corpus_coverage", {}),
            text_reply("second"),
        ]
    )
    for question in ("q1", "q2"):
        memory.record_turn(db, session, question, ask(db, question, provider=provider))
    db.commit()

    assert memory.load_history(db, session, turns=1)[0]["role"] == "user"


def test_data_gaps_are_counted_as_a_coverage_metric(db):
    session = memory.get_or_create_session(db)
    provider = ScriptedProvider([text_reply("ungrounded guess")])
    memory.record_turn(db, session, "q", ask(db, "q", provider=provider))
    db.commit()

    summary = memory.session_summary(db, session)
    assert summary["n_data_gaps"] == 1
    assert summary["messages_by_role"] == {"user": 1, "assistant": 1}


def test_transcript_returns_the_evidence_for_audit(db):
    session = memory.get_or_create_session(db)
    provider = ScriptedProvider(
        [tool_reply("list_sample_fits", {"sample_id": SAMPLE}), text_reply("103.4 A.")]
    )
    memory.record_turn(db, session, "How thick?", ask(db, "How thick?", provider=provider))
    db.commit()

    rows = memory.transcript(db, session)
    assert len(rows) == 2
    assert rows[1]["tool_calls"][0]["tool"] == "list_sample_fits"


def test_optional_tool_arguments_stay_optional_under_strict_mode():
    """Auto-requiring every property would force the model to invent filters."""
    by_name = {spec.name: spec.as_anthropic() for spec in tool_specs()}

    #  Mandatory arguments are still mandatory.
    assert by_name["list_sample_fits"]["input_schema"]["required"] == ["sample_id"]
    assert by_name["search_corpus"]["input_schema"]["required"] == ["query"]

    #  Pure-filter tools require nothing: "list HfO2's property values" must not
    #  also have to supply a property_key.
    assert by_name["lookup_property_values"]["input_schema"]["required"] == []
    assert by_name["corpus_coverage"]["input_schema"]["required"] == []

    #  An optional argument is never promoted into required.
    compare = by_name["compare_fit_techniques"]["input_schema"]
    assert "layer_label" in compare["properties"]
    assert "layer_label" not in compare["required"]


# --- write gating ----------------------------------------------------------


def test_only_knowledge_cards_are_writable_and_only_on_request():
    """The read-only rule is about analysis tables; cards are the narrow exception."""
    from cnms_fom.rag_backend.tools import write_tool_names

    assert write_tool_names() == ["link_cards", "write_card"]

    read_only = {spec.name for spec in tool_specs()}
    with_writes = {spec.name for spec in tool_specs(allow_writes=True)}
    assert not (read_only & set(write_tool_names()))
    assert with_writes - read_only == set(write_tool_names())

    #  Nothing reaches an analysis table, whichever surface is offered.
    forbidden = ("property", "material", "fit_record", "score", "promote")
    assert not [
        name for name in write_tool_names() if any(word in name for word in forbidden)
    ]


def test_a_write_tool_is_refused_when_writes_are_not_enabled(db):
    """The schema is withheld, so a call means the model invented the name."""
    result = run_tool(
        db, "write_card", {"slug": "concepts/x", "title": "X", "body": "Y"}, allow_writes=False
    )
    assert "writes are not enabled" in result["error"]
    from cnms_fom.db.models import KnowledgeCard

    assert db.query(KnowledgeCard).count() == 0


def test_a_card_written_by_the_assistant_lands_unreviewed(db):
    result = run_tool(
        db,
        "write_card",
        {
            "slug": "concepts/ald-window-hfo2",
            "title": "ALD window for HfO2",
            "body": "GPC saturates at 0.98 A/cycle between 200 and 300 C [1].",
            "sources": [{"kind": "document", "document_id": 1, "page": 7}],
        },
        allow_writes=True,
    )
    assert result["written"] is True
    assert result["status"] == "proposed"
    assert result["citable"] is False
    assert "not citable" in result["reminder"]


def test_the_agent_withholds_write_tools_by_default(db):
    provider = ScriptedProvider(
        [tool_reply("corpus_coverage", {}), text_reply("One ALD document is indexed.")]
    )
    ask(db, "What is indexed?", provider=provider)
    offered = provider.calls[0]["tools"]
    assert "write_card" not in offered
    assert "search_cards" in offered
    assert "read-only this turn" in provider.calls[0]["system"]


def test_the_agent_offers_write_tools_when_asked(db):
    provider = ScriptedProvider(
        [tool_reply("corpus_coverage", {}), text_reply("Recorded what I found.")]
    )
    ask(db, "What is indexed?", provider=provider, allow_card_writes=True)
    assert "write_card" in provider.calls[0]["tools"]
    assert "writable this turn" in provider.calls[0]["system"]


# --- BO tools --------------------------------------------------------------


@pytest.fixture
def campaign(db):
    from cnms_fom.db.models import BoObservation, BoRun, BoSuggestion

    run = BoRun(
        name="hfo2_logic",
        search_space={"substrate_temp_c": {"type": "float", "bounds": [200, 300]}},
        acquisition="qLogEI",
        objective_sense="max",
        random_seed=7,
    )
    db.add(run)
    db.flush()
    #  Best improves, then stalls for three evaluations.
    for value, feasible in ((-2.0, True), (-1.2, True), (-1.2, True), (None, False), (-1.5, True)):
        db.add(
            BoObservation(
                bo_run_id=run.id,
                parameters={"substrate_temp_c": 250},
                objective_value=value,
                is_feasible=feasible,
            )
        )
    db.add(
        BoSuggestion(
            bo_run_id=run.id,
            parameters={"substrate_temp_c": 300},
            acquisition_value=0.01,
            predicted_mean=-1.2,
            predicted_std=0.4,
        )
    )
    db.commit()
    return run


def test_bo_campaign_lists_and_reads(db, campaign):
    listing = run_tool(db, "lookup_bo_campaign", {})
    assert listing["campaigns"][0]["name"] == "hfo2_logic"

    detail = run_tool(db, "lookup_bo_campaign", {"run_id": campaign.id})
    assert detail["n_observations"] == 5
    assert detail["n_infeasible"] == 1
    assert detail["n_pending_suggestions"] == 1
    assert "ln F" in detail["objective_scale_note"]


def test_bo_history_reports_the_stall_not_just_the_points(db, campaign):
    """'Is this working' is a question about the trend, not any single point."""
    history = run_tool(db, "lookup_bo_history", {"run_id": campaign.id})
    assert history["best_objective"] == pytest.approx(-1.2)
    #  Best last improved at observation 2, and three feasible/infeasible
    #  evaluations have happened since.
    assert history["evaluations_since_best_improved"] == 3
    assert history["n_infeasible"] == 1
    assert len(history["trajectory"]) == 5
    assert "clipping the optimum" in history["interpretation_note"]


def test_bo_suggestions_expose_the_uncertainty(db, campaign):
    result = run_tool(db, "lookup_bo_suggestions", {"run_id": campaign.id})
    assert result["n_suggestions"] == 1
    assert result["uncertainty_summary"]["max_std"] == pytest.approx(0.4)
    assert "ln F scale" in result["interpretation_note"]


def test_an_unapproved_objective_is_flagged_on_the_campaign(db, campaign):
    """The campaign is optimising toward weights nobody has approved."""
    from cnms_fom.db.models import FomDefinition

    definition = FomDefinition(
        name="logic", version=1, application="logic",
        weights={"k": 0.5, "Eg": 0.5},
        normalization={}, floor_eps=1e-3, approved=False,
    )
    db.add(definition)
    db.flush()
    campaign.fom_definition_id = definition.id
    db.commit()

    detail = run_tool(db, "lookup_bo_campaign", {"run_id": campaign.id})
    assert "unapproved" in detail["objective_warning"]
    assert "not a result" in detail["objective_warning"]


def test_unknown_campaign_is_an_error_payload_not_an_exception(db):
    assert "error" in run_tool(db, "lookup_bo_history", {"run_id": 999})


# --- plausibility tools ----------------------------------------------------


def test_plausibility_tool_grades_its_findings(db):
    result = run_tool(
        db,
        "check_physical_plausibility",
        {"values": {"k": 0.4, "Eg": 5.7}, "formula": "HfO2"},
    )
    assert result["physical"] is False
    assert result["n_violations"] == 1
    assert "never grounds to exclude" in result["note"]


def test_plausibility_tool_explains_itself_when_given_nothing(db):
    result = run_tool(db, "check_physical_plausibility", {"values": {}})
    assert "usage" in result
    assert "k" in result["checkable_keys"]


def test_fit_plausibility_catches_an_sld_that_disagrees_with_its_density(db):
    """The fixture's HfO2 layer has no density, so add one that does not match."""
    from cnms_fom.db.models import FitLayer

    layer = db.query(FitLayer).filter(FitLayer.role == "layer").one()
    layer.density_g_cm3 = 4.0  # implies ~28e-6, not the stored 64.6
    db.commit()

    result = run_tool(db, "check_fit_plausibility", {"sample_id": SAMPLE})
    assert result["n_layers_checked"] == 1
    entry = result["layers"][0]
    assert entry["physical"] is False
    assert any("disagree" in f["message"] for f in entry["inconsistencies"])
    assert result["all_physical"] is False


def test_fit_plausibility_passes_a_consistent_layer(db):
    from cnms_fom.db.models import FitLayer

    layer = db.query(FitLayer).filter(FitLayer.role == "layer").one()
    layer.density_g_cm3 = 9.1  # consistent with the stored 64.6e-6
    db.commit()

    result = run_tool(db, "check_fit_plausibility", {"sample_id": SAMPLE})
    assert result["all_physical"] is True


def test_fit_plausibility_on_an_unknown_sample_says_so(db):
    result = run_tool(db, "check_fit_plausibility", {"sample_id": "NOPE"})
    assert result["layers"] == []
    assert "No ModalFit refinements" in result["note"]


def test_layer_label_resolves_by_material_as_well_as_label(db):
    """A live run caught these two tools disagreeing on what a layer_label means.

    `compare_fit_techniques` accepted the material ("HfO2") while
    `check_fit_plausibility` wanted only the label ("hfo2_film"), so the same
    argument produced a comparison from one and silence from the other — and
    silence reads as "checked, nothing to report".
    """
    for label in ("hfo2_film", "HfO2", "hfo2"):
        compared = run_tool(
            db,
            "compare_fit_techniques",
            {"sample_id": SAMPLE, "parameter": "thickness", "layer_label": label},
        )
        checked = run_tool(
            db, "check_fit_plausibility", {"sample_id": SAMPLE, "layer_label": label}
        )
        assert compared["determinations"], f"compare found nothing for {label!r}"
        assert checked["n_layers_checked"] >= 1, f"plausibility checked nothing for {label!r}"


def test_a_layer_label_that_matches_nothing_says_so_loudly(db):
    """Fits exist, the filter matched none — that must not look like a pass."""
    result = run_tool(
        db, "check_fit_plausibility", {"sample_id": SAMPLE, "layer_label": "not_a_layer"}
    )
    assert result["n_layers_checked"] == 0
    assert result["all_physical"] is None
    assert "not a clean result" in result["note"]
    #  And it names what the caller could have passed instead.
    assert "hfo2_film" in result["note"]


# --- the empty-evidence guard ---------------------------------------------


def test_an_answer_is_withheld_when_every_tool_came_back_empty(db):
    """Observed on a local model: it correctly said the corpus had nothing, then
    supplied "200-350 degC" from memory. Flagging the answer is not enough —
    the flag is metadata and the text is what a person reads."""
    from cnms_fom.rag_backend.agent import EMPTY_EVIDENCE_ANSWER

    provider = ScriptedProvider(
        [
            tool_reply("search_corpus", {"query": "MBE GaAs on Ge substrate temperature"}),
            text_reply(
                "The corpus does not cover this. However, literature suggests substrate "
                "temperatures below 300 C, and 200-350 C is a reasonable starting range."
            ),
        ]
    )
    answer = ask(db, "What substrate temperature for MBE of GaAs on Ge?", provider=provider)

    assert answer.answer == EMPTY_EVIDENCE_ANSWER
    assert answer.insufficient_context is True
    #  The fabricated numbers are gone from the answer...
    assert "200-350" not in answer.answer
    assert "300 C" not in answer.answer
    #  ...but preserved for transparency rather than silently dropped.
    assert "200-350 C" in answer.suppressed_answer
    assert answer.steps  # the empty search is still in the trail


def test_an_answer_stands_when_any_tool_returned_content(db):
    """Only a wholly empty turn is withheld. A real result must survive."""
    provider = ScriptedProvider(
        [
            tool_reply("search_corpus", {"query": "nothing here"}),
            tool_reply("list_sample_fits", {"sample_id": SAMPLE}),
            text_reply("XRR gives 103.4 A on fit record 1 [fit_record:1]."),
        ]
    )
    answer = ask(db, "How thick is the film?", provider=provider)

    assert "103.4" in answer.answer
    assert answer.suppressed_answer is None


def test_content_detection_ignores_a_tools_own_echo_and_notes(db, stub_dense_retrieval):
    """Every tool echoes its arguments and a note even when it found nothing, so a
    truthiness test on the payload would call an empty search 'content'.

    ``stub_dense_retrieval`` because the case needs a search that *completes* and
    finds nothing. Without it the test passed locally and failed in CI: with the
    ``rag`` extra absent the dense leg cannot run, ``hybrid_search`` correctly
    raises rather than reporting a data gap it did not establish, and the payload
    is an error instead of an empty result."""
    from cnms_fom.rag_backend.agent import _step_has_content

    empty_search = run_tool(db, "search_corpus", {"query": "gallium arsenide germanium"})
    assert empty_search.get("note")  # it does explain itself
    assert _step_has_content(empty_search) is False

    real = run_tool(db, "list_sample_fits", {"sample_id": SAMPLE})
    assert _step_has_content(real) is True

    assert _step_has_content({"error": "boom"}) is False


def test_a_plausibility_only_turn_is_not_treated_as_empty(db):
    """Checking whether a number is physical needs no retrieval at all."""
    provider = ScriptedProvider(
        [
            tool_reply("check_physical_plausibility", {"values": {"k": 0.4}}),
            text_reply("A relative permittivity of 0.4 is impossible: it would polarize "
                       "against the field."),
        ]
    )
    answer = ask(db, "Is k = 0.4 physical?", provider=provider)
    assert "impossible" in answer.answer
    assert answer.suppressed_answer is None
