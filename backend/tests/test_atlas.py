"""The materials atlas as the assistant sees it: five read-only tools over a fixture index.

The fixture (``fixtures/atlas/index``) is a real atlas build of twelve real materials,
made with ``python materials-atlas/build.py --materials 38,42,72,77,132,133,219,223,232,310,316,1840
--out backend/tests/fixtures/atlas --index-only``. They were chosen for what they exercise:
fused silica (one source, grade C, a synonym), TiO2 (a formula, rutile data only), As2S3,
galena (a synonym), diamond and graphite (one formula, two materials), CuGaS2 (a voter whose
label contradicts its stored temperature), As2Se3 (the suspicious k consensus), water (grade A
and three ice phases graded T), GaAs (a temperature series), CdTe (model fits only, grade D)
and a SCHOTT catalog glass (a maker code).
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from cnms_fom.config import get_settings
from cnms_fom.rag_backend.agent import EMPTY_EVIDENCE_ANSWER, ask
from cnms_fom.rag_backend.tools import TOOLS, run_tool, tool_specs
from tests.fakes import ScriptedProvider, text_reply, tool_reply

FIXTURE = Path(__file__).parent / "fixtures" / "atlas" / "index" / "atlas_index.sqlite"
ATLAS_PACKAGE = Path(__file__).resolve().parents[1] / "cnms_fom" / "atlas"
ATLAS_TOOLS = ("atlas_compare", "atlas_coverage", "atlas_lookup", "atlas_read", "atlas_search")


@pytest.fixture(autouse=True)
def atlas_index(monkeypatch):
    monkeypatch.setenv("MATERIALS_ATLAS_INDEX", str(FIXTURE))
    monkeypatch.setenv("MATERIALS_ATLAS_RELEASE", "0.22.0")
    get_settings.cache_clear()
    yield FIXTURE
    get_settings.cache_clear()


def lookup(material, quantity="all"):
    return run_tool(None, "atlas_lookup", {"material": material, "quantity": quantity})


def values(result, quantity):
    return [v for m in result["matches"] for v in m["values"] if v["quantity"] == quantity]


# --- the surface -----------------------------------------------------------------------


def test_the_five_atlas_tools_are_offered_read_only():
    offered = {spec.name for spec in tool_specs()}
    assert set(ATLAS_TOOLS) <= offered
    assert not any(TOOLS[name].writes for name in ATLAS_TOOLS)


# --- lookup ------------------------------------------------------------------------------


def test_lookup_by_synonym_returns_the_stored_value_with_its_provenance():
    result = lookup("fused silica", "n")
    assert result["origin"] == "materials-db v0.22.0 (reference)"
    assert len(result["release_sha256"]) == 64
    [match] = result["matches"]
    assert (match["material_id"], match["material"], match["matched_on"]) == (38, "Silicon dioxide (fused silica)", ["synonym"])
    [n] = values(result, "n_633nm")
    assert n["value"] == 1.457012124643149  # exactly the consensus_properties value
    assert n["scope"] == "structure:amorphous"
    assert (n["grade"], n["classification"], n["citable"]) == ("C", "single_source", True)
    assert n["sources"][0]["doi"] == "10.1364/JOSA.55.001205"


def test_a_formula_naming_two_materials_returns_both_and_picks_neither():
    result = lookup("C", "density")
    assert result["ambiguous"] is True
    assert {m["material"] for m in result["matches"]} == {"Diamond", "Graphite"}
    assert "do not present any one of them as the answer" in result["note"]
    #  The ready-to-quote lines name both, each with its id.
    assert any(line.startswith("Diamond (id 132): ") for line in result["summary"])
    assert any(line.startswith("Graphite (id 133): ") for line in result["summary"])
    for match in result["matches"]:
        assert match["matched_on"] == ["formula"]


def test_tio2_is_one_material_whose_data_are_rutile_only():
    result = lookup("TiO2", "n_633nm")
    [match] = result["matches"]
    assert match["key"] == "oxides_50:TiO2@rutile"
    assert {v["scope"] for v in values(result, "n_633nm")} == {"rutile | o-ray", "rutile | e-ray"}
    assert "Several n-at-633-nm entries" in result["note"]


def test_model_fit_only_is_grade_d_with_the_family_table_value():
    [n] = values(lookup("CdTe", "n"), "n_633nm")
    assert (n["grade"], n["basis"]) == ("D", "primary_dataset_point")
    assert "only model fits" in n["grade_reason"]
    assert n["citable"] is False  # a family-table value: the note names its dataset's source


def test_water_has_an_independent_consensus_and_ice_only_at_other_temperatures():
    entries = {v["scope"]: v for v in values(lookup("Water", "n"), "n_633nm")}
    assert entries["liquid"]["grade"] == "A"
    assert entries["liquid"]["independent_groups"] == 5
    assert entries["ice"]["grade"] == "T"
    assert "value" not in entries["ice"] or entries["ice"]["value"] is None


def test_the_suspicious_k_consensus_is_reported_as_suspicious():
    [k] = values(lookup("As2Se3", "k"), "k_633nm")
    assert k["classification"] == "suspicious"


def test_a_voter_contradicting_its_own_label_is_excluded_from_the_grade():
    entries = {v["scope"]: v for v in values(lookup("Copper gallium sulfide", "n"), "n_633nm")}
    assert entries["e-ray"]["grade"] == "C"
    assert "Boyd1971-120C | e-ray names 120 degC" in entries["e-ray"]["grade_reason"]


def test_k_status_is_a_status_and_a_missing_k_is_never_zero():
    result = lookup("fused silica", "k_status")
    [status] = values(result, "k_status")
    assert status["status"] == "not_measured"
    assert "value" not in status
    assert values(result, "k_633nm") == []


def test_calculated_values_say_so():
    result = lookup("galena", "xray_sld")
    [real], [imag] = values(result, "xray_sld_real"), values(result, "xray_sld_imag")
    assert real["basis"] == imag["basis"] == "calculated"
    assert real["kind"] == "calculated_sld" and "periodictable" in real["kind_reason"]
    assert "grade" not in real


def test_a_catalog_glass_by_maker_code():
    result = lookup("N-BK7HT", "density")
    [match] = result["matches"]
    assert match["material"] == "SCHOTT N-BK7HT" and match["matched_on"] == ["maker_code"]


def test_a_material_the_release_rejected_says_why():
    result = lookup("Indium nitride", "density")
    assert result["matches"] == []
    assert "no RI.info dataset" in result["absent_from_release"][0]["reason"]


def test_a_material_not_in_the_atlas_is_not_answered():
    result = lookup("unobtainium", "density")
    assert result["matches"] == [] and "none may be supplied from memory" in result["note"]


def test_an_unknown_quantity_is_an_error_with_the_valid_ones():
    result = lookup("fused silica", "colour")
    assert "error" in result and "n_633nm" in result["valid_quantities"]


# --- search, read, compare, coverage ----------------------------------------------------------


def test_search_puts_an_exact_name_first():
    result = run_tool(None, "atlas_search", {"query": "fused silica"})
    assert result["hits"][0]["material_id"] == 38
    assert "exact alias" in result["hits"][0]["found_by"]


@pytest.mark.parametrize(("token", "material_id"), [("SiO2", 38), ("o-ray", None), ("structure:amorphous", None)])
def test_search_keeps_formulas_and_labels_intact(token, material_id):
    hits = run_tool(None, "atlas_search", {"query": token, "limit": 12})["hits"]
    assert hits, token
    if material_id is not None:
        assert hits[0]["material_id"] == material_id
    if token == "o-ray":
        assert all("e-ray" not in h["snippet"] or "o-ray" in h["snippet"] for h in hits)


def test_read_one_section_stands_alone():
    result = run_tool(None, "atlas_read", {"note_id": "material.oxides_50.sio2-amorphous", "section": "Evidence at a glance"})
    [section] = result["sections"]
    assert "Silicon dioxide (fused silica)" in section["text"] and "id 38" in section["text"]
    assert result["frontmatter"]["material_id"] == 38
    assert {"relation": "member_of", "to": "family.oxides_50", "note": ""} in result["links"]


def test_read_an_unknown_note_suggests_rather_than_guesses():
    result = run_tool(None, "atlas_read", {"note_id": "fused silica"})
    assert "error" in result and "material.oxides_50.sio2-amorphous" in result["did_you_mean"]


def test_compare_keeps_each_material_separate():
    result = run_tool(None, "atlas_compare", {"materials": ["Diamond", "Graphite"], "quantity": "density"})
    assert [r["matches"][0]["material"] for r in result["materials"]] == ["Diamond", "Graphite"]
    assert "never average" in result["note"]


def test_coverage_whole_atlas_and_one_material():
    whole = run_tool(None, "atlas_coverage", {})
    assert whole["coverage"]["counts"]["materials"] == 12
    assert any("band gap" in item for item in whole["coverage"]["not_in_the_atlas"])
    one = run_tool(None, "atlas_coverage", {"material": "Water"})
    assert one["coverage"][0]["material_id"] == 232


# --- a missing or stale index is an error, not "no data" ------------------------------------


def test_a_missing_index_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MATERIALS_ATLAS_INDEX", str(tmp_path / "absent.sqlite"))
    get_settings.cache_clear()
    result = lookup("fused silica", "n")
    assert result["atlas_unavailable"] is True
    assert "not built" in result["error"] and "not the same as the literature having no value" in result["error"]


def test_a_stale_index_is_refused(monkeypatch):
    monkeypatch.setenv("MATERIALS_ATLAS_RELEASE", "0.21.0")
    get_settings.cache_clear()
    result = lookup("fused silica", "n")
    assert result["atlas_unavailable"] is True and "stale" in result["error"]


def test_a_sample_build_is_refused(tmp_path, monkeypatch):
    copy = tmp_path / "sample.sqlite"
    copy.write_bytes(FIXTURE.read_bytes())
    with sqlite3.connect(copy) as connection:
        connection.execute("UPDATE meta SET value = 'true' WHERE key = 'sample'")
    monkeypatch.setenv("MATERIALS_ATLAS_INDEX", str(copy))
    get_settings.cache_clear()
    assert "sample build" in lookup("fused silica", "n")["error"]


# --- no write path ------------------------------------------------------------------------------


def test_the_atlas_package_imports_no_session_or_model_writer():
    forbidden = ("sqlalchemy", "cnms_fom.db", "cnms_fom.ingest", "cnms_fom.modalfit", "cnms_fom.fom_engine", "cnms_fom.bo_engine")
    for path in ATLAS_PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"


def test_the_index_is_opened_read_only():
    from cnms_fom.atlas import open_atlas

    atlas = open_atlas(FIXTURE)
    with pytest.raises(sqlite3.OperationalError):
        atlas.connection.execute("DELETE FROM notes")
    atlas.close()
    source = (ATLAS_PACKAGE / "reader.py").read_text()
    assert "mode=ro" in source and source.count("sqlite3.connect(") == 1


# --- in the assistant loop ---------------------------------------------------------------------------


FRAMING = (
    "Fused silica has one measured value at 633 nm; with a single source it has not been "
    "cross-checked, so treat it as a reference value."
)


def test_an_atlas_answer_is_a_checked_framing_over_verbatim_values():
    provider = ScriptedProvider(
        [
            tool_reply("atlas_lookup", {"material": "fused silica", "quantity": "n_633nm"}),
            text_reply("n of fused silica is 1.457 (grade C)."),  # the loop's draft
            text_reply(FRAMING),  # the framing call, tools withheld
        ]
    )
    answer = ask(None, "What is n of fused silica at 633 nm?", provider=provider)
    assert answer.composed is True and answer.model_text == FRAMING
    #  The facts are the lookup's own lines, quoted under a heading that says what they are.
    assert answer.answer.startswith(FRAMING + "\n\n**Reference values**")
    [line] = answer.reference_values
    assert line.startswith("Silicon dioxide (fused silica) (id 38): n_633nm | structure:amorphous = 1.457012124643149")
    assert line in answer.answer
    assert answer.suppressed_answer == "n of fused silica is 1.457 (grade C)."
    [citation] = [c for c in answer.citations if c["kind"] == "atlas_value"]
    assert citation["material_id"] == 38 and citation["citable"] is True and citation["grade"] == "C"
    #  The framing call sees the values as lines and no tools.
    assert provider.calls[2]["tools"] == [] and "- Silicon dioxide (fused silica) (id 38)" in provider.calls[2]["messages"][0]["content"]


def test_a_framing_that_invents_is_withheld_and_a_template_stands_in():
    provider = ScriptedProvider(
        [
            tool_reply("atlas_lookup", {"material": "TiO2", "quantity": "n_633nm"}),
            text_reply("draft"),
            text_reply("TiO2 has rutile values, and there is anatase data too, with n near 2.49."),
        ]
    )
    answer = ask(None, "n of TiO2 at 633 nm?", provider=provider)
    assert answer.composed is True
    assert "anatase data too" not in answer.answer and "2.49" not in answer.answer
    assert answer.model_text.startswith("The atlas record for 'TiO2' is Titanium dioxide (rutile / anatase) (id 42).")
    assert "[withheld framing — " in answer.suppressed_answer and "'anatase'" in answer.suppressed_answer
    #  The verbatim lines say the name mentions anatase and no dataset is anatase.
    assert any("its name mentions anatase, but no dataset" in line for line in answer.reference_values)


def test_an_answer_from_an_empty_atlas_lookup_is_withheld_and_says_what_was_searched():
    provider = ScriptedProvider(
        [
            tool_reply("atlas_lookup", {"material": "unobtainium", "quantity": "density"}),
            text_reply("The atlas has no unobtainium, but its density is about 7.2 g/cm3."),
        ]
    )
    answer = ask(None, "What is the density of unobtainium?", provider=provider)
    assert answer.answer.startswith(EMPTY_EVIDENCE_ANSWER)
    assert "7.2" not in answer.answer
    assert "atlas_lookup(material='unobtainium', quantity='density'): No material in materials-db v0.22.0" in answer.answer


def test_a_name_inside_a_phrase_resolves_and_says_so():
    result = lookup("liquid water", "n")
    [match] = result["matches"]
    assert match["material_id"] == 232 and "named inside" in match["matched_on"][0]
    assert "matched by a name it contains" in result["note"]


def test_summary_lines_carry_id_grade_meaning_and_source():
    [line] = lookup("fused silica", "n")["summary"]
    assert line.startswith("Silicon dioxide (fused silica) (id 38): n_633nm | structure:amorphous = 1.457012124643149")
    assert "grade C (one measured dataset, not cross-checked)" in line
    assert "doi:10.1364/JOSA.55.001205" in line and "reference value, materials-db v0.22.0" in line


def test_a_suspicious_class_is_spelled_out_as_disagreement():
    [k] = values(lookup("As2Se3", "k"), "k_633nm")
    assert "DISAGREE" in k["classification_meaning"]


def test_a_compound_name_is_not_resolved_to_the_element_it_contains():
    from cnms_fom.atlas import open_atlas
    from cnms_fom.atlas import tools as atlas_tools

    atlas = open_atlas(FIXTURE)
    try:
        #  "nitride" is not a qualifier, so "gallium nitride" is not resolved to anything it
        #  merely contains; "solid diamond" is diamond.
        assert atlas_tools._named_matches(atlas, "gallium nitride") == []
        assert [m["material_id"] for m in atlas_tools._named_matches(atlas, "solid diamond")] == [132]
    finally:
        atlas.close()


# --- numbers the tools did not return are named in the answer -------------------------------


def test_with_mixed_evidence_the_answer_stays_checked_with_the_values_appended(monkeypatch):
    """Atlas values plus a corpus passage: the model's answer connects them, so it stays."""
    from cnms_fom.rag_backend import tools as rag_tools
    from cnms_fom.rag_backend.providers import ToolSpec

    passage = {"passages": [{"citation": "ald.pdf", "page": 2, "text": "HfO2 films were annealed."}]}
    fake = rag_tools.Tool(spec=ToolSpec(name="search_corpus", description="x", input_schema={"type": "object"}), run=lambda session, **_: passage)
    monkeypatch.setitem(rag_tools.TOOLS, "search_corpus", fake)
    provider = ScriptedProvider(
        [
            tool_reply("atlas_lookup", {"material": "fused silica", "quantity": "n_633nm"}),
            tool_reply("search_corpus", {"query": "anneal"}),
            text_reply("Fused silica (id 38) has n 1.457, about 1.4440 at 1550 nm, grade A."),
        ]
    )
    answer = ask(None, "n of fused silica, and did the films anneal?", provider=provider)
    assert answer.composed is False
    #  1.457 rounds the stored 1.457012124643149; 1.4440 and 1550 (nm) were never returned,
    #  and the stored grade is C, not A.
    assert answer.unsupported_numbers == ["1.4440", "1550"]
    assert answer.unsupported_claims == ["grade A: no returned value has that grade"]
    assert "[Unsupported numbers: 1.4440" in answer.answer and "[Unsupported claims: grade A" in answer.answer
    assert answer.answer.rstrip().endswith(answer.reference_values[-1])


def test_supported_numbers_raise_no_flag():
    from cnms_fom.rag_backend.agent import AgentStep, unsupported_numbers

    step = AgentStep(step=1, tool="atlas_lookup", arguments={}, result={"value": 1.331790355084919, "year": 1973})
    assert unsupported_numbers("n = 1.33 (Hale 1973), 8 datasets, 633 nm", [step], "n at 633 nm?") == []
    #  Seen on llama3.1:8b, all supported: a page read out of a DOI, the release version,
    #  and the 633 that only appears inside identifiers.
    cited = AgentStep(
        step=1, tool="atlas_lookup", arguments={},
        result={"doi": "10.1364/JOSA.55.001205", "origin": "materials-db v0.22.0 (reference)", "quantity": "n_633nm"},
    )
    assert unsupported_numbers("J. Opt. Soc. Am. 55, 1205; atlas version 0.22.0; k at 633 nm", [cited]) == []
    assert unsupported_numbers("version 0.23.0", [cited]) == ["0.23.0"]


# --- the claim checks ----------------------------------------------------------------------


def _step(result, tool="atlas_lookup"):
    from cnms_fom.rag_backend.agent import AgentStep

    return AgentStep(step=1, tool=tool, arguments={}, result=result)


def test_claims_ids_grades_and_classes():
    from cnms_fom.rag_backend.answer_checks import unsupported_claims

    step = _step({"matches": [{"material_id": 38, "values": [{"grade": "C", "classification": "single_source", "label": "n_633nm"}]}]})
    assert unsupported_claims("Fused silica (id 38) is grade C.", [step]) == []
    assert unsupported_claims("Fused silica (id 39) is grade A.", [step]) == [
        "material id 39: no tool returned it",
        "grade A: no returned value has that grade",
    ]
    assert unsupported_claims("The sources agree excellently, an excellent result.", [step]) == [
        "'excellent': no returned consensus is classed excellent"
    ]


def test_claims_catch_agreement_where_the_release_says_suspicious():
    from cnms_fom.rag_backend.answer_checks import unsupported_claims

    step = _step({"matches": [{"material_id": 223, "values": [{"classification": "suspicious", "label": "k_633nm"}]}]})
    assert unsupported_claims("The sources agree on k.", [step]) == [
        "says the sources agree; the release classes them suspicious, i.e. they disagree"
    ]
    assert unsupported_claims("The sources disagree; they do not agree.", [step]) == []


def test_claims_phases_and_axes_need_a_dataset_unless_negated_or_a_name():
    from cnms_fom.rag_backend.answer_checks import unsupported_claims

    step = _step(
        {"matches": [{"material": "Titanium dioxide (rutile / anatase)", "key": "oxides_50:TiO2@rutile",
                      "values": [{"label": "n_633nm | rutile | e-ray", "scope": "rutile | e-ray"}]}]}
    )
    assert unsupported_claims("Rutile e-ray data are given.", [step]) == []
    assert unsupported_claims("There is anatase data and an o-ray value.", [step]) == [
        "'anatase': no returned dataset or value is labelled anatase",
        "'o-ray': no returned dataset or value is labelled o-ray",
    ]
    assert unsupported_claims("There is no anatase dataset.", [step]) == []
    assert unsupported_claims("Its name mentions anatase.", [step]) == []


def test_thinking_text_is_not_the_answer():
    from cnms_fom.rag_backend.providers import _THINK

    assert _THINK.sub("", "<think>long reasoning</think>\nThe answer.").strip() == "The answer."


def test_a_phase_in_a_name_but_in_no_dataset_is_called_out():
    [match] = lookup("TiO2", "n")["matches"]
    caveats = [line for line in lookup("TiO2", "n")["summary"] if "its name mentions" in line]
    assert caveats == [
        "Titanium dioxide (rutile / anatase) (id 42): its name mentions anatase, but no dataset or value in "
        "materials-db v0.22.0 is labelled anatase; a phase in a name is not data"
    ]
    assert match["material_id"] == 42


# --- what the second end-to-end round showed llama3.1:8b doing ------------------------------


def test_n_at_another_wavelength_says_what_is_stored_and_never_interpolates():
    result = lookup("fused silica", "n at 1550 nm")
    [match] = result["matches"]
    assert match["material_id"] == 38 and match["values"] == []
    assert result["summary"][0].startswith(
        "Silicon dioxide (fused silica) (id 38): no scalar n at 1550 nm is stored"
    )
    assert "spans 1550 nm" in result["summary"][1]
    assert lookup("fused silica", "n_633")["quantity"] == "n_633nm"
    assert lookup("fused silica", "k at 633 nm")["quantity"] == "k_633nm"


def test_a_phase_word_qualifies_a_name_and_is_answered_with_what_the_data_hold():
    result = lookup("anatase TiO2", "n")
    [match] = result["matches"]
    assert match["material_id"] == 42 and "named inside" in match["matched_on"][0]
    assert any("its name mentions anatase, but no dataset" in line for line in result["summary"])
    #  A chemical word is still not a qualifier.
    assert lookup("Indium nitride", "n")["matches"] == []


def test_an_empty_platform_lookup_points_to_the_atlas(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import cnms_fom.db.models  # noqa: F401
    from cnms_fom.db.base import Base

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'empty.db'}", future=True)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, future=True)() as session:
        result = run_tool(session, "lookup_property_values", {"formula": "C", "property_key": "density"})
    assert result["n_values"] == 0 and "call atlas_lookup" in result["note"]


def test_every_ollama_call_has_a_timeout():
    """A stuck request (seen with qwen3:14b: Ollama idle, connection open) must fail, not hang."""
    pytest.importorskip("langchain_ollama")
    from cnms_fom.rag_backend.providers import OllamaProvider

    client = OllamaProvider("llama3.1:8b")._client(0.0, None)
    assert client.client_kwargs == {"timeout": get_settings().ollama_timeout_seconds}
    assert get_settings().ollama_timeout_seconds == 300.0


def test_a_server_that_never_answers_fails_the_call_instead_of_hanging(monkeypatch):
    """The failure mode itself: a socket that accepts and then says nothing."""
    pytest.importorskip("langchain_ollama")
    import socket
    import threading
    import time

    from cnms_fom.rag_backend.providers import OllamaProvider

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    held: list = []
    threading.Thread(target=lambda: held.append(server.accept()), daemon=True).start()
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.getsockname()[1]}")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "1")
    get_settings.cache_clear()
    started = time.monotonic()
    with pytest.raises(Exception) as caught:
        OllamaProvider("llama3.1:8b").send("system", [{"role": "user", "content": "hello"}])
    assert time.monotonic() - started < 10
    assert "timed out" in str(caught.value).lower() or "timeout" in type(caught.value).__name__.lower()
    server.close()


def test_framings_that_quote_names_titles_and_lines_are_not_claims():
    """Two qwen3:14b framings the first version of the check withheld wrongly (2026-10-02)."""
    from cnms_fom.rag_backend.answer_checks import unsupported_claims

    step = _step(lookup("TiO2", "n_633nm"))
    copied = (
        "The refractive index (n) of TiO2 at 633 nm is reported for rutile. No dataset explicitly labeled "
        '"anatase" is present in the provided reference values.\n\n- ' + "\n- ".join(step.result["summary"])
    )
    assert unsupported_claims(copied, [step]) == []
    absent = "For rutile there are two values. Anatase's refractive index at this wavelength is absent in the given dataset."
    assert unsupported_claims(absent, [step]) == []
    #  The checks still catch the claim they exist for, and a negation does not leak across sentences.
    assert unsupported_claims("There is anatase data too.", [step]) == [
        "'anatase': no returned dataset or value is labelled anatase"
    ]
    assert unsupported_claims("The release has no brookite. Anatase data are given.", [step]) == [
        "'anatase': no returned dataset or value is labelled anatase"
    ]


# --- the third end-to-end round (llama3.1:8b, full suite) -------------------------------------


def test_a_lookup_with_nothing_stored_says_so_and_k_carries_its_status():
    summary = lookup("fused silica", "k")["summary"]
    assert summary == [
        "Silicon dioxide (fused silica) (id 38): no stored k_633nm in materials-db v0.22.0",
        "Silicon dioxide (fused silica) (id 38): k status = not_measured; a missing k is unknown, not 0 "
        "(materials-db v0.22.0)",
    ]


def test_coverage_gives_ready_to_quote_lines():
    result = run_tool(None, "atlas_coverage", {"material": "Water"})
    assert result["summary"][0].startswith("Water (id 232): ")
    assert "optical datasets" in result["summary"][0] and "k status partially_measured" in result["summary"][0]


def test_a_coverage_answer_is_composed_from_its_lines():
    provider = ScriptedProvider(
        [tool_reply("atlas_coverage", {"material": "Water"}), text_reply("draft"), text_reply("Water is well covered.")]
    )
    answer = ask(None, "What does the atlas hold for water?", provider=provider)
    assert answer.composed is True and answer.reference_values[0].startswith("Water (id 232): ")


def test_a_model_reported_gap_still_shows_the_releases_own_facts():
    provider = ScriptedProvider(
        [
            tool_reply("atlas_lookup", {"material": "unobtainium", "quantity": "density"}),
            tool_reply("atlas_lookup", {"material": "fused silica", "quantity": "k"}),
            text_reply("k is not stored. [DATA GAP: no k value]"),
        ]
    )
    answer = ask(None, "What is k of fused silica?", provider=provider)
    assert answer.insufficient_context is True and answer.composed is False
    assert "**Reference values**" in answer.answer
    assert "k status = not_measured; a missing k is unknown, not 0" in answer.answer
