"""The research assistant's tool surface.

Two kinds of evidence exist in this platform and they need different retrieval.

*Documents* — papers, process notes, facility guides — are unstructured, and the
right way to find a passage in them is hybrid semantic plus lexical search.

*Records* — ModalFit refinements, property values, FOM scores — are structured,
and the right way to find them is SQL.  A fitted thickness of 103.4 Å must never
be located by cosine similarity: an embedding of "103.4" is close to an embedding
of "130.4", the retrieval would be silently wrong, and a number is exactly the
kind of thing a model will report without hedging.  So every stored number
reaches the model through a typed query with its provenance attached, and
retrieval-by-similarity is reserved for prose.

Every tool here is **read-only**.  There is no tool that writes a property value,
edits a fit, or creates a material, and that is a design constraint rather than
an unimplemented feature: FOM_PROOF Sec. 15.2 forbids a model-mediated path into
the analysis tables, and the way to forbid it is not to build one.  Promotion of
a fitted value into ``property_values`` lives in ``modalfit.promote``, runs from
an explicit API call, and requires a material identity a person supplied.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from cnms_fom.db.enums import SynthesisTechnique

from .providers import ToolSpec

logger = logging.getLogger(__name__)

#  A hard cap on rows any tool returns. Not for performance — for context: a
#  tool that dumps 500 property rows pushes the retrieval instructions out of
#  the window, and the first thing a model forgets is the rule about not
#  inventing numbers.
MAX_ROWS = 40


@dataclass
class Tool:
    """A tool the assistant can call: its schema and its implementation."""

    spec: ToolSpec
    run: Callable[..., dict]

    @property
    def name(self) -> str:
        return self.spec.name


# ---------------------------------------------------------------------------
# Corpus search
# ---------------------------------------------------------------------------


def _search_corpus(session, *, query: str, techniques: list[str] | None = None, k: int = 6) -> dict:
    """Hybrid search over the document corpus, graded for relevance."""
    from .grading import retrieve_with_correction

    parsed: list[SynthesisTechnique] = []
    unknown: list[str] = []
    for name in techniques or []:
        try:
            parsed.append(SynthesisTechnique(str(name).strip().lower()))
        except ValueError:
            unknown.append(str(name))

    outcome = retrieve_with_correction(
        session, query, k=min(int(k), 12), techniques=parsed or None, grade=True
    )
    payload = {
        "query": query,
        "effective_query": outcome.effective_query,
        "query_was_rewritten": outcome.rewritten,
        "sufficient_evidence": outcome.sufficient,
        "passages": [
            {
                "citation": hit.fused.hit.citation(),
                "technique": hit.fused.hit.technique,
                "page": hit.fused.hit.page,
                "doi": hit.fused.hit.doi,
                "relevance_grade": hit.grade,
                "text": hit.fused.hit.text.strip(),
            }
            for hit in outcome.useful
        ],
    }
    if unknown:
        payload["ignored_technique_filters"] = unknown
        payload["valid_techniques"] = [t.value for t in SynthesisTechnique]
    if not outcome.sufficient:
        payload["note"] = (
            "No passage graded useful for this question. The corpus does not answer it — say so "
            "rather than answering from background knowledge."
        )
    return payload


def _corpus_coverage(session) -> dict:
    """What is actually indexed, by technique."""
    from .vectorstore import corpus_stats

    stats = corpus_stats(session)
    stats["note"] = (
        "A technique with zero documents cannot be answered from. Absence here is the reason for "
        "a data gap, not a reason to substitute general knowledge."
    )
    return stats


# ---------------------------------------------------------------------------
# ModalFit records
# ---------------------------------------------------------------------------


def _list_sample_fits(session, *, sample_id: str) -> dict:
    """Every stored ModalFit refinement for one sample, described in full."""
    from cnms_fom.modalfit.compare import describe_fit, fits_for_sample

    records = fits_for_sample(session, sample_id)
    if not records:
        return {
            "sample_id": sample_id,
            "fits": [],
            "note": (
                f"No ModalFit refinements are stored for sample {sample_id!r}. Either none has "
                "been imported, or the sample id differs from the one ModalFit exported."
            ),
        }
    return {
        "sample_id": sample_id,
        "n_fits": len(records),
        "fits": [
            {
                "fit_record_id": record.id,
                "techniques": record.techniques,
                "algorithm": record.algorithm,
                "chi2_total": record.chi2_total,
                "description": describe_fit(record),
            }
            for record in records[:MAX_ROWS]
        ],
    }


def _compare_fit_techniques(
    session, *, sample_id: str, parameter: str = "thickness", layer_label: str | None = None
) -> dict:
    """Compare one parameter across the techniques that determined it."""
    from cnms_fom.modalfit.compare import compare_parameter

    return compare_parameter(
        session, sample_id, parameter=parameter, layer_label=layer_label
    )


def _fit_disagreements(session, *, sample_id: str, layer_label: str | None = None) -> dict:
    """Every cross-technique disagreement on one sample."""
    from cnms_fom.modalfit.compare import cross_technique_report

    return cross_technique_report(session, sample_id, layer_label=layer_label)


def _list_samples_with_fits(session, *, limit: int = 20) -> dict:
    """Samples that have stored refinements — the answer to "what do we have?"."""
    from sqlalchemy import func

    from cnms_fom.db.models import FitRecord

    rows = (
        session.query(
            FitRecord.sample_id,
            func.count(FitRecord.id),
            func.max(FitRecord.fitted_at),
        )
        .filter(FitRecord.sample_id.isnot(None))
        .group_by(FitRecord.sample_id)
        .order_by(func.max(FitRecord.fitted_at).desc().nullslast())
        .limit(min(int(limit), MAX_ROWS))
        .all()
    )
    return {
        "samples": [
            {"sample_id": sample_id, "n_fits": int(count), "most_recent_fit": str(latest) if latest else None}
            for sample_id, count, latest in rows
        ],
        "n_samples": len(rows),
    }


# ---------------------------------------------------------------------------
# Property values, descriptors, scores
# ---------------------------------------------------------------------------


def _lookup_property_values(
    session,
    *,
    formula: str | None = None,
    property_key: str | None = None,
    limit: int = 20,
) -> dict:
    """Stored property values with their full measurement context.

    The context comes back with every row because a property without it is not a
    value.  FOM_PROOF Table 1 is explicit, and two dielectric constants at
    different frequencies are two measurements — presenting either one as "the"
    permittivity of the material is the error this shape prevents.
    """
    from cnms_fom.db.models import Material, PropertyValue

    query = session.query(PropertyValue, Material).join(
        Material, PropertyValue.material_id == Material.id
    )
    if formula:
        needle = formula.strip()
        query = query.filter(
            (Material.formula_reduced.ilike(needle)) | (Material.formula.ilike(needle))
        )
    if property_key:
        query = query.filter(PropertyValue.property_key == property_key.strip())

    rows = query.order_by(PropertyValue.property_key, PropertyValue.id).limit(
        min(int(limit), MAX_ROWS)
    ).all()

    return {
        "filters": {"formula": formula, "property_key": property_key},
        "n_values": len(rows),
        "values": [
            {
                "material": f"{material.formula_reduced} / {material.polymorph} / "
                f"{material.specimen_form.value}",
                "property_key": value.property_key,
                "value": value.value,
                "units": value.units,
                "uncertainty": value.uncertainty,
                "provenance_tier": value.provenance_tier.value,
                "context": {
                    key: getattr(value, key)
                    for key in (
                        "temperature_k",
                        "frequency_hz",
                        "thickness_nm",
                        "substrate",
                        "electrode",
                        "interface",
                        "tensor_component",
                        "method",
                        "software",
                        "xc_functional",
                    )
                    if getattr(value, key) is not None
                },
                "source": {
                    key: getattr(value, key)
                    for key in ("doi", "source_url", "database_identifier", "source_locator")
                    if getattr(value, key) is not None
                },
            }
            for value, material in rows
        ],
        "note": (
            "Each row is one measurement in one context. Values under different contexts are not "
            "alternatives to be averaged — report them separately with their conditions."
        ),
    }


def _descriptor_dictionary(session, *, key: str | None = None) -> dict:  # noqa: ARG001
    """The descriptor and property dictionary: real keys, units, and caveats.

    Here so the assistant uses the platform's vocabulary instead of inventing a
    plausible-looking key. A question about "the band gap" should resolve to
    ``Eg`` with its declared units and transform, not to free text.
    """
    from cnms_fom.descriptors.registry import PHYSICAL_PROPERTIES, STRUCTURAL_DESCRIPTORS

    if key:
        needle = key.strip()
        spec = STRUCTURAL_DESCRIPTORS.get(needle) or PHYSICAL_PROPERTIES.get(needle)
        if spec is None:
            return {
                "key": needle,
                "found": False,
                "available_descriptors": sorted(STRUCTURAL_DESCRIPTORS),
                "available_properties": sorted(PHYSICAL_PROPERTIES),
            }
        return {"key": needle, "found": True, "spec": spec.as_dict()}

    return {
        "descriptors": {k: v.units for k, v in STRUCTURAL_DESCRIPTORS.items()},
        "properties": {k: v.units for k, v in PHYSICAL_PROPERTIES.items()},
        "note": "Pass a key to get its full definition, formula, transform, and caveats.",
    }


def _lookup_fom_scores(
    session, *, fom_name: str | None = None, limit: int = 20
) -> dict:
    """FOM scores, with the status and definition version that make them readable."""
    from cnms_fom.db.models import FomDefinition, FomScore, Material

    query = (
        session.query(FomScore, Material, FomDefinition)
        .join(Material, FomScore.material_id == Material.id)
        .join(FomDefinition, FomScore.fom_definition_id == FomDefinition.id)
    )
    if fom_name:
        query = query.filter(FomDefinition.name == fom_name.strip())

    rows = query.order_by(FomScore.value.desc().nullslast()).limit(
        min(int(limit), MAX_ROWS)
    ).all()

    return {
        "filters": {"fom_name": fom_name},
        "n_scores": len(rows),
        "scores": [
            {
                "material": f"{material.formula_reduced} / {material.polymorph}",
                "fom": f"{definition.name} v{definition.version}",
                "approved": definition.approved,
                "value": score.value,
                "status": score.status.value,
                "missing_inputs": score.missing_inputs,
                "uses_modeled_inputs": score.uses_modeled_inputs,
            }
            for score, material, definition in rows
        ],
        "note": (
            "A not_scored material has an incomplete input set, not a low score. An illustrative "
            "score rests on modeled inputs and must not be compared against measured ones. An "
            "unapproved definition is a draft: its weights have no named owner yet."
        ),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, Tool] = {
    "search_corpus": Tool(
        spec=ToolSpec(
            name="search_corpus",
            description=(
                "Search the synthesis document corpus (MBE, PLD, ALD, sputtering, CVD process "
                "notes and CNMS user guides) for passages relevant to a question. Returns graded, "
                "citable excerpts. Use this for anything the literature or facility documentation "
                "would state. Returns sufficient_evidence=false when the corpus does not answer "
                "the question."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, in methods-section vocabulary.",
                    },
                    "techniques": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to corpus partitions: mbe, pld, ald, sputtering, "
                        "cvd, solution, cnms_user_doc, other. Omit to search everything.",
                    },
                    "k": {"type": "integer", "description": "How many passages to return (1-12)."},
                },
                "required": ["query"],
            },
        ),
        run=_search_corpus,
    ),
    "corpus_coverage": Tool(
        spec=ToolSpec(
            name="corpus_coverage",
            description=(
                "Report how many documents and chunks are indexed, by technique. Use this to "
                "explain *why* a question could not be answered — an empty partition is a "
                "coverage gap, not a reason to answer from general knowledge."
            ),
            input_schema={"type": "object", "properties": {}},
        ),
        run=_corpus_coverage,
    ),
    "list_samples_with_fits": Tool(
        spec=ToolSpec(
            name="list_samples_with_fits",
            description=(
                "List samples that have stored ModalFit refinements, newest first. Use this when "
                "the user refers to a sample without naming it exactly, or asks what fitted data "
                "exists."
            ),
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Max samples (1-40)."}},
            },
        ),
        run=_list_samples_with_fits,
    ),
    "list_sample_fits": Tool(
        spec=ToolSpec(
            name="list_sample_fits",
            description=(
                "Get every stored ModalFit refinement for one sample: the fitted stack, which "
                "techniques were co-refined (SE/SPR/QCM/XRR/NR), per-technique chi-squared, which "
                "parameters were varied, and the caveats that qualify each number. Use this "
                "whenever the question is about a specific sample's measured structure."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string", "description": "The ModalFit sample id."}
                },
                "required": ["sample_id"],
            },
        ),
        run=_list_sample_fits,
    ),
    "compare_fit_techniques": Tool(
        spec=ToolSpec(
            name="compare_fit_techniques",
            description=(
                "Compare one fitted parameter (thickness, roughness, density, sld_real, sld_imag, "
                "n, k) across every technique that determined it for one sample. Returns each "
                "determination separately with its caveats, plus the spread and a verdict — never "
                "an average. Use this for 'do XRR and SE agree', 'how thick is it really', or any "
                "question where one number came from more than one measurement."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "parameter": {
                        "type": "string",
                        "description": "thickness, roughness, density, sld_real, sld_imag, n, or k.",
                    },
                    "layer_label": {
                        "type": "string",
                        "description": "Which layer, when the stack has more than one film.",
                    },
                },
                "required": ["sample_id"],
            },
        ),
        run=_compare_fit_techniques,
    ),
    "fit_disagreements": Tool(
        spec=ToolSpec(
            name="fit_disagreements",
            description=(
                "Run the cross-technique comparison over every comparable parameter for one "
                "sample and report which ones disagree. Use this for 'is this fit trustworthy' or "
                "'what is wrong with this sample'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "layer_label": {"type": "string"},
                },
                "required": ["sample_id"],
            },
        ),
        run=_fit_disagreements,
    ),
    "lookup_property_values": Tool(
        spec=ToolSpec(
            name="lookup_property_values",
            description=(
                "Look up stored property values for a material, each with its full measurement "
                "context (temperature, frequency, thickness, substrate, method) and source (DOI, "
                "locator). Use this for any question about a material property this platform "
                "holds. Values under different contexts are separate measurements."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "formula": {"type": "string", "description": "e.g. HfO2. Case-insensitive."},
                    "property_key": {
                        "type": "string",
                        "description": "A registry key such as k, Eg, Ebd, sld_xray. Call "
                        "descriptor_dictionary if unsure.",
                    },
                    "limit": {"type": "integer"},
                },
            },
        ),
        run=_lookup_property_values,
    ),
    "descriptor_dictionary": Tool(
        spec=ToolSpec(
            name="descriptor_dictionary",
            description=(
                "The platform's descriptor and property dictionary: valid keys, symbols, units, "
                "declared transforms, and caveats. Call this before using a property key you are "
                "not certain of, rather than guessing one."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "A single key to define, e.g. Eg."}
                },
            },
        ),
        run=_descriptor_dictionary,
    ),
    "lookup_fom_scores": Tool(
        spec=ToolSpec(
            name="lookup_fom_scores",
            description=(
                "Look up figure-of-merit scores with their status and definition version. Use for "
                "'which material ranks best for X'. Respect the status: not_scored means missing "
                "inputs, and illustrative scores rest on modeled inputs."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "fom_name": {"type": "string", "description": "e.g. logic, power, rf."},
                    "limit": {"type": "integer"},
                },
            },
        ),
        run=_lookup_fom_scores,
    ),
}


def tool_specs(names: list[str] | None = None) -> list[ToolSpec]:
    """Tool schemas to offer the model, in a stable order.

    Stable because tool definitions are rendered ahead of everything else in a
    cached prompt prefix: reordering them on each request would invalidate the
    cache on every call.
    """
    selected = names or sorted(TOOLS)
    return [TOOLS[name].spec for name in selected if name in TOOLS]


def run_tool(session, name: str, arguments: dict) -> dict:
    """Execute one tool call.

    A bad tool name or bad arguments comes back as an error payload rather than
    an exception: the model can read the error, correct itself, and try again,
    where an exception would end the turn and lose the conversation.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return {
            "error": f"No tool named {name!r}.",
            "available_tools": sorted(TOOLS),
        }
    try:
        return tool.run(session, **(arguments or {}))
    except TypeError as exc:
        return {
            "error": f"Bad arguments for {name}: {exc}",
            "expected_schema": tool.spec.input_schema,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
        logger.warning("Tool %s failed: %s", name, exc, exc_info=True)
        return {"error": f"{name} failed: {exc}"}
