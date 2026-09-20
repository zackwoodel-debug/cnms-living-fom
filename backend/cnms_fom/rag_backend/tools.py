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

Every tool that touches an **analysis table is read-only**.  There is no tool
that writes a property value, edits a fit, or creates a material, and that is a
design constraint rather than an unimplemented feature: FOM_PROOF Sec. 15.2
forbids a model-mediated path into the analysis tables, and the way to forbid one
is not to build it.  Promotion of a fitted value into ``property_values`` lives in
``modalfit.promote``, runs from an explicit API call, and requires a material
identity a person supplied.

Two tools do write, and only to knowledge cards: ``write_card`` and
``link_cards``.  The exception is deliberate and narrow.  A card is explicitly a
*reading aid* — the assistant's synthesis, written down so it accumulates instead
of evaporating — and it is quarantined by construction: an assistant-written card
is ``PROPOSED``, ``citable`` is False until a named person has checked it against
resolved sources, and no code path leads from a card to a stored property value.
They are also opt-in: ``tool_specs`` and the agent withhold them unless
``allow_writes=True``, so the default surface is read-only.
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
    #  True for the two knowledge-card tools, and nothing else. A write tool is
    #  offered only when the caller opts in (``allow_writes``), and it can reach
    #  knowledge cards alone — never ``property_values``, ``fit_records``, or any
    #  analysis table. See the module docstring.
    writes: bool = False

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
# Bayesian optimization — reading a campaign, not driving it
# ---------------------------------------------------------------------------


def _lookup_bo_campaign(session, *, run_id: int | None = None, name: str | None = None) -> dict:
    """One campaign's configuration and where it currently stands."""
    from sqlalchemy import func

    from cnms_fom.db.models import BoObservation, BoRun, BoSuggestion, FomDefinition

    query = session.query(BoRun)
    if run_id is not None:
        query = query.filter(BoRun.id == int(run_id))
    elif name:
        query = query.filter(BoRun.name == name.strip())
    else:
        runs = session.query(BoRun).order_by(BoRun.id.desc()).limit(MAX_ROWS).all()
        return {
            "campaigns": [
                {"bo_run_id": r.id, "name": r.name, "status": r.status,
                 "acquisition": r.acquisition, "objective_sense": r.objective_sense}
                for r in runs
            ],
            "note": "Pass run_id or name for the full configuration and history.",
        }

    run = query.one_or_none()
    if run is None:
        return {"error": f"No BO campaign matching run_id={run_id!r} name={name!r}."}

    n_obs = session.query(func.count(BoObservation.id)).filter(
        BoObservation.bo_run_id == run.id
    ).scalar() or 0
    n_infeasible = session.query(func.count(BoObservation.id)).filter(
        BoObservation.bo_run_id == run.id, BoObservation.is_feasible.is_(False)
    ).scalar() or 0
    n_pending = session.query(func.count(BoSuggestion.id)).filter(
        BoSuggestion.bo_run_id == run.id, BoSuggestion.status == "proposed"
    ).scalar() or 0

    definition = (
        session.get(FomDefinition, run.fom_definition_id) if run.fom_definition_id else None
    )

    payload = {
        "bo_run_id": run.id,
        "name": run.name,
        "status": run.status,
        "acquisition": run.acquisition,
        "objective_sense": run.objective_sense,
        "random_seed": run.random_seed,
        "search_space": run.search_space,
        "constraints": run.constraints,
        "n_observations": int(n_obs),
        "n_infeasible": int(n_infeasible),
        "n_pending_suggestions": int(n_pending),
        "objective": (
            {
                "fom": f"{definition.name} v{definition.version}",
                "approved": definition.approved,
                "frozen": definition.frozen,
                "weights": definition.weights,
            }
            if definition
            else None
        ),
        "objective_scale_note": (
            "The surrogate models ln F, not F (Eq. 30). F is a weighted geometric mean of terms "
            "in (0, 1], so it is strongly skewed and its residuals are nothing like the Gaussian "
            "a GP assumes. ln is monotone, so the argmax is unchanged."
        ),
    }
    if definition is not None and not definition.approved:
        payload["objective_warning"] = (
            f"The FOM definition '{definition.name} v{definition.version}' is unapproved — its "
            "weights are uniform placeholders with no named owner. The campaign is optimising "
            "toward a policy choice nobody has made yet, so its ranking is not a result."
        )
    return payload


def _lookup_bo_history(session, *, run_id: int, limit: int = 30) -> dict:
    """The evaluated points, and whether the campaign is actually improving.

    Returns the best-so-far trajectory rather than just the observations, because
    "is this working?" is a question about the trend and not about any single
    point. A campaign whose best value has not moved in a dozen runs is telling
    you something — usually that the search space is wrong, not that the
    optimiser is.
    """
    from cnms_fom.db.models import BoObservation, BoRun

    run = session.get(BoRun, int(run_id))
    if run is None:
        return {"error": f"No BO campaign {run_id}."}

    observations = (
        session.query(BoObservation)
        .filter(BoObservation.bo_run_id == run.id)
        .order_by(BoObservation.id)
        .all()
    )
    if not observations:
        return {
            "bo_run_id": run.id,
            "observations": [],
            "note": "No observations yet. The first batch will be a space-filling Sobol design, "
            "not a model-driven proposal — there is nothing to fit a surrogate on.",
        }

    maximising = run.objective_sense == "max"
    best = None
    trajectory: list[dict] = []
    for index, observation in enumerate(observations, start=1):
        value = observation.objective_value
        if value is not None and observation.is_feasible:
            if best is None or (value > best if maximising else value < best):
                best = value
        trajectory.append({"n": index, "objective": value, "best_so_far": best,
                           "feasible": observation.is_feasible})

    #  How long since the best actually moved. The single most useful number for
    #  "should I keep going?", and nothing else in the schema exposes it.
    stalled_for = 0
    for entry in reversed(trajectory):
        if entry["best_so_far"] == best:
            stalled_for += 1
        else:
            break

    feasible_values = [
        o.objective_value for o in observations if o.is_feasible and o.objective_value is not None
    ]
    return {
        "bo_run_id": run.id,
        "objective_sense": run.objective_sense,
        "n_observations": len(observations),
        "n_infeasible": sum(1 for o in observations if not o.is_feasible),
        "best_objective": best,
        "best_recipe": next(
            (o.parameters for o in observations if o.objective_value == best and o.is_feasible),
            None,
        ),
        "evaluations_since_best_improved": stalled_for - 1 if stalled_for else 0,
        "objective_range": (
            {"min": min(feasible_values), "max": max(feasible_values)}
            if feasible_values
            else None
        ),
        "trajectory": trajectory[-int(limit):],
        "observations": [
            {"parameters": o.parameters, "objective": o.objective_value,
             "noise": o.objective_noise, "feasible": o.is_feasible}
            for o in observations[-int(limit):]
        ],
        "interpretation_note": (
            "A flat best-so-far is not automatically convergence. Check the pending suggestions' "
            "predicted_std first: if the surrogate is still uncertain and the acquisition keeps "
            "proposing the same corner of the space, the bounds are probably clipping the optimum "
            "rather than containing it. Infeasible observations also carry information — they "
            "bound the feasible region — and a campaign that is mostly infeasible has a "
            "constraint problem, not a search problem."
        ),
    }


def _lookup_bo_suggestions(session, *, run_id: int, include_completed: bool = False) -> dict:
    """Pending proposals with the acquisition value and prediction behind each.

    ``predicted_std`` is the part worth reading. A batch with large spread is the
    optimiser exploring; a batch with small spread and small acquisition values is
    one that thinks it is done.
    """
    from cnms_fom.db.models import BoRun, BoSuggestion

    run = session.get(BoRun, int(run_id))
    if run is None:
        return {"error": f"No BO campaign {run_id}."}

    query = session.query(BoSuggestion).filter(BoSuggestion.bo_run_id == run.id)
    if not include_completed:
        query = query.filter(BoSuggestion.status == "proposed")
    suggestions = query.order_by(BoSuggestion.id.desc()).limit(MAX_ROWS).all()

    stds = [s.predicted_std for s in suggestions if s.predicted_std is not None]
    return {
        "bo_run_id": run.id,
        "acquisition": run.acquisition,
        "n_suggestions": len(suggestions),
        "suggestions": [
            {
                "suggestion_id": s.id,
                "parameters": s.parameters,
                "acquisition_value": s.acquisition_value,
                "predicted_mean": s.predicted_mean,
                "predicted_std": s.predicted_std,
                "batch_index": s.batch_index,
                "status": s.status,
            }
            for s in suggestions
        ],
        "uncertainty_summary": (
            {"min_std": min(stds), "max_std": max(stds), "mean_std": sum(stds) / len(stds)}
            if stds
            else None
        ),
        "interpretation_note": (
            "predicted_mean is on the ln F scale, so it is negative for any F below 1 and a "
            "difference of 0.7 is a factor of two in F. A large predicted_std means the surrogate "
            "has not seen that region; a batch of small-std, low-acquisition proposals means it "
            "believes it has converged, which is worth checking against the search-space bounds "
            "before believing it."
        ),
    }


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------


def _check_physical_plausibility(
    session,  # noqa: ARG001 - no database access; kept for a uniform tool signature
    *,
    values: dict | None = None,
    formula: str | None = None,
    growth_technique: str | None = None,
    growth_per_cycle_ang: float | None = None,
) -> dict:
    """Check a set of values for physical possibility, in three graded tiers."""
    from cnms_fom.fom_engine.plausibility import check_values

    if not values:
        from cnms_fom.fom_engine.plausibility import HARD_BOUNDS

        return {
            "error": "No values given.",
            "usage": "Pass values as {registry_key: number}, e.g. "
            '{"k": 25.0, "Eg": 5.7, "dEc": 1.5}. Add formula for the composition-dependent '
            "checks (X-ray SLD against density).",
            "checkable_keys": sorted(HARD_BOUNDS),
        }

    numeric = {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    context = {}
    if growth_technique:
        context["growth_technique"] = growth_technique
    if growth_per_cycle_ang is not None:
        context["growth_per_cycle_ang"] = growth_per_cycle_ang

    report = check_values(numeric, formula=formula, context=context)
    payload = report.as_dict()
    payload["values_checked"] = numeric
    payload["formula"] = formula
    return payload


def _check_fit_plausibility(
    session, *, sample_id: str, layer_label: str | None = None
) -> dict:
    """Run the physical checks over every film layer of a sample's stored fits.

    The check that earns its keep here is X-ray SLD against mass density: they
    are related by the electron density, so for a fixed composition they are one
    measurement and not two. A co-refinement that left both free will often trade
    one against the other — they are nearly degenerate in XRR — and the result is
    a fit that reproduces the curve while disagreeing with itself.
    """
    from cnms_fom.db.models import FitRecord
    from cnms_fom.fom_engine.plausibility import check_fit_layer

    records = (
        session.query(FitRecord)
        .filter(FitRecord.sample_id == sample_id)
        .order_by(FitRecord.id.desc())
        .limit(MAX_ROWS)
        .all()
    )
    if not records:
        return {
            "sample_id": sample_id,
            "n_layers_checked": 0,
            "layers": [],
            "all_physical": None,
            "note": f"No ModalFit refinements stored for {sample_id!r}.",
        }

    wanted = layer_label.strip().lower() if layer_label else None
    out: list[dict] = []
    for record in records:
        for layer in record.layers:
            if layer.role != "layer":
                continue
            if wanted and (layer.label or "").strip().lower() != wanted:
                continue
            report = check_fit_layer(layer, techniques=record.techniques)
            out.append(
                {
                    "fit_record_id": record.id,
                    "techniques": record.techniques,
                    "layer": layer.label or layer.material,
                    "formula": layer.formula or layer.material,
                    **report.as_dict(),
                }
            )

    return {
        "sample_id": sample_id,
        "n_layers_checked": len(out),
        "layers": out,
        "all_physical": all(entry["physical"] for entry in out) if out else None,
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
    "lookup_bo_campaign": Tool(
        spec=ToolSpec(
            name="lookup_bo_campaign",
            description=(
                "Read a Bayesian-optimization campaign's configuration and current state: search "
                "space, instrument constraints, acquisition function, objective, and counts. Call "
                "with no arguments to list campaigns. Use this before interpreting any BO result "
                "— the search space and the objective definition are what a result means."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer", "description": "The bo_run id."},
                    "name": {"type": "string", "description": "The campaign name."},
                },
                "required": [],
            },
        ),
        run=_lookup_bo_campaign,
    ),
    "lookup_bo_history": Tool(
        spec=ToolSpec(
            name="lookup_bo_history",
            description=(
                "The evaluated points of a campaign, plus the best-so-far trajectory, how many "
                "evaluations since the best improved, and how many were infeasible. Use this for "
                "'is this campaign working', 'has it converged', or 'should I keep going'. The "
                "trend answers those; a single observation does not."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "limit": {"type": "integer", "description": "How many recent points (1-40)."},
                },
                "required": ["run_id"],
            },
        ),
        run=_lookup_bo_history,
    ),
    "lookup_bo_suggestions": Tool(
        spec=ToolSpec(
            name="lookup_bo_suggestions",
            description=(
                "Pending proposals for a campaign with the acquisition value, predicted mean, and "
                "predicted standard deviation behind each. Use this to judge whether the "
                "optimizer is exploring or exploiting, and to sanity-check a proposed recipe "
                "before anyone runs it on a tool."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "integer"},
                    "include_completed": {"type": "boolean"},
                },
                "required": ["run_id"],
            },
        ),
        run=_lookup_bo_suggestions,
    ),
    "check_physical_plausibility": Tool(
        spec=ToolSpec(
            name="check_physical_plausibility",
            description=(
                "Check whether a set of numbers is physically possible. Returns three graded "
                "tiers: violations (impossible — a permittivity below 1, a band offset larger "
                "than the gap), inconsistencies (two values that must agree and do not, such as "
                "X-ray SLD versus mass density), and heuristic flags (domain expectations with "
                "real exceptions, such as the k-Eg tradeoff). Use this on any number you are "
                "about to report or act on. A heuristic flag is never grounds to discard a value."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "values": {
                        "type": "object",
                        "description": "Registry keys to numbers, e.g. "
                        '{"k": 25.0, "Eg": 5.7, "dEc": 1.5, "rho": 9.1, "sld_xray": 64.6}.',
                        "additionalProperties": {"type": "number"},
                    },
                    "formula": {
                        "type": "string",
                        "description": "Composition, e.g. HfO2. Enables the SLD-versus-density "
                        "cross-check.",
                    },
                    "growth_technique": {
                        "type": "string",
                        "description": "e.g. ald, for the process heuristics.",
                    },
                    "growth_per_cycle_ang": {
                        "type": "number",
                        "description": "Growth per cycle in angstroms, for the ALD monolayer check.",
                    },
                },
                "required": ["values"],
            },
        ),
        run=_check_physical_plausibility,
    ),
    "check_fit_plausibility": Tool(
        spec=ToolSpec(
            name="check_fit_plausibility",
            description=(
                "Run the physical checks over every film layer of a sample's stored ModalFit "
                "refinements. Catches a fit that reproduces its data while disagreeing with "
                "itself — most often an X-ray SLD and a density that were both left free and "
                "traded against each other, since they are nearly degenerate in XRR. Use this "
                "before quoting any fitted number."
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
        run=_check_fit_plausibility,
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


def tool_specs(
    names: list[str] | None = None, *, allow_writes: bool = False
) -> list[ToolSpec]:
    """Tool schemas to offer the model, in a stable order.

    Stable because tool definitions are rendered ahead of everything else in a
    cached prompt prefix: reordering them on each request would invalidate the
    cache on every call.

    ``allow_writes`` adds the two knowledge-card writers. Withheld by default, so
    the assistant cannot create a card unless the caller asked for that — and even
    then a card is quarantined as PROPOSED and not citable.
    """
    selected = names or sorted(TOOLS)
    return [
        TOOLS[name].spec
        for name in selected
        if name in TOOLS and (allow_writes or not TOOLS[name].writes)
    ]


def write_tool_names() -> list[str]:
    """The tools that write. Two, both of them knowledge cards."""
    return sorted(name for name, tool in TOOLS.items() if tool.writes)


def run_tool(session, name: str, arguments: dict, *, allow_writes: bool = False) -> dict:
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
    if tool.writes and not allow_writes:
        #  Belt and braces: the schema was withheld, so a call here means the
        #  model invented the tool name. Refuse rather than execute.
        return {
            "error": f"{name} writes, and writes are not enabled for this turn.",
            "why": "Knowledge-card writing is opt-in. Nothing else the assistant can reach "
            "writes at all.",
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


# ---------------------------------------------------------------------------
# Knowledge cards
# ---------------------------------------------------------------------------


def _search_cards(
    session,
    *,
    query: str | None = None,
    card_type: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
    citable_only: bool = False,
    limit: int = 20,
) -> dict:
    """Find knowledge cards. Check `citable` before building on one."""
    from cnms_fom.knowledge.cards import search_cards

    try:
        return search_cards(
            session,
            query,
            card_type=card_type or None,
            status=status or None,
            tags=tags,
            citable_only=citable_only,
            limit=limit,
        )
    except ValueError as exc:
        from cnms_fom.db.enums import CardStatus, CardType

        return {
            "error": str(exc),
            "valid_card_types": [t.value for t in CardType],
            "valid_statuses": [s.value for s in CardStatus],
        }


def _read_card(session, *, slug: str) -> dict:
    """One card with its typed links in both directions."""
    from cnms_fom.knowledge.cards import read_card

    try:
        return read_card(session, slug)
    except ValueError as exc:
        return {"error": str(exc)}


def _card_graph(session, *, root_slug: str | None = None, depth: int = 2) -> dict:
    """The card graph as nodes and typed edges, optionally rooted at one card."""
    from cnms_fom.knowledge.cards import card_graph

    try:
        return card_graph(session, root_slug, depth=depth)
    except ValueError as exc:
        return {"error": str(exc)}


def _card_stats(session) -> dict:
    """Card-corpus health: counts, review backlog, contradictions, orphans."""
    from cnms_fom.knowledge.cards import card_stats

    return card_stats(session)


def _write_card(
    session,
    *,
    slug: str,
    title: str,
    body: str,
    card_type: str = "concept",
    summary: str | None = None,
    sources: list | None = None,
    tags: list[str] | None = None,
    confidence: float | None = None,
) -> dict:
    """Create or update a knowledge card. It lands unreviewed and not citable."""
    from cnms_fom.knowledge.cards import CardError, card_as_dict, upsert_card

    try:
        card = upsert_card(
            session,
            slug=slug,
            title=title,
            body=body,
            card_type=card_type,
            summary=summary,
            sources=sources,
            tags=tags,
            confidence=confidence,
            authored_by="assistant",
        )
    except (CardError, LookupError) as exc:
        return {"error": str(exc)}

    return {
        "written": True,
        **card_as_dict(card, include_body=False),
        "reminder": (
            "This card is PROPOSED and not citable. Do not present it back as an established "
            "result in this conversation — it is your own note, and a person has to check it "
            "against its sources first."
        ),
    }


def _link_cards(
    session, *, from_slug: str, to_slug: str, relation: str, note: str | None = None
) -> dict:
    """Link two cards with a typed edge. A 'contradicts' link requires a note."""
    from cnms_fom.db.enums import CardRelation
    from cnms_fom.knowledge.cards import CardError, link_cards

    try:
        link = link_cards(session, from_slug, to_slug, relation=relation, note=note)
    except (CardError, LookupError) as exc:
        return {"error": str(exc), "valid_relations": [r.value for r in CardRelation]}
    except ValueError as exc:
        return {"error": str(exc), "valid_relations": [r.value for r in CardRelation]}

    return {
        "linked": True,
        "from": from_slug,
        "to": to_slug,
        "relation": link.relation.value,
        "note": link.note,
    }


TOOLS.update(
    {
        "search_cards": Tool(
            spec=ToolSpec(
                name="search_cards",
                description=(
                    "Search the knowledge cards — accumulated concept pages, source summaries, "
                    "methods, findings, and open questions. Check these BEFORE searching the raw "
                    "corpus: a card is the integrated version of work already done, so starting "
                    "there means not re-deriving it. Always check each result's `citable` flag: a "
                    "proposed card is an unreviewed draft."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text to match in title, slug, summary, or body."},
                        "card_type": {
                            "type": "string",
                            "description": "concept, source, method, finding, or question.",
                        },
                        "status": {"type": "string", "description": "proposed, reviewed, or superseded."},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "citable_only": {
                            "type": "boolean",
                            "description": "True returns only reviewed, sourced, non-stale cards.",
                        },
                        "limit": {"type": "integer"},
                    },
                    "required": [],
                },
            ),
            run=_search_cards,
        ),
        "read_card": Tool(
            spec=ToolSpec(
                name="read_card",
                description=(
                    "Read one knowledge card in full, with its typed links in both directions. "
                    "The incoming links matter: an incoming 'contradicts' is a conflict the "
                    "card's own author never wrote down. `unresolved_contradictions` on the "
                    "response is worth reporting whenever it is non-empty."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"slug": {"type": "string", "description": "e.g. concepts/ald-window-hfo2."}},
                    "required": ["slug"],
                },
            ),
            run=_read_card,
        ),
        "card_graph": Tool(
            spec=ToolSpec(
                name="card_graph",
                description=(
                    "The knowledge graph as nodes and typed edges, optionally rooted at one card. "
                    "Use it to answer 'what does this rest on' and 'what would break if this is "
                    "wrong'. `orphans` names cards nothing links to — knowledge that failed to "
                    "connect to anything."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "root_slug": {"type": "string"},
                        "depth": {"type": "integer", "description": "Hops from the root (1-4)."},
                    },
                    "required": [],
                },
            ),
            run=_card_graph,
        ),
        "card_stats": Tool(
            spec=ToolSpec(
                name="card_stats",
                description=(
                    "Health of the card corpus: counts by type and status, the review backlog, "
                    "stale reviews, unresolved contradictions, and orphans. Use it to answer "
                    "'what do we actually know' and 'what needs attention'."
                ),
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            run=_card_stats,
        ),
        "write_card": Tool(
            spec=ToolSpec(
                name="write_card",
                description=(
                    "Create or update a knowledge card, so what you worked out is kept instead of "
                    "being re-derived next time. Write one when you have integrated something "
                    "across sources — a growth window, a mechanism, a contradiction between two "
                    "papers, an open question. Cite every factual claim in the body and list the "
                    "sources. The card lands PROPOSED and not citable: it is your note until a "
                    "person checks it, and you must not present it back as established."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "slug": {
                            "type": "string",
                            "description": "Path-like and lower case, e.g. concepts/ald-window-hfo2.",
                        },
                        "title": {"type": "string"},
                        "body": {
                            "type": "string",
                            "description": "Markdown. Cite each claim inline against the sources listed.",
                        },
                        "card_type": {
                            "type": "string",
                            "description": "concept, source, method, finding, or question.",
                        },
                        "summary": {"type": "string", "description": "One or two sentences."},
                        "sources": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": True},
                            "description": 'Citation records, e.g. [{"kind": "document", '
                            '"document_id": 3, "page": 12}, {"kind": "fit_record", '
                            '"fit_record_id": 7}].',
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "description": "0-1."},
                    },
                    "required": ["slug", "title", "body"],
                },
            ),
            run=_write_card,
            writes=True,
        ),
        "link_cards": Tool(
            spec=ToolSpec(
                name="link_cards",
                description=(
                    "Link two cards with a typed edge: fed_by, relates_to, depends_on, "
                    "contradicts, measured_by, or answers. A 'contradicts' link requires a note "
                    "saying which claims conflict and on what basis — recording that two cards "
                    "disagree while discarding what they disagree about helps nobody."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "from_slug": {"type": "string"},
                        "to_slug": {"type": "string"},
                        "relation": {
                            "type": "string",
                            "description": "fed_by, relates_to, depends_on, contradicts, "
                            "measured_by, or answers.",
                        },
                        "note": {"type": "string", "description": "Required for 'contradicts'."},
                    },
                    "required": ["from_slug", "to_slug", "relation"],
                },
            ),
            run=_link_cards,
            writes=True,
        ),
    }
)
