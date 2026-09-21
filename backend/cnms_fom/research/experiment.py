"""Summarising what an experiment actually showed.

The input is assembled deterministically from records — the recipe, the BO
prediction that preceded it, the ModalFit fits, their plausibility, the FOM result
and its status, the cross-technique comparison, whether the point was feasible.
The model's only job is to explain it, and every sentence it writes is labelled
evidence, interpretation, or proposal.

That labelling is the point of the module.  A summary mixes three things that read
alike and must not be treated alike: what was recorded, what somebody thinks it
means, and what to do next.  People quote summaries onward, and the boundary they
lose first is the one between the second and the first.

What this does not do
---------------------
It does not update the campaign.  It does not write a property value.  It does not
mark anything measured.  A summary can *propose* an experiment-summary card, which
lands ``proposed`` and non-citable like any other, and it can *propose* campaign
context, which goes through the review gate in :mod:`bo_context`.  Neither happens
as a side effect of summarising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cnms_fom.db.enums import StatementKind
from cnms_fom.research.contracts import EvidenceItem, LabelledStatement

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """\
You are writing the summary of one completed experiment for a materials scientist. \
Everything you need is in front of you: the recipe, what the optimizer predicted \
before it ran, what was measured, how the fits behaved, whether the values are \
physically consistent, and the resulting figure of merit.

Produce JSON only:

{
  "statements": [{"kind": "evidence" | "interpretation" | "proposal", "text": "..."}],
  "open_questions": ["..."],
  "next_question": "..."
}

Label every statement:
  evidence        restates a record or a warning above. Name the record.
  interpretation  your reading of it. Say what it rests on.
  proposal        something to do next. Say what it assumes.

Rules:
1. Introduce no number that is not above. Not a typical value, not a converted one, \
not one you recall.
2. Compare the prediction with the result, and say which way it missed. The \
surrogate works on ln F, so a difference of 0.7 is a factor of two — do not \
describe it as "0.7 lower".
3. A modeled or simulated result is not a measurement. If the tier says modeled, \
every conclusion from it is illustrative and you must say so.
4. Repeat every plausibility violation and inconsistency. A fit that reproduces its \
data while disagreeing with itself is the most important thing on the page.
5. Where two techniques disagree, report both and say what differs. Never average \
them.
6. A parameter held fixed or clamped on its bound is not a measurement of anything. \
Say so wherever such a value appears.
7. If the experiment does not answer what it was run to answer, say that plainly \
and propose the shortest thing that would.
8. Prefer one discriminating next experiment over a list of thorough ones."""


@dataclass
class ExperimentOutcome:
    """The audited, deterministic input to a summary. Assembled from records only."""

    experiment_id: int | None = None
    bo_run_id: int | None = None
    sample_id: str | None = None
    recipe: dict | None = None

    #  What the optimizer expected, from the suggestion that produced this run.
    predicted_mean: float | None = None
    predicted_std: float | None = None
    acquisition_value: float | None = None

    #  What came back.
    objective_value: float | None = None
    is_feasible: bool | None = None
    #  "measured" or "modeled" — the tier of the property values behind the score.
    #  A simulated result makes every conclusion from it illustrative (Sec. 2.3).
    result_tier: str | None = None
    fom_value: float | None = None
    fom_status: str | None = None
    fom_definition: str | None = None
    fom_missing_inputs: list | None = None

    fits: list[dict] = field(default_factory=list)
    fit_warnings: list[str] = field(default_factory=list)
    plausibility: dict | None = None
    cross_technique: dict | None = None

    #  Deterministic warnings, attached whether or not the model mentions them.
    warnings: list[str] = field(default_factory=list)

    @property
    def prediction_error(self) -> float | None:
        if self.predicted_mean is None or self.objective_value is None:
            return None
        return self.objective_value - self.predicted_mean

    @property
    def prediction_was_within_uncertainty(self) -> bool | None:
        error = self.prediction_error
        if error is None or self.predicted_std is None:
            return None
        return abs(error) <= 2.0 * self.predicted_std

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "bo_run_id": self.bo_run_id,
            "sample_id": self.sample_id,
            "recipe": self.recipe,
            "prediction": {
                "predicted_mean_ln_f": self.predicted_mean,
                "predicted_std": self.predicted_std,
                "acquisition_value": self.acquisition_value,
            },
            "result": {
                "objective_value_ln_f": self.objective_value,
                "is_feasible": self.is_feasible,
                "tier": self.result_tier,
                "prediction_error": self.prediction_error,
                "within_2_sigma": self.prediction_was_within_uncertainty,
            },
            "fom": {
                "definition": self.fom_definition,
                "value": self.fom_value,
                "status": self.fom_status,
                "missing_inputs": self.fom_missing_inputs,
            },
            "fits": self.fits,
            "fit_warnings": self.fit_warnings,
            "plausibility": self.plausibility,
            "cross_technique": self.cross_technique,
            "warnings": self.warnings,
            "scale_note": (
                "Objective values are ln F (Eq. 30). A difference of 0.7 is a factor of two in F."
            ),
        }


@dataclass
class ExperimentSummary:
    """The outcome, plus a labelled narrative over it."""

    outcome: ExperimentOutcome
    statements: list[LabelledStatement] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_question: str | None = None
    proposed_card_slug: str | None = None
    model: str = ""
    provider: str = ""

    @property
    def evidence_statements(self) -> list[LabelledStatement]:
        return [s for s in self.statements if s.kind is StatementKind.EVIDENCE]

    @property
    def proposals(self) -> list[LabelledStatement]:
        return [s for s in self.statements if s.kind is StatementKind.PROPOSAL]

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome.as_dict(),
            "statements": [s.as_dict() for s in self.statements],
            "open_questions": self.open_questions,
            "next_question": self.next_question,
            "proposed_card_slug": self.proposed_card_slug,
            "model": self.model,
            "provider": self.provider,
            "disclaimer": (
                "This summary updated nothing. No observation, property value, FOM score, or "
                "campaign constraint was changed by producing it. A proposed card is unreviewed "
                "and non-citable; proposed campaign context goes through "
                "/research/campaigns/{run_id}/context and requires a named reviewer."
            ),
        }


def collect_outcome(
    db,
    *,
    experiment_id: int | None = None,
    bo_run_id: int | None = None,
    sample_id: str | None = None,
    observation_id: int | None = None,
) -> ExperimentOutcome:
    """Assemble the deterministic half. No model, no writes."""
    from cnms_fom.db.models import (
        BoObservation,
        BoSuggestion,
        Experiment,
        FomDefinition,
        FomScore,
    )
    from cnms_fom.rag_backend.tools import run_tool

    outcome = ExperimentOutcome(
        experiment_id=experiment_id, bo_run_id=bo_run_id, sample_id=sample_id
    )

    experiment = db.get(Experiment, experiment_id) if experiment_id else None
    if experiment is not None:
        outcome.recipe = experiment.recipe
        outcome.sample_id = sample_id or experiment.sample_id

    observation = None
    if observation_id:
        observation = db.get(BoObservation, observation_id)
    elif experiment_id:
        observation = (
            db.query(BoObservation)
            .filter(BoObservation.experiment_id == experiment_id)
            .order_by(BoObservation.id.desc())
            .first()
        )

    if observation is not None:
        outcome.bo_run_id = outcome.bo_run_id or observation.bo_run_id
        outcome.recipe = outcome.recipe or observation.parameters
        outcome.objective_value = observation.objective_value
        outcome.is_feasible = observation.is_feasible

        #  The suggestion that produced this recipe, matched on parameters. There is
        #  no foreign key between them, so this is a best effort and says so when it
        #  fails rather than reporting a prediction that belongs to another point.
        if observation.parameters:
            match = (
                db.query(BoSuggestion)
                .filter(BoSuggestion.bo_run_id == observation.bo_run_id)
                .order_by(BoSuggestion.id.desc())
                .all()
            )
            for suggestion in match:
                if suggestion.parameters == observation.parameters:
                    outcome.predicted_mean = suggestion.predicted_mean
                    outcome.predicted_std = suggestion.predicted_std
                    outcome.acquisition_value = suggestion.acquisition_value
                    break
            else:
                outcome.warnings.append(
                    "No pending or completed suggestion matches this recipe exactly, so the "
                    "optimizer's prediction for it could not be recovered. The result stands; the "
                    "prediction-versus-outcome comparison does not."
                )

        if observation.fom_score_id:
            score = db.get(FomScore, observation.fom_score_id)
            if score is not None:
                outcome.fom_value = score.value
                outcome.fom_status = score.status.value
                outcome.fom_missing_inputs = score.missing_inputs
                definition = db.get(FomDefinition, score.fom_definition_id)
                if definition is not None:
                    outcome.fom_definition = f"{definition.name} v{definition.version}"
                    if not definition.approved:
                        outcome.warnings.append(
                            f"The score was computed against the UNAPPROVED definition "
                            f"'{outcome.fom_definition}', whose weights are uniform placeholders "
                            "with no named owner. This number is not a result (Sec. 6.2)."
                        )
                if score.uses_modeled_inputs:
                    outcome.result_tier = "modeled"
                    outcome.warnings.append(
                        "This score rests on MODELED inputs, so it is ILLUSTRATIVE and must not "
                        "be compared with a measurement-based score (Sec. 2.3)."
                    )
                elif score.status.value == "scored":
                    outcome.result_tier = "measured"
                if score.status.value == "not_scored":
                    outcome.warnings.append(
                        "The material came back NOT SCORED: its input set is incomplete "
                        f"({score.missing_inputs}). That is a missing measurement, not a low score."
                    )

    if outcome.is_feasible is False:
        outcome.warnings.append(
            "This point was infeasible. That is information — it bounds the feasible region — but "
            "it is not a failed measurement and should not be read as one."
        )

    #  The fits, their plausibility, and the cross-technique comparison, through the
    #  assistant's own tools so a summary sees what the chat path sees.
    if outcome.sample_id:
        fits = run_tool(db, "list_sample_fits", {"sample_id": outcome.sample_id})
        outcome.fits = fits.get("fits") or []

        from cnms_fom.modalfit.compare import fit_process_warnings, fits_for_sample

        for record in fits_for_sample(db, outcome.sample_id):
            outcome.fit_warnings.extend(
                f"Fit #{record.id}: {finding}" for finding in fit_process_warnings(record)
            )

        outcome.plausibility = run_tool(
            db, "check_fit_plausibility", {"sample_id": outcome.sample_id}
        )
        for layer in outcome.plausibility.get("layers") or []:
            for finding in layer.get("violations") or []:
                outcome.warnings.append(
                    f"PHYSICALLY IMPOSSIBLE value in fit #{layer.get('fit_record_id')} "
                    f"({layer.get('layer')}): {finding.get('message')}"
                )
            for finding in layer.get("inconsistencies") or []:
                outcome.warnings.append(
                    f"Fit #{layer.get('fit_record_id')} ({layer.get('layer')}) disagrees with "
                    f"itself: {finding.get('message')}"
                )

        outcome.cross_technique = run_tool(
            db, "fit_disagreements", {"sample_id": outcome.sample_id}
        )
        for parameter in outcome.cross_technique.get("disagreements") or []:
            outcome.warnings.append(
                f"Techniques disagree on {parameter}. Both determinations are kept; nothing was "
                "averaged (Sec. 2.1)."
            )

    if outcome.prediction_was_within_uncertainty is False:
        outcome.warnings.append(
            f"The result missed the optimizer's prediction by "
            f"{outcome.prediction_error:+.3g} in ln F, more than two standard deviations "
            f"({outcome.predicted_std:.3g}). Either the surrogate is miscalibrated in this region "
            "or something about this run differed from the others."
        )

    return outcome


def summarise(db, outcome: ExperimentOutcome, *, provider=None) -> ExperimentSummary:
    """Add a labelled narrative to an outcome. Writes nothing."""
    import json

    from cnms_fom.research.extract import _extract_json

    summary = ExperimentSummary(
        outcome=outcome,
        model=getattr(provider, "model", "") or "",
        provider=getattr(provider, "name", "") or "",
    )
    if provider is None:
        return summary

    try:
        result = provider.send(
            SUMMARY_SYSTEM_PROMPT,
            [{"role": "user", "content": json.dumps(outcome.as_dict(), indent=2, default=str)}],
        )
    except Exception as exc:  # noqa: BLE001
        outcome.warnings.append(
            f"The narrative could not be generated ({exc}). The outcome above is unaffected — it "
            "was assembled from records, not written."
        )
        return summary

    if getattr(result, "refused", False):
        outcome.warnings.append(
            "The model declined to write the narrative. The outcome above is unaffected."
        )
        return summary

    parsed = _extract_json(result.text)
    if parsed is None:
        outcome.warnings.append(
            "The narrative reply was not parseable, so none was recorded. The outcome above is "
            "unaffected."
        )
        return summary

    #  An EVIDENCE statement must carry its warrant. In a brief that warrant is a
    #  retrieved passage; here it is the outcome record itself, which is what the
    #  model was shown. Built once and shared, so a reader of any evidence
    #  statement can get back to the records behind it.
    warrant = _outcome_as_evidence(outcome)

    for raw in parsed.get("statements") or []:
        if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
            continue
        try:
            kind = StatementKind(str(raw.get("kind", "interpretation")).strip().lower())
        except ValueError:
            kind = StatementKind.INTERPRETATION
        summary.statements.append(
            LabelledStatement(
                kind=kind,
                text=str(raw["text"]).strip(),
                evidence=[warrant] if kind is StatementKind.EVIDENCE else [],
            )
        )

    summary.open_questions = [
        str(q).strip() for q in (parsed.get("open_questions") or []) if str(q).strip()
    ]
    next_question = str(parsed.get("next_question") or "").strip()
    summary.next_question = next_question or None
    return summary


def _outcome_as_evidence(outcome: ExperimentOutcome) -> EvidenceItem:
    """The outcome itself, as a citable record reference.

    Not a document and not a page — this is instrument and optimizer data, so its
    locator is the record id. Kept distinguishable from a literature citation on
    purpose: ``experiment:12`` is not ``Kim 2024, p. 7``, and a reader must be able
    to tell which kind of thing a sentence rests on.
    """
    identifier = (
        f"experiment:{outcome.experiment_id}"
        if outcome.experiment_id is not None
        else f"sample:{outcome.sample_id}"
        if outcome.sample_id
        else f"bo_run:{outcome.bo_run_id}"
    )
    parts = []
    if outcome.recipe:
        parts.append(f"recipe {outcome.recipe}")
    if outcome.objective_value is not None:
        parts.append(f"objective (ln F) {outcome.objective_value:.4g}")
    if outcome.predicted_mean is not None:
        parts.append(f"predicted {outcome.predicted_mean:.4g}")
    if outcome.fom_status:
        parts.append(f"FOM {outcome.fom_status}")
    if outcome.result_tier:
        parts.append(f"tier {outcome.result_tier}")

    return EvidenceItem(
        document_id=None,
        document_title=identifier,
        page=None,
        quote="; ".join(parts) or "outcome record with no populated fields",
        retrieval_method="record",
        grade=3,
        grade_reason="platform record, not a literature citation",
    )


def propose_summary_card(db, summary: ExperimentSummary, *, slug: str | None = None) -> str:
    """Record the summary as a PROPOSED experiment-summary card.

    Explicitly called, never a side effect of summarising. The card lands
    unreviewed and non-citable, and its body carries the statement labels so a
    reviewer can see which sentences were evidence and which were somebody's reading
    of it.
    """
    from cnms_fom.db.enums import CardCategory
    from cnms_fom.knowledge.cards import upsert_card

    outcome = summary.outcome
    identifier = outcome.sample_id or f"experiment-{outcome.experiment_id}"
    slug = slug or f"findings/{str(identifier).strip().lower().replace('_', '-')}"

    lines = [f"# Experiment summary: {identifier}", ""]
    if outcome.recipe:
        lines += ["## Recipe", f"`{outcome.recipe}`", ""]
    lines += ["## Result"]
    if outcome.objective_value is not None:
        lines.append(f"- Objective (ln F): {outcome.objective_value:.4g}")
    if outcome.predicted_mean is not None:
        lines.append(
            f"- Optimizer predicted {outcome.predicted_mean:.4g}"
            + (f" ± {outcome.predicted_std:.3g}" if outcome.predicted_std else "")
        )
    if outcome.fom_status:
        lines.append(f"- FOM status: {outcome.fom_status} ({outcome.fom_definition})")
    if outcome.result_tier:
        lines.append(f"- Result tier: {outcome.result_tier}")
    lines.append("")

    for kind, heading in (
        (StatementKind.EVIDENCE, "## Evidence"),
        (StatementKind.INTERPRETATION, "## Interpretation"),
        (StatementKind.PROPOSAL, "## Proposals"),
    ):
        matching = [s for s in summary.statements if s.kind is kind]
        if matching:
            lines.append(heading)
            lines += [f"- {s.text}" for s in matching]
            lines.append("")

    if outcome.warnings:
        lines += ["## Caveats that qualify every number above"]
        lines += [f"- {w}" for w in outcome.warnings]
        lines.append("")
    if summary.open_questions:
        lines += ["## Open questions", *(f"- {q}" for q in summary.open_questions), ""]
    if summary.next_question:
        lines += ["## Proposed next question", summary.next_question, ""]

    sources: list[dict] = []
    for fit in outcome.fits:
        if fit.get("fit_record_id"):
            sources.append({"kind": "fit_record", "fit_record_id": fit["fit_record_id"]})
    if outcome.experiment_id:
        sources.append({"kind": "experiment", "experiment_id": outcome.experiment_id})

    card = upsert_card(
        db,
        slug=slug,
        title=f"Experiment summary: {identifier}",
        body="\n".join(lines),
        card_type="finding",
        category=CardCategory.EXPERIMENT_SUMMARY,
        summary=(summary.next_question or "Summary of one completed experiment."),
        sources=sources,
        authored_by="assistant",
    )
    summary.proposed_card_slug = card.slug
    logger.info("Proposed experiment-summary card %s (unreviewed)", card.slug)
    return card.slug
