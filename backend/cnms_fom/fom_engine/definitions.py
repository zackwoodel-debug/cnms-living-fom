"""Draft FOM definitions (FOM_PROOF Table 4).

    "The score weights w_aq are not physical measurements. They are
     application-policy choices and must be documented."
    "Example score components. Replace only after formal approval."

Everything in this module is therefore marked ``approved=False``.  Every score
computed from an unapproved definition carries a DRAFT note (see
``scores.score_material``), and the API refuses to publish a ranking from one.

TODO(FOM_PROOF): before any of these is used for a reportable ranking —
  1. Set the weights by the documented application-policy process, with a named
     approver, rather than the uniform placeholders below.
  2. Re-derive every (lo, hi) from the *frozen* eligible material set using
     ``normalization.bounds_from_population``.  The bounds below are screening
     ranges for oxide dielectrics, not the bounds of any particular study, and
     they are the single most likely thing to be wrong for your population.
  3. Record ``bounds_basis``, ``eligible_set_query``, and
     ``n_materials_in_bounds`` on the ``FomDefinition`` row, then freeze it.
"""

from __future__ import annotations

from cnms_fom.db.enums import Direction, Transform

from .normalization import NormalizationSpec
from .scores import FomSpec

#  DRAFT screening bounds, in transformed space (log10 where the transform says so).
#  Sourced from the usual range of oxide dielectric literature values so that the
#  scaffold runs end to end; NOT a frozen study population.
DRAFT_BOUNDS: dict[str, dict] = {
    #  k from SiO2 (3.9) to a soft-mode perovskite (~300).
    "k": {"lo": 0.591, "hi": 2.477, "direction": "benefit", "transform": "log10"},
    "Eg": {"lo": 1.0, "hi": 9.0, "direction": "benefit", "transform": "none"},
    "dEc": {"lo": 0.0, "hi": 3.5, "direction": "benefit", "transform": "none"},
    #  E_bd from 0.1 to 30 MV/cm.
    "Ebd": {"lo": -1.0, "hi": 1.477, "direction": "benefit", "transform": "log10"},
    #  tan delta from 1e-5 to 1e-1, detrimental — Eq. (25)'s log-scale form.
    "tan_delta": {"lo": -5.0, "hi": -1.0, "direction": "cost", "transform": "log10"},
    "kappa_th": {"lo": 0.5, "hi": 400.0, "direction": "benefit", "transform": "none"},
}

#  Table 4: which properties each application scores on.
APPLICATION_COMPONENTS: dict[str, tuple[str, ...]] = {
    "logic": ("k", "Eg", "dEc", "Ebd"),
    "power": ("k", "Eg", "dEc", "Ebd", "kappa_th"),
    "rf": ("k", "tan_delta", "Ebd", "kappa_th"),
}

APPLICATION_NOTES: dict[str, str] = {
    "logic": "Capacitance benefit balanced against bandgap, band alignment, and reliability.",
    "power": "Breakdown and thermal performance may dominate depending on the device.",
    "rf": "Permittivity is useful only if loss is low under matched RF conditions.",
}


def _normalization_for(components: tuple[str, ...], floor_eps: float) -> dict[str, NormalizationSpec]:
    specs: dict[str, NormalizationSpec] = {}
    for key in components:
        bounds = DRAFT_BOUNDS[key]
        specs[key] = NormalizationSpec(
            property_key=key,
            lo=float(bounds["lo"]),
            hi=float(bounds["hi"]),
            direction=Direction(bounds["direction"]),
            transform=Transform(bounds["transform"]),
            floor_eps=floor_eps,
        )
    return specs


def draft_fom(application: str, *, floor_eps: float = 1e-3, version: int = 1) -> FomSpec:
    """Build the uniform-weight draft FOM for one application.

    Uniform weights are a deliberate placeholder, not a recommendation: they
    make the absence of a real policy decision obvious rather than burying an
    arbitrary choice behind plausible-looking numbers.
    """
    try:
        components = APPLICATION_COMPONENTS[application]
    except KeyError:
        raise KeyError(
            f"Unknown application {application!r}; known: {sorted(APPLICATION_COMPONENTS)}."
        ) from None

    weight = 1.0 / len(components)
    return FomSpec(
        name=application,
        application=application,
        version=version,
        weights={key: weight for key in components},
        normalization=_normalization_for(components, floor_eps),
        description=(
            f"DRAFT {application} score (FOM_PROOF Table 4). {APPLICATION_NOTES[application]} "
            "Uniform placeholder weights; not approved for reporting."
        ),
        approved=False,
        floor_eps=floor_eps,
    )


def all_draft_foms(*, floor_eps: float = 1e-3) -> dict[str, FomSpec]:
    """Every draft application score, keyed by name."""
    return {app: draft_fom(app, floor_eps=floor_eps) for app in APPLICATION_COMPONENTS}


def to_definition_kwargs(spec: FomSpec) -> dict:
    """Flatten a :class:`FomSpec` into ``db.models.FomDefinition`` column values."""
    return {
        "name": spec.name,
        "version": spec.version,
        "application": spec.application,
        "description": spec.description,
        "weights": dict(spec.weights),
        "normalization": {k: v.as_dict() for k, v in spec.normalization.items()},
        "floor_eps": spec.floor_eps,
        "bounds_basis": (
            "DRAFT literature screening range for oxide dielectrics. "
            "Re-derive from the frozen eligible set before reporting (Sec. 5.3)."
        ),
        "approved": spec.approved,
        "frozen": False,
    }
