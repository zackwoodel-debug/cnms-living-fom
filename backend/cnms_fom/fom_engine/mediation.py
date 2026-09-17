"""Mediated structure-to-function calculation (FOM_PROOF Sec. 10).

This is the primary quantitative result of the whole protocol:

    B_jq   = dP_q / dS_j                                            (Eq. 47)
    Gamma_qa = d ln F_a / dP_q = w_aq * d ln z_q / dP_q             (Eq. 54)
    M_SF   = B_SP Gamma_PF                                          (Eq. 57)
    M_ja   = sum_q B_jq Gamma_qa = d ln F_a / dS_j                  (Eqs. 55-56)

and, crucially, the per-channel decomposition

    Contribution_jqa = B_jq Gamma_qa                                (Eq. 65)

which is what distinguishes

    soft phonon -> higher k -> higher capacitance

from

    soft phonon -> higher k -> lower breakdown margin -> lower Power score.

B can be supplied two ways: estimated empirically from a regression, or taken
from the simplified oscillator model (Eqs. 50-53).  The source is recorded on
the result, because they support different claims — an empirical B is evidence
about this material population; a theoretical B is a physical prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .normalization import dlnz_dproperty
from .scores import FomSpec

# ---------------------------------------------------------------------------
# Theoretical elasticities — FOM_PROOF Eqs. (49)-(53)
# ---------------------------------------------------------------------------

#  eps_ionic ~ (Z*)^2 / (V mu omega_TO^2)  =>  d ln eps_ionic / d ln X
THEORY_ELASTICITIES: dict[str, dict[str, float]] = {
    "eps_ionic": {
        "Z_RMS_star": +2.0,   # Eq. (50)
        "omega_TO_min": -2.0,  # Eq. (51)
        "V_fu": -1.0,          # Eq. (52)
        "mu_eff": -1.0,        # Eq. (53)
    }
}


@dataclass
class MediationResultSet:
    """M and its decomposition for one application."""

    application: str
    fom_name: str
    fom_version: int
    descriptors: list[str]
    properties: list[str]
    #  M_ja = d ln F_a / dS_j, in natural units of S_j.
    mediated_effect: dict[str, float]
    #  S_j * M_ja — dimensionless, comparable across descriptors with different units.
    mediated_elasticity: dict[str, float]
    #  {descriptor: {property: B_jq * Gamma_qa}}  (Eq. 65)
    contributions: dict[str, dict[str, float]]
    dominant_property: dict[str, str]
    sensitivity_source: str
    reference_point: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "application": self.application,
            "fom_name": self.fom_name,
            "fom_version": self.fom_version,
            "descriptors": self.descriptors,
            "properties": self.properties,
            "mediated_effect": self.mediated_effect,
            "mediated_elasticity": self.mediated_elasticity,
            "contributions": self.contributions,
            "dominant_property": self.dominant_property,
            "sensitivity_source": self.sensitivity_source,
            "reference_point": self.reference_point,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# B: structure -> property sensitivity (Sec. 10.1)
# ---------------------------------------------------------------------------


def sensitivity_from_elasticities(
    elasticities: dict[str, dict[str, float]],
    reference_properties: dict[str, float],
    reference_descriptors: dict[str, float],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert log-log elasticities (Eq. 48) into natural-unit B (Eq. 47).

        d ln P / d ln S = (S / P) dP/dS   =>   B_jq = elasticity * P_q / S_j

    The conversion is only valid at the reference point, so the caller must
    supply one and it is stored on the result.  Use the population median of the
    eligible set, not a single material.

    ``elasticities`` is ``{property_key: {descriptor_key: d ln P / d ln S}}``.
    """
    properties = sorted(elasticities)
    descriptors = sorted({d for row in elasticities.values() for d in row})

    b = np.zeros((len(descriptors), len(properties)), dtype=float)
    for qi, q in enumerate(properties):
        p_ref = reference_properties.get(q)
        if p_ref is None or p_ref <= 0:
            raise ValueError(
                f"Reference value for property {q!r} must be positive to convert an elasticity; "
                f"got {p_ref!r}."
            )
        for ji, s in enumerate(descriptors):
            elasticity = elasticities[q].get(s)
            if elasticity is None:
                continue
            s_ref = reference_descriptors.get(s)
            if s_ref is None or s_ref == 0:
                raise ValueError(
                    f"Reference value for descriptor {s!r} must be non-zero; got {s_ref!r}."
                )
            b[ji, qi] = elasticity * p_ref / s_ref
    return b, descriptors, properties


def sensitivity_from_regressions(
    fits: dict[str, object],
    *,
    log_log: bool = True,
    reference_properties: dict[str, float] | None = None,
    reference_descriptors: dict[str, float] | None = None,
    descriptor_keys: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build B from fitted regressions, one per property (Sec. 9.1 -> Sec. 10.1).

    ``fits`` maps property key to a :class:`~cnms_fom.fom_engine.regression.RegressionFit`.
    When ``log_log`` is true the coefficients are elasticities (Eq. 48) and are
    converted at the supplied reference point; otherwise they are already
    dP/dS and are used directly.

    Coefficients from a fit that carried a collinearity warning are propagated
    unchanged — the warning travels with the fit, and suppressing the number
    here would hide the problem rather than fix it.
    """
    properties = sorted(fits)
    if descriptor_keys is None:
        descriptor_keys = sorted(
            {
                name.removeprefix("log_")
                for fit in fits.values()
                for name in getattr(fit, "coefficients", {})
            }
        )

    if log_log:
        elasticities = {
            q: {
                name.removeprefix("log_"): float(value)
                for name, value in fits[q].coefficients.items()
                if name.removeprefix("log_") in descriptor_keys
            }
            for q in properties
        }
        if reference_properties is None or reference_descriptors is None:
            raise ValueError(
                "Converting log-log coefficients to dP/dS requires a reference point "
                "(median of the eligible set)."
            )
        return sensitivity_from_elasticities(
            elasticities, reference_properties, reference_descriptors
        )

    b = np.zeros((len(descriptor_keys), len(properties)), dtype=float)
    for qi, q in enumerate(properties):
        for ji, s in enumerate(descriptor_keys):
            b[ji, qi] = float(fits[q].coefficients.get(s, 0.0))
    return b, list(descriptor_keys), properties


# ---------------------------------------------------------------------------
# Gamma: property -> function sensitivity (Sec. 10.2)
# ---------------------------------------------------------------------------


def gamma_matrix(
    spec: FomSpec, reference_properties: dict[str, float], properties: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Eq. (54): Gamma_qa = w_aq * d ln z_q / dP_q, evaluated at a reference point.

    Returns a column vector (one application) plus the property ordering.  The
    derivative is analytic — see ``normalization.dlnz_dproperty`` — so it follows
    the declared transform and direction correction exactly rather than
    approximating them numerically.
    """
    properties = properties or sorted(spec.weights)
    gamma = np.zeros((len(properties), 1), dtype=float)

    for qi, q in enumerate(properties):
        weight = spec.weights.get(q, 0.0)
        if weight == 0.0:
            continue
        p_ref = reference_properties.get(q)
        if p_ref is None:
            raise ValueError(
                f"Gamma needs a reference value for weighted property {q!r}; none supplied."
            )
        gamma[qi, 0] = weight * dlnz_dproperty(float(p_ref), spec.normalization[q])
    return gamma, properties


# ---------------------------------------------------------------------------
# M = B Gamma (Sec. 10.3)
# ---------------------------------------------------------------------------


def mediated_effect(
    b: np.ndarray,
    descriptors: list[str],
    properties: list[str],
    spec: FomSpec,
    reference_properties: dict[str, float],
    reference_descriptors: dict[str, float] | None = None,
    *,
    sensitivity_source: str = "regression",
) -> MediationResultSet:
    """Eq. (57): M_SF = B_SP Gamma_PF, with the Eq. (65) channel decomposition.

    ``b`` is (n_descriptors x n_properties) and must be ordered to match
    ``descriptors`` and ``properties``.
    """
    gamma, gamma_properties = gamma_matrix(spec, reference_properties, properties)
    if gamma_properties != list(properties):
        raise ValueError("Property ordering of B and Gamma disagree.")
    if b.shape != (len(descriptors), len(properties)):
        raise ValueError(
            f"B has shape {b.shape}; expected {(len(descriptors), len(properties))}."
        )

    m = b @ gamma  # (n_descriptors x 1)

    effects: dict[str, float] = {}
    elasticities: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    dominant: dict[str, str] = {}

    for ji, descriptor in enumerate(descriptors):
        effects[descriptor] = float(m[ji, 0])
        channel = {q: float(b[ji, qi] * gamma[qi, 0]) for qi, q in enumerate(properties)}
        contributions[descriptor] = channel
        dominant[descriptor] = max(channel, key=lambda q: abs(channel[q])) if channel else ""
        if reference_descriptors and reference_descriptors.get(descriptor):
            # d ln F / d ln S — dimensionless, so descriptors in different units compare.
            elasticities[descriptor] = float(
                m[ji, 0] * reference_descriptors[descriptor]
            )

    notes = [
        "M_ja = d ln F_a / dS_j (Eq. 55). It is a local derivative at the stated reference "
        "point, not a global ranking.",
        "Sec. 10.3: B describes estimated physical structure-property relationships; Gamma "
        "describes how the chosen application definition rewards each property. A sign change "
        "between applications comes from Gamma, not from new physics.",
    ]
    if sensitivity_source == "theory":
        notes.append(
            "B came from the simplified oscillator model (Eqs. 49-53). This is a physical prior, "
            "not evidence from this material population."
        )
    if not spec.approved:
        notes.append(
            f"FOM {spec.name!r} v{spec.version} is unapproved; Gamma inherits draft weights."
        )

    return MediationResultSet(
        application=spec.application,
        fom_name=spec.name,
        fom_version=spec.version,
        descriptors=list(descriptors),
        properties=list(properties),
        mediated_effect=effects,
        mediated_elasticity=elasticities,
        contributions=contributions,
        dominant_property=dominant,
        sensitivity_source=sensitivity_source,
        reference_point={
            **{f"P:{k}": v for k, v in reference_properties.items()},
            **{f"S:{k}": v for k, v in (reference_descriptors or {}).items()},
        },
        notes=notes,
    )


def propagate_ionic_elasticities_to_k(
    ionic_elasticities: dict[str, float], eps_ionic: float, k: float
) -> dict[str, float]:
    """Carry the Eqs. (50)-(53) elasticities from eps_ionic across to k, via Eq. (11).

    The oscillator model predicts the *lattice* response, but application scores
    are written against the measured dielectric constant k = eps_static.  Eq. (11)
    splits that as

        k = eps_inf + eps_ionic

    and the polar-mode mechanism moves only the second term, so at fixed eps_inf

        dk/dS = d eps_ionic / dS

    which in elasticity form is

        d ln k / d ln S = (d ln eps_ionic / d ln S) * (eps_ionic / k).

    The eps_ionic/k factor is the fraction of the dielectric response the
    mechanism can actually reach.  It matters: for a material whose permittivity
    is mostly electronic, a large lattice elasticity still moves k very little,
    and dropping the factor would overstate the structural lever by that ratio.

    Holding eps_inf fixed is an approximation — a structural change that alters
    the volume also shifts the electronic polarizability.  It is stated here
    rather than buried, and it is why the result is a prior, not evidence.
    """
    if eps_ionic <= 0 or k <= 0:
        raise ValueError(
            f"eps_ionic ({eps_ionic!r}) and k ({k!r}) must be positive to propagate elasticities."
        )
    if eps_ionic > k:
        raise ValueError(
            f"eps_ionic ({eps_ionic}) exceeds k ({k}); Eq. (11) requires eps_ionic <= eps_static. "
            "Check that both values come from the same material and context."
        )
    fraction = eps_ionic / k
    return {s: value * fraction for s, value in ionic_elasticities.items()}


def mediated_effect_from_theory(
    spec: FomSpec,
    reference_properties: dict[str, float],
    reference_descriptors: dict[str, float],
    *,
    elasticities: dict[str, dict[str, float]] | None = None,
    propagate_through_eq11: bool = True,
) -> MediationResultSet:
    """Convenience path: B from Eqs. (50)-(53), Gamma from the FOM definition.

    Useful before any data exists — it answers "which structural knob should
    move this application score, if the oscillator picture holds?", and gives the
    BO loop a physically motivated prior direction.

    When the FOM weights ``k`` but the elasticities are stated for ``eps_ionic``
    (which is the usual case, since Sec. 4.1 derives the mechanism for the
    lattice term), ``propagate_through_eq11`` bridges the two via
    :func:`propagate_ionic_elasticities_to_k`.  That needs both ``eps_ionic`` and
    ``k`` in ``reference_properties``.

    The oscillator model says nothing about Eg, dEc, Ebd, or kappa_th, so those
    channels are *omitted* rather than entered as zeros.  A zero would assert
    "this descriptor does not affect breakdown", which the model does not claim.
    The resulting M is therefore a partial effect over the covered channels only,
    and the uncovered ones are listed in ``notes``.
    """
    elasticities = dict(elasticities or THEORY_ELASTICITIES)
    notes: list[str] = []

    if (
        propagate_through_eq11
        and spec.weights.get("k", 0.0) > 0.0
        and "k" not in elasticities
        and "eps_ionic" in elasticities
    ):
        eps_ionic = reference_properties.get("eps_ionic")
        k_ref = reference_properties.get("k")
        if eps_ionic is None or k_ref is None:
            raise ValueError(
                "Propagating the oscillator elasticities to k needs both 'eps_ionic' and 'k' in "
                "reference_properties (Eq. 11). Supply them, or pass propagate_through_eq11=False."
            )
        elasticities["k"] = propagate_ionic_elasticities_to_k(
            elasticities["eps_ionic"], float(eps_ionic), float(k_ref)
        )
        notes.append(
            f"k elasticities derived from eps_ionic via Eq. (11) at eps_ionic/k = "
            f"{float(eps_ionic) / float(k_ref):.3f}, holding eps_inf fixed."
        )

    b, descriptors, properties = sensitivity_from_elasticities(
        elasticities, reference_properties, reference_descriptors
    )
    keep = [qi for qi, q in enumerate(properties) if spec.weights.get(q, 0.0) > 0.0]
    if not keep:
        raise ValueError(
            f"None of the elasticity target properties {properties} carry weight in FOM "
            f"{spec.name!r}. Supply elasticities for {sorted(spec.required_properties)}."
        )

    covered = {properties[qi] for qi in keep}
    uncovered = sorted(set(spec.required_properties) - covered)
    if uncovered:
        notes.append(
            f"No theoretical channel for {uncovered}; these are omitted, not set to zero. "
            "M is a partial effect over the covered channels only — estimate the missing ones "
            "empirically (Sec. 9.1) before treating it as the full mediated effect."
        )

    result = mediated_effect(
        b[:, keep],
        descriptors,
        [properties[qi] for qi in keep],
        spec,
        reference_properties,
        reference_descriptors,
        sensitivity_source="theory",
    )
    result.notes = [*notes, *result.notes]
    return result
