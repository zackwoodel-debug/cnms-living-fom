"""Is this number physically possible?

Every other module here asks whether a value is *eligible* — does it have its
context, does it match the analysis row.  This one asks whether it is *physical*.
They are different questions, and a value can pass the first and fail the second:
a relative permittivity of 0.4 with a complete measurement context is a
well-documented impossibility.

Three tiers, kept apart on purpose, because conflating them is how a heuristic
gets reported as a law:

``violation``
    Breaks an identity or a bound that cannot be broken. A permittivity below 1
    would mean the material polarizes against the field; a band offset larger
    than the band gap would put the conduction band below the valence band. If
    one of these fires, a number is wrong — not surprising, wrong.

``inconsistency``
    Two values that must agree, and do not, to within a stated tolerance.
    X-ray SLD and mass density are related by the electron density; a fit that
    reports both and disagrees with itself has a problem in one of them.

``heuristic``
    A domain expectation with real counterexamples. The k-Eg tradeoff, the 1 eV
    band-offset rule for a silicon gate dielectric, one monolayer per ALD cycle.
    Worth a second look, never worth overriding data. Each one carries the reason
    it exists and what would legitimately break it.

Nothing here rejects, edits, or excludes anything. It reports. FOM_PROOF Sec. 2.3
governs what may be *removed* from an analysis, and "a heuristic did not like it"
is not on that list — a surprising value that survives scrutiny is the most
interesting kind of result there is.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

#  Classical electron radius, in angstroms.  Sets the X-ray scattering length.
ELECTRON_RADIUS_ANG = 2.8179403262e-5
#  Avogadro's number scaled to angstroms: atoms per Å^3 per (g/cm^3) per (g/mol).
_AVOGADRO_PER_ANG3 = 0.602214076

#  Atomic number and mass for the elements this platform's oxides and electrodes
#  are made of.  Deliberately not a full periodic table: a short, checkable list
#  beats a dependency, and an element that is missing produces "cannot check"
#  rather than a wrong answer.
_ATOMIC: dict[str, tuple[int, float]] = {
    "H": (1, 1.008), "Li": (3, 6.94), "Be": (4, 9.012), "B": (5, 10.81),
    "C": (6, 12.011), "N": (7, 14.007), "O": (8, 15.999), "F": (9, 18.998),
    "Na": (11, 22.990), "Mg": (12, 24.305), "Al": (13, 26.982), "Si": (14, 28.085),
    "P": (15, 30.974), "S": (16, 32.06), "Cl": (17, 35.45), "K": (19, 39.098),
    "Ca": (20, 40.078), "Sc": (21, 44.956), "Ti": (22, 47.867), "V": (23, 50.942),
    "Cr": (24, 51.996), "Mn": (25, 54.938), "Fe": (26, 55.845), "Co": (27, 58.933),
    "Ni": (28, 58.693), "Cu": (29, 63.546), "Zn": (30, 65.38), "Ga": (31, 69.723),
    "Ge": (32, 72.63), "As": (33, 74.922), "Se": (34, 78.971), "Sr": (38, 87.62),
    "Y": (39, 88.906), "Zr": (40, 91.224), "Nb": (41, 92.906), "Mo": (42, 95.95),
    "Ru": (44, 101.07), "Rh": (45, 102.906), "Pd": (46, 106.42), "Ag": (47, 107.868),
    "Cd": (48, 112.414), "In": (49, 114.818), "Sn": (50, 118.71), "Sb": (51, 121.760),
    "Te": (52, 127.60), "Ba": (56, 137.327), "La": (57, 138.905), "Ce": (58, 140.116),
    "Nd": (60, 144.242), "Sm": (62, 150.36), "Eu": (63, 151.964), "Gd": (64, 157.25),
    "Dy": (66, 162.500), "Er": (68, 167.259), "Yb": (70, 173.045), "Lu": (71, 174.967),
    "Hf": (72, 178.486), "Ta": (73, 180.948), "W": (74, 183.84), "Re": (75, 186.207),
    "Os": (76, 190.23), "Ir": (77, 192.217), "Pt": (78, 195.084), "Au": (79, 196.967),
    "Hg": (80, 200.592), "Tl": (81, 204.38), "Pb": (82, 207.2), "Bi": (83, 208.980),
}

#  Hard bounds per registry key: (low, high, why).  ``None`` means unbounded on
#  that side.  These are impossibilities, not typical ranges — a value outside one
#  is a data-entry or unit error, and the message says which unit was expected
#  because that is the usual cause.
HARD_BOUNDS: dict[str, tuple[float | None, float | None, str]] = {
    "k": (1.0, None, "Relative permittivity below 1 would mean the material polarizes "
                     "against the applied field. Below 1 usually means an absolute "
                     "permittivity was entered instead of a relative one."),
    "eps_inf": (1.0, None, "Optical permittivity below 1 is unphysical for the same reason. "
                           "eps_inf = n^2 - k^2 in a transparent window, and n >= 1."),
    "eps_ionic": (0.0, None, "The ionic contribution is a sum of oscillator strengths; it "
                             "cannot be negative."),
    "Eg": (0.0, 15.0, "A band gap is non-negative by definition, and nothing above ~14 eV "
                      "(LiF) is known. A value above 15 eV is usually meV mistaken for eV."),
    "dEc": (None, None, "A conduction-band offset may be negative — that is a type-II "
                        "alignment, and a real finding. It is checked against Eg instead."),
    "Ebd": (0.0, 60.0, "Breakdown fields above ~60 MV/cm exceed the intrinsic limit of any "
                       "known dielectric. A value near 10^6 means V/cm was entered as MV/cm."),
    "tan_delta": (0.0, None, "A loss tangent is a ratio of positive quantities."),
    "kappa_th": (0.0, 3000.0, "Thermal conductivity is positive, and diamond at ~2200 "
                              "W/(m K) is the practical ceiling."),
    "rho": (0.05, 25.0, "Densities run from solid hydrogen (~0.09) to osmium (22.6) "
                        "g/cm^3. Outside that, check whether kg/m^3 was entered."),
    "sld_xray": (0.0, 200.0, "The real part of an X-ray SLD is positive — it counts "
                             "electrons. Units here are 1e-6 A^-2."),
    "sld_xray_imag": (0.0, None, "X-ray absorption is non-negative."),
    "sld_neutron": (-10.0, 20.0, "Neutron SLD may be negative — hydrogen is about -3.7 — "
                                 "but nothing reaches +20 or -10 in 1e-6 A^-2."),
    "sld_neutron_imag": (0.0, None, "Neutron absorption is non-negative."),
    "alpha_th": (-50.0, 200.0, "Thermal expansion in 1e-6/K. Negative is real (ZrW2O8), "
                               "but not below about -30."),
    "V_fu": (0.0, None, "A volume per formula unit is positive."),
    "omega_TO_min": (0.0, None, "A phonon frequency is positive. A soft mode driven to zero "
                                "is the ferroelectric limit, not a negative number."),
    "thickness_nm": (0.0, None, "A film thickness is positive."),
    "temperature_k": (0.0, None, "Absolute temperature is positive."),
}


@dataclass
class Finding:
    """One thing worth saying about a set of values."""

    #  "violation" | "inconsistency" | "heuristic"
    tier: str
    keys: list[str]
    message: str
    #  What would make this finding go away, or legitimately explain it.
    resolution: str = ""

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "keys": self.keys,
            "message": self.message,
            "resolution": self.resolution,
        }


@dataclass
class PlausibilityReport:
    """Findings across all three tiers, plus what could not be checked."""

    findings: list[Finding] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    not_checked: dict[str, str] = field(default_factory=dict)

    @property
    def violations(self) -> list[Finding]:
        return [f for f in self.findings if f.tier == "violation"]

    @property
    def inconsistencies(self) -> list[Finding]:
        return [f for f in self.findings if f.tier == "inconsistency"]

    @property
    def heuristics(self) -> list[Finding]:
        return [f for f in self.findings if f.tier == "heuristic"]

    @property
    def physical(self) -> bool:
        """False only when something *cannot* be true. Heuristics do not vote."""
        return not self.violations and not self.inconsistencies

    def as_dict(self) -> dict:
        return {
            "physical": self.physical,
            "n_violations": len(self.violations),
            "n_inconsistencies": len(self.inconsistencies),
            "n_heuristic_flags": len(self.heuristics),
            "violations": [f.as_dict() for f in self.violations],
            "inconsistencies": [f.as_dict() for f in self.inconsistencies],
            "heuristics": [f.as_dict() for f in self.heuristics],
            "checked": self.checked,
            "not_checked": self.not_checked,
            "note": (
                "A heuristic flag is a prompt to look again, never grounds to exclude a value. "
                "FOM_PROOF Sec. 2.3 governs exclusion, and 'a heuristic disliked it' is not on "
                "that list — a surprising value that survives scrutiny is the most interesting "
                "result there is."
            ),
        }


def parse_formula(formula: str) -> dict[str, float] | None:
    """Parse ``HfO2``, ``Hf0.5Zr0.5O2``, ``SrTiO3`` into element counts.

    Returns None when anything is unrecognised, rather than a partial parse: a
    composition missing an element gives a wrong electron count, and a wrong
    electron count produces a confident, wrong SLD comparison.
    """
    if not formula or not formula.strip():
        return None
    text = formula.strip().replace(" ", "")
    #  Parentheses and hydrates are real but not handled; say so rather than
    #  silently dropping them.
    if any(ch in text for ch in "()[]·*·"):
        return None

    counts: dict[str, float] = {}
    position = 0
    pattern = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
    while position < len(text):
        match = pattern.match(text, position)
        if match is None:
            return None
        symbol, amount = match.group(1), match.group(2)
        if symbol not in _ATOMIC:
            return None
        counts[symbol] = counts.get(symbol, 0.0) + (float(amount) if amount else 1.0)
        position = match.end()
    return counts or None


def electron_fraction(formula: str) -> float | None:
    """Z/A for a composition — the electrons-per-gram factor in an X-ray SLD."""
    counts = parse_formula(formula)
    if not counts:
        return None
    total_z = sum(_ATOMIC[symbol][0] * n for symbol, n in counts.items())
    total_a = sum(_ATOMIC[symbol][1] * n for symbol, n in counts.items())
    return (total_z / total_a) if total_a > 0 else None


def xray_sld_from_density(formula: str, density_g_cm3: float) -> float | None:
    """Predicted X-ray SLD in 1e-6 A^-2, from composition and mass density.

    ``SLD = r_e * rho_e``, with the electron density
    ``rho_e = rho * N_A * (Z/A)``.  This is far from the absorption edges, where
    the real part is essentially the electron count; near an edge the dispersion
    correction matters and this prediction should not be used.
    """
    fraction = electron_fraction(formula)
    if fraction is None or density_g_cm3 is None or density_g_cm3 <= 0:
        return None
    electrons_per_ang3 = _AVOGADRO_PER_ANG3 * density_g_cm3 * fraction
    return ELECTRON_RADIUS_ANG * electrons_per_ang3 * 1.0e6


#  Tolerance on the SLD/density cross-check. Wide on purpose: it has to absorb
#  the dispersion correction away from an edge, a film that is not fully dense,
#  and a fitted density that was itself free. Tighter than this and it fires on
#  every real film; looser and it stops catching unit errors.
SLD_DENSITY_TOLERANCE = 0.15

#  The permittivity-gap tradeoff for gate dielectrics, stated as the corner it
#  actually forbids rather than as a product range.
#
#  Both quantities trace to the same polarizability, so they move against each
#  other across the oxide family: SiO2 is k~3.9 / Eg~9, Al2O3 k~9 / Eg~8.8,
#  HfO2 and ZrO2 k~25 / Eg~5.7, TiO2 k~80 / Eg~3.1. The product is not constant —
#  it climbs with k — so a product band is the wrong shape for this rule. What is
#  genuinely rare is the top-right corner: high k *and* wide gap together, which
#  is where a mixed-up pair of numbers lands and where La2O3 (k~30, Eg~6) sits at
#  about the known limit.
#
#  Flagged as a heuristic, not a bound. A real material in that corner would be a
#  significant find, and this check exists to make someone look twice, never to
#  reject it.
KEG_BOTH_HIGH_K = 30.0
KEG_BOTH_HIGH_EG = 6.0

#  Minimum conduction-band offset for a dielectric on silicon, the standard rule
#  of thumb for keeping thermionic leakage acceptable.
MIN_BAND_OFFSET_EV = 1.0

#  One monolayer of a typical oxide. An ALD process whose growth-per-cycle
#  exceeds this is not self-limiting.
MONOLAYER_ANG = 3.0


def check_values(
    values: dict[str, float | None],
    *,
    formula: str | None = None,
    context: dict | None = None,
) -> PlausibilityReport:
    """Check a set of registry-keyed values for physical possibility.

    ``values`` maps registry keys (``k``, ``Eg``, ``sld_xray``, ``rho`` ...) to
    numbers. ``formula`` enables the composition-dependent cross-checks.
    ``context`` may carry ``substrate``, ``growth_technique``, and
    ``growth_per_cycle_ang`` for the process heuristics.
    """
    report = PlausibilityReport()
    context = context or {}
    present = {key: value for key, value in values.items() if isinstance(value, (int, float))}

    _check_hard_bounds(present, report)
    _check_identities(present, report)
    _check_sld_density(present, report, formula)
    _check_heuristics(present, report, context)

    report.checked = sorted(present)
    for key in values:
        if key not in present:
            report.not_checked[key] = "no numeric value given"
    return report


def _check_hard_bounds(values: dict[str, float], report: PlausibilityReport) -> None:
    for key, value in values.items():
        bounds = HARD_BOUNDS.get(key)
        if bounds is None:
            report.not_checked.setdefault(key, "no declared physical bound for this key")
            continue
        low, high, why = bounds
        if math.isnan(value):
            report.findings.append(
                Finding("violation", [key], f"{key} is NaN.", "NaN is missing data, not a value: leave it out (Eq. 4).")
            )
            continue
        if low is not None and value < low:
            report.findings.append(
                Finding("violation", [key], f"{key} = {value:g} is below the physical minimum {low:g}. {why}",
                        "Check the unit, then the sign convention, then the source.")
            )
        if high is not None and value > high:
            report.findings.append(
                Finding("violation", [key], f"{key} = {value:g} exceeds the physical maximum {high:g}. {why}",
                        "Check the unit first — this is almost always a factor of 10^3 or 10^6.")
            )


def _check_identities(values: dict[str, float], report: PlausibilityReport) -> None:
    """Inequalities between values that hold for physical reasons."""
    k = values.get("k")
    eps_inf = values.get("eps_inf")
    eps_ionic = values.get("eps_ionic")
    gap = values.get("Eg")
    offset = values.get("dEc")

    if k is not None and eps_inf is not None and eps_inf > k * (1 + 1e-9):
        report.findings.append(
            Finding(
                "violation",
                ["k", "eps_inf"],
                f"eps_inf = {eps_inf:g} exceeds the static permittivity k = {k:g}. The static "
                "response includes every contribution the optical response does, plus the ionic "
                "one, so it cannot be the smaller of the two.",
                "One of the two is wrong, or they were measured on different specimens. Check "
                "which frequency each was taken at — k at 10 kHz and eps_inf from ellipsometry "
                "are not the same material state if the film relaxed in between.",
            )
        )

    if k is not None and eps_inf is not None and eps_ionic is not None:
        #  Eq. (11): k = eps_inf + eps_ionic. Reported as an inconsistency rather
        #  than a violation, because all three are separately measured and the
        #  sum holds to measurement error, not exactly.
        expected = eps_inf + eps_ionic
        if expected > 0 and abs(k - expected) / expected > 0.1:
            report.findings.append(
                Finding(
                    "inconsistency",
                    ["k", "eps_inf", "eps_ionic"],
                    f"k = {k:g}, but eps_inf + eps_ionic = {expected:g} — a "
                    f"{abs(k - expected) / expected:.0%} discrepancy against Eq. (11).",
                    "Either a contribution is missing (a low-frequency dipolar or space-charge "
                    "term that k picks up and the sum does not), or the three values come from "
                    "different specimens.",
                )
            )

    if gap is not None and offset is not None and offset > gap:
        report.findings.append(
            Finding(
                "violation",
                ["Eg", "dEc"],
                f"dEc = {offset:g} eV exceeds Eg = {gap:g} eV. A conduction-band offset larger "
                "than the whole gap would place the dielectric's conduction band below its own "
                "valence band.",
                "Check whether dEc was recorded against the wrong reference, or whether Eg is "
                "an optical gap where the offset was taken against a transport gap.",
            )
        )


def _check_sld_density(
    values: dict[str, float], report: PlausibilityReport, formula: str | None
) -> None:
    """X-ray SLD against mass density, through the electron density."""
    sld = values.get("sld_xray")
    density = values.get("rho") or values.get("density")
    if sld is None or density is None:
        return
    if not formula:
        report.not_checked["sld_xray_vs_rho"] = (
            "no formula given, so the electron count is unknown"
        )
        return

    predicted = xray_sld_from_density(formula, density)
    if predicted is None:
        report.not_checked["sld_xray_vs_rho"] = (
            f"could not parse {formula!r} against the bundled element table"
        )
        return
    if predicted <= 0:
        return

    deviation = abs(sld - predicted) / predicted
    if deviation > SLD_DENSITY_TOLERANCE:
        report.findings.append(
            Finding(
                "inconsistency",
                ["sld_xray", "rho"],
                f"An X-ray SLD of {sld:g}e-6 A^-2 and a density of {density:g} g/cm^3 disagree "
                f"for {formula}: the electron density implies {predicted:.1f}e-6 A^-2, a "
                f"{deviation:.0%} difference. SLD = r_e * rho * N_A * (Z/A), so for a fixed "
                "composition these two are one measurement, not two.",
                "In a co-refinement this usually means density and SLD were both left free and "
                "the fit traded one against the other — they are nearly degenerate in XRR. Fix "
                "density from the composition and refit, or check for porosity, which lowers the "
                "real density below the crystallographic one.",
            )
        )


def _check_heuristics(
    values: dict[str, float], report: PlausibilityReport, context: dict
) -> None:
    k = values.get("k")
    gap = values.get("Eg")
    offset = values.get("dEc")

    if k is not None and gap is not None and k >= KEG_BOTH_HIGH_K and gap >= KEG_BOTH_HIGH_EG:
        report.findings.append(
            Finding(
                "heuristic",
                ["k", "Eg"],
                f"k = {k:g} and Eg = {gap:g} eV are both high. Permittivity and band gap trade "
                "off against each other across the oxides — both come from the same "
                "polarizability — so this corner is close to empty: La2O3 at about k 30 / Eg 6 is "
                "roughly the known limit, and HfO2 at k 25 buys its permittivity with a 5.7 eV "
                "gap. A material with both is either a significant find or a mixed-up pair of "
                "numbers.",
                "Confirm each against its own source before building on it — they are the two "
                "most commonly swapped numbers in a dielectric table. A real exception here is a "
                "result, not an error; this is a trend across a family, not a law.",
            )
        )

    if offset is not None and offset < MIN_BAND_OFFSET_EV:
        report.findings.append(
            Finding(
                "heuristic",
                ["dEc"],
                f"dEc = {offset:g} eV is below the ~{MIN_BAND_OFFSET_EV:g} eV usually wanted for "
                "a dielectric on silicon; thermionic emission over a barrier this low tends to "
                "dominate the leakage regardless of how thick the film is.",
                "Not a data problem — a device-viability one. It belongs in the interpretation, "
                "not in an exclusion.",
            )
        )

    technique = str(context.get("growth_technique", "")).lower()
    gpc = context.get("growth_per_cycle_ang")
    if technique == "ald" and isinstance(gpc, (int, float)) and gpc > MONOLAYER_ANG:
        report.findings.append(
            Finding(
                "heuristic",
                ["growth_per_cycle_ang"],
                f"A growth-per-cycle of {gpc:g} A exceeds about one monolayer "
                f"(~{MONOLAYER_ANG:g} A). Self-limiting surface chemistry cannot deposit more "
                "than a monolayer per cycle, so this is CVD-like behaviour rather than ALD.",
                "Usually an unpurged precursor or a substrate temperature above the ALD window. "
                "Check the purge times and the window before trusting the thickness calibration.",
            )
        )


def check_fit_layer(layer, *, techniques: list[str] | None = None) -> PlausibilityReport:
    """Run the checks against one stored ``FitLayer``.

    Bridges a ModalFit refinement into the same checker the property tables use,
    so "is this fit physical?" and "is this literature value physical?" get the
    same answer for the same reason.
    """
    parameters = layer.parameters or {}
    values: dict[str, float | None] = {
        "rho": layer.density_g_cm3,
        "thickness_nm": (layer.thickness_ang / 10.0) if layer.thickness_ang else None,
    }
    xray = parameters.get("xray") or {}
    neutron = parameters.get("neutron") or {}
    if "sld_real" in xray:
        values["sld_xray"] = xray["sld_real"]
    if "sld_imag" in xray:
        values["sld_xray_imag"] = xray["sld_imag"]
    if "sld_real" in neutron:
        values["sld_neutron"] = neutron["sld_real"]
    if "sld_imag" in neutron:
        values["sld_neutron_imag"] = neutron["sld_imag"]

    report = check_values(values, formula=layer.formula or layer.material)

    #  A roughness comparable to the layer it sits on is not a rough interface,
    #  it is a stack the model cannot resolve as discrete layers.
    if layer.thickness_ang and layer.roughness_ang and layer.thickness_ang > 0:
        ratio = layer.roughness_ang / layer.thickness_ang
        if ratio > 0.5:
            report.findings.append(
                Finding(
                    "heuristic",
                    ["roughness_ang", "thickness_ang"],
                    f"Roughness {layer.roughness_ang:g} A is {ratio:.0%} of the layer's "
                    f"{layer.thickness_ang:g} A thickness. A Nevot-Croce factor models a "
                    "perturbation on a sharp interface; at this ratio the layer is not a slab "
                    "with a rough edge, it is a gradient, and the fitted thickness of a gradient "
                    "depends on the model more than on the sample.",
                    "Consider an explicit interlayer or a graded profile instead of one slab "
                    "with large roughness.",
                )
            )

    if techniques and "SPR" in techniques and layer.roughness_ang:
        report.not_checked["roughness_spr"] = (
            "ModalFit's SPR forward model applies no roughness, so this value was not "
            "constrained by the SPR data"
        )
    return report
