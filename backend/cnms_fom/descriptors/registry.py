"""The descriptor dictionary (FOM_PROOF Sec. 13.1).

Every descriptor and property the platform knows about is declared here, with
its formula, units, physical interpretation, declared transform, and
missing-value policy.  Nothing computes a quantity that is not in this file.

Why a registry instead of loose column names: Sec. 13.1 requires the dictionary
to be part of the released output, and Sec. 5.1 requires the transform to be
stored with the run.  Keeping both on the same object makes the two consistent
by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from cnms_fom.db.enums import Direction, ProvenanceTier, Transform


@dataclass(frozen=True)
class DescriptorSpec:
    """One structural descriptor S_j."""

    key: str
    symbol: str
    name: str
    units: str
    formula: str
    interpretation: str
    default_transform: Transform = Transform.NONE
    default_tier: ProvenanceTier = ProvenanceTier.CALCULATED
    source_method: str = ""
    #  Sec. 2.3 — what happens when the value is absent.  Always "exclude".
    missing_policy: str = "exclude record from any analysis requiring this descriptor"
    caveat: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["default_transform"] = self.default_transform.value
        d["default_tier"] = self.default_tier.value
        return d


@dataclass(frozen=True)
class PropertySpec:
    """One physical property P_q."""

    key: str
    symbol: str
    name: str
    units: str
    interpretation: str
    #  Sec. 5.3 — beneficial or detrimental when larger.  Drives Eq. (23) vs (24).
    direction: Direction = Direction.BENEFIT
    default_transform: Transform = Transform.NONE
    #  Context fields that make the value meaningful (Sec. 16 checklist).
    required_context: tuple[str, ...] = field(default_factory=tuple)
    caveat: str = ""
    #  Whether this property may appear in a composite score.  Some quantities
    #  are worth storing and correlating without being application figures of
    #  merit — scattering length density characterises a material precisely but
    #  is not "better when larger" for any device.  ``direction`` is ignored
    #  when this is False.
    fom_eligible: bool = True

    def as_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["default_transform"] = self.default_transform.value
        d["required_context"] = list(self.required_context)
        return d


# ---------------------------------------------------------------------------
# Structural descriptors — FOM_PROOF Eq. (6) and Table 2
# ---------------------------------------------------------------------------

STRUCTURAL_DESCRIPTORS: dict[str, DescriptorSpec] = {
    "V_fu": DescriptorSpec(
        key="V_fu",
        symbol="V_fu",
        name="Volume per formula unit",
        units="A^3/f.u.",
        formula="V_cell / Z",
        interpretation="Volume available per formula unit; may influence polarization density.",
        default_transform=Transform.LOG10,  # Eq. (19)
        source_method="pymatgen Structure.volume / formula units per cell",
    ),
    "rho": DescriptorSpec(
        key="rho",
        symbol="rho",
        name="Mass density",
        units="g/cm^3",
        formula="m_cell / V_cell",
        interpretation="Packing-density descriptor.",
        source_method="pymatgen Structure.density",
        caveat=(
            "Table 2: must not be read as a direct causal mechanism without controls; "
            "density is collinear with volume and bonding."
        ),
    ),
    "CN": DescriptorSpec(
        key="CN",
        symbol="CN",
        name="Cation coordination number",
        units="dimensionless",
        formula="mean nearest-neighbour count over cation sites",
        interpretation="Simple local-environment descriptor.",
        source_method="pymatgen CrystalNN",
        caveat=(
            "Sec. 9.2 and 15.2: often confounded with chemistry and bonding. "
            "A correlation with CN alone does not establish that CN is the cause."
        ),
    ),
    "d_M_O": DescriptorSpec(
        key="d_M_O",
        symbol="d_M-O",
        name="Mean metal-oxygen bond length",
        units="A",
        formula="mean over cation-anion nearest-neighbour bonds",
        interpretation="Local bond-scale descriptor.",
        source_method="pymatgen CrystalNN neighbour distances",
    ),
    "sigma_d": DescriptorSpec(
        key="sigma_d",
        symbol="sigma_d",
        name="Bond-length standard deviation",
        units="A",
        formula="sample standard deviation of cation-anion bond lengths",
        interpretation="Quantifies local coordination distortion.",
        source_method="pymatgen CrystalNN neighbour distances",
        caveat="State whether a polyhedral-distortion index is used instead; do not mix the two.",
    ),
    "delta_chi": DescriptorSpec(
        key="delta_chi",
        symbol="delta_chi",
        name="Cation-anion electronegativity difference",
        units="Pauling units",
        formula="chi_anion - chi_cation (composition-weighted)",
        interpretation="Approximate ionic/covalent bonding descriptor.",
        source_method="pymatgen Element.X",
    ),
    "Z_RMS_star": DescriptorSpec(
        key="Z_RMS_star",
        symbol="Z*_RMS",
        name="RMS Born effective charge",
        units="e",
        formula="declared scalar reduction of the Born effective-charge tensor",
        interpretation="Dynamical charge response relevant to polar-mode strength.",
        default_tier=ProvenanceTier.CALCULATED,
        source_method="DFPT (record software, XC functional, pseudopotentials)",
        caveat=(
            "Sec. 16 item 8: the reduction rule and the computational method must both be "
            "recorded. Z* from different functionals are not interchangeable."
        ),
    ),
    "omega_TO_min": DescriptorSpec(
        key="omega_TO_min",
        symbol="omega_TO,min",
        name="Lowest IR-active TO frequency",
        units="cm^-1",
        formula="min over selected infrared-active transverse-optical modes",
        interpretation="Polar-mode softness descriptor.",
        default_transform=Transform.LOG10,  # Eq. (19)
        default_tier=ProvenanceTier.CALCULATED,
        source_method="DFPT phonons or IR/Raman spectroscopy",
        caveat=(
            "Sec. 16 item 9: record mode symmetry, polarization, temperature, and whether the "
            "value is harmonic, anharmonic, or measured."
        ),
    ),
    "S_osc": DescriptorSpec(
        key="S_osc",
        symbol="S_osc",
        name="Mode oscillator strength",
        units="dimensionless",
        formula="dipole strength of the selected polar mode",
        interpretation="Measures the dipole strength of a polar vibrational mode.",
        default_tier=ProvenanceTier.CALCULATED,
        source_method="DFPT mode-resolved oscillator strengths",
    ),
    "A_eps": DescriptorSpec(
        key="A_eps",
        symbol="A_eps",
        name="Dielectric anisotropy",
        units="dimensionless",
        formula="eps_max / eps_min",
        interpretation="Retains directional information that an isotropic average would destroy.",
        source_method="eigenvalues of the dielectric tensor",
        caveat="Sec. 3.2: never collapse a tensor to a scalar without declaring the rule.",
    ),
}


# ---------------------------------------------------------------------------
# Physical properties — FOM_PROOF Eq. (10)
# ---------------------------------------------------------------------------

PHYSICAL_PROPERTIES: dict[str, PropertySpec] = {
    "k": PropertySpec(
        key="k",
        symbol="k",
        name="Dielectric constant (relevant direction)",
        units="dimensionless",
        interpretation="Capacitance benefit; k = eps_static in the device-relevant direction.",
        direction=Direction.BENEFIT,
        default_transform=Transform.LOG10,  # Eq. (19)
        required_context=("temperature_k", "frequency_hz", "tensor_component"),
        caveat="Sec. 15.2: a high k does not by itself make a good dielectric for an application.",
    ),
    "eps_inf": PropertySpec(
        key="eps_inf",
        symbol="eps_inf",
        name="High-frequency permittivity",
        units="dimensionless",
        interpretation="Electronic contribution in Eq. (11).",
        direction=Direction.BENEFIT,
        required_context=("method",),
        caveat=(
            "Obtainable either from DFPT or from optical dispersion via eps = n^2 - k^2 in a "
            "transparent window. `method` must say which, and a first-principles value must also "
            "record `xc_functional` — demanding it unconditionally would reject every "
            "experimentally derived value, which have no exchange-correlation functional."
        ),
    ),
    "eps_ionic": PropertySpec(
        key="eps_ionic",
        symbol="eps_ionic",
        name="Ionic (lattice) permittivity",
        units="dimensionless",
        interpretation="Lattice contribution in Eq. (11); the target of the polar-mode mechanism.",
        direction=Direction.BENEFIT,
        required_context=("temperature_k", "method"),
    ),
    "Eg": PropertySpec(
        key="Eg",
        symbol="Eg",
        name="Band gap",
        units="eV",
        interpretation="Leakage and reliability headroom.",
        direction=Direction.BENEFIT,
        required_context=("method",),
        caveat=(
            "Semi-local DFT underestimates Eg; do not mix functionals within one analysis, and "
            "record `xc_functional` for any calculated value. An optical or photoemission gap "
            "has no functional, and is also not the same quantity as a Kohn-Sham gap — `method` "
            "is what distinguishes them."
        ),
    ),
    "dEc": PropertySpec(
        key="dEc",
        symbol="dEc",
        name="Conduction-band offset",
        units="eV",
        interpretation="Barrier to electron injection at the device interface.",
        direction=Direction.BENEFIT,
        required_context=("substrate", "interface", "method"),
        caveat="Sec. 15.2: a band offset is not universal; it depends on interface and processing.",
    ),
    "Ebd": PropertySpec(
        key="Ebd",
        symbol="E_bd",
        name="Breakdown field",
        units="MV/cm",
        interpretation="Reliability limit under applied field.",
        direction=Direction.BENEFIT,
        default_transform=Transform.LOG10,  # Eq. (19)
        required_context=(
            "thickness_nm",
            "electrode",
            "temperature_k",
            "area_cm2",
            "failure_criterion",
        ),
        caveat="Sec. 16 item 6: thickness, electrode, area, and failure criterion are mandatory.",
    ),
    "tan_delta": PropertySpec(
        key="tan_delta",
        symbol="tan delta",
        name="Loss tangent",
        units="dimensionless",
        interpretation="Dissipation; dominates RF usefulness.",
        direction=Direction.COST,
        default_transform=Transform.LOG10,  # Eq. (25) uses a log-scale normalization
        required_context=("frequency_hz", "temperature_k", "field_amplitude_v_per_cm"),
        caveat="Sec. 15.2: a loss value without frequency and temperature cannot rank RF materials.",
    ),
    "kappa_th": PropertySpec(
        key="kappa_th",
        symbol="kappa_th",
        name="Thermal conductivity",
        units="W/(m K)",
        interpretation="Heat extraction; matters for power devices.",
        direction=Direction.BENEFIT,
        required_context=("temperature_k", "method"),
    ),
    "sld_neutron": PropertySpec(
        key="sld_neutron",
        symbol="SLD_n",
        name="Neutron scattering length density",
        units="1e-6 A^-2",
        interpretation=(
            "Neutron contrast. Sets reflectivity and small-angle scattering, so it determines "
            "whether a layer is visible in a neutron experiment at all."
        ),
        required_context=("method",),
        fom_eligible=False,
        caveat=(
            "Isotope-dependent: a deuterated or isotopically enriched sample has a different "
            "SLD at identical composition. Record the isotopic assumption in `method`."
        ),
    ),
    "sld_xray": PropertySpec(
        key="sld_xray",
        symbol="SLD_x",
        name="X-ray scattering length density",
        units="1e-6 A^-2",
        interpretation="X-ray contrast; drives XRR fitting of film thickness and density.",
        required_context=("method",),
        fom_eligible=False,
        caveat="Energy-dependent. State the edge or wavelength (e.g. Cu K-alpha) in `method`.",
    ),
    "sld_xray_imag": PropertySpec(
        key="sld_xray_imag",
        symbol="SLD_x''",
        name="X-ray scattering length density (imaginary part)",
        units="1e-6 A^-2",
        interpretation="X-ray absorption. Sets how fast a reflectivity curve damps with angle.",
        required_context=("method",),
        fom_eligible=False,
        caveat="Strongly energy-dependent near an absorption edge; state the edge in `method`.",
    ),
    "sld_neutron_imag": PropertySpec(
        key="sld_neutron_imag",
        symbol="SLD_n''",
        name="Neutron scattering length density (imaginary part)",
        units="1e-6 A^-2",
        interpretation="Neutron absorption; negligible except for B, Cd, Gd and a few others.",
        required_context=("method",),
        fom_eligible=False,
    ),
    "alpha_th": PropertySpec(
        key="alpha_th",
        symbol="alpha_th",
        name="Thermal expansion coefficient",
        units="1e-6 / K",
        interpretation="Thermomechanical compatibility with the substrate stack.",
        direction=Direction.COST,
        required_context=("temperature_k", "method"),
    ),
}


def descriptor_dictionary() -> dict[str, list[dict]]:
    """Emit the Sec. 13.1 descriptor dictionary as serialisable records.

    This is what ships with a released analysis, and what ``/materials/dictionary``
    returns.
    """
    return {
        "structural_descriptors": [s.as_dict() for s in STRUCTURAL_DESCRIPTORS.values()],
        "physical_properties": [p.as_dict() for p in PHYSICAL_PROPERTIES.values()],
    }


def fom_eligible_properties() -> dict[str, PropertySpec]:
    """Properties that may legitimately appear in a composite score."""
    return {k: v for k, v in PHYSICAL_PROPERTIES.items() if v.fom_eligible}


def require_property(key: str) -> PropertySpec:
    """Look up a property, failing loudly on an unknown key."""
    try:
        return PHYSICAL_PROPERTIES[key]
    except KeyError:
        raise KeyError(
            f"Unknown property {key!r}. Add it to PHYSICAL_PROPERTIES with units, direction, "
            "and required context before using it in a FOM."
        ) from None


def require_descriptor(key: str) -> DescriptorSpec:
    """Look up a descriptor, failing loudly on an unknown key."""
    try:
        return STRUCTURAL_DESCRIPTORS[key]
    except KeyError:
        raise KeyError(
            f"Unknown descriptor {key!r}. Add it to STRUCTURAL_DESCRIPTORS with a formula, "
            "units, and physical interpretation before computing it."
        ) from None
