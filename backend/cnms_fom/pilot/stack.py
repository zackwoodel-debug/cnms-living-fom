"""Build a layer stack from the materials database.

A stack is what a reflectivity calculation actually consumes: an ordered list of
layers from ambient to substrate, each with a thickness, a roughness, and a
scattering length density.

SLD comes from the database where the import supplied it, and is otherwise
computed from composition and density. Both paths record which was used —
FOM_PROOF Sec. 2.2 wants provenance on every number, and "where did this SLD
come from" is exactly the question a reflectivity fit gets challenged on.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field

from sqlalchemy import select

from cnms_fom.db.models import Material, PropertyValue, SpectralPoint, SpectralSeries

#  Classical electron radius in angstrom.
R_ELECTRON_ANG = 2.8179403262e-5
AVOGADRO = 6.02214076e23
#  cm^3 -> A^3
CM3_PER_ANG3 = 1e-24

#  Native oxide on Si. Always present under an HfO2 film grown on silicon, and
#  it dominates the achievable EOT, so leaving it out of the stack would make
#  the pilot optimistic in a way that matters.
NATIVE_SIO2_THICKNESS_ANG = 10.0


@dataclass
class StackLayer:
    """One layer, ambient-side first."""

    name: str
    thickness_ang: float           # 0 for semi-infinite ambient/substrate
    roughness_ang: float           # roughness of this layer's *top* interface
    sld_real: float                # 1e-6 A^-2
    sld_imag: float = 0.0          # 1e-6 A^-2, absorption
    material_id: int | None = None
    sld_source: str = ""           # how the SLD was obtained

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "thickness_ang": self.thickness_ang,
            "roughness_ang": self.roughness_ang,
            "sld_real_1e6_ang-2": self.sld_real,
            "sld_imag_1e6_ang-2": self.sld_imag,
            "material_id": self.material_id,
            "sld_source": self.sld_source,
        }


@dataclass
class Stack:
    """An ordered stack plus the recipe and provenance that produced it."""

    layers: list[StackLayer]
    recipe: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "layers": [layer.as_dict() for layer in self.layers],
            "recipe": self.recipe,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)


#  Formula -> (molar mass g/mol, total atomic number per formula unit, density g/cm^3).
#  Used only when the database has no measured SLD. Densities are bulk
#  crystalline values; a real film is usually 90-97% of bulk, which is itself a
#  fit parameter in XRR and one of the reasons a measured SLD is preferable.
_COMPOSITION: dict[str, tuple[float, int, float]] = {
    "HfO2": (210.49, 72 + 2 * 8, 9.68),
    "SiO2": (60.08, 14 + 2 * 8, 2.20),
    "Si": (28.0855, 14, 2.329),
    "Al2O3": (101.96, 2 * 13 + 3 * 8, 3.95),
}


def xray_sld_from_composition(formula: str, density_g_cm3: float | None = None) -> float:
    """Compute x-ray SLD from composition and density, in 1e-6 A^-2.

        SLD = r_e * (rho * N_A / M) * sum(Z)

    Neglects anomalous dispersion (f', f''), which is the right approximation
    well away from an absorption edge and a poor one near it. Hafnium's L3 edge
    sits at 9.56 keV, above Cu K-alpha at 8.05 keV, so the film is below the
    edge and f' is a few percent — acceptable for a pilot, and a reason to
    prefer the measured value when the database has one.
    """
    try:
        molar_mass, z_total, default_density = _COMPOSITION[formula]
    except KeyError:
        raise KeyError(
            f"No composition entry for {formula!r}. Add it to _COMPOSITION, or ingest a "
            "measured SLD so the calculation is not needed."
        ) from None

    density = density_g_cm3 if density_g_cm3 is not None else default_density
    number_density_ang3 = density * AVOGADRO / molar_mass * CM3_PER_ANG3
    return R_ELECTRON_ANG * number_density_ang3 * z_total * 1e6


def _lookup_sld(session, formula: str) -> tuple[float, float, int | None, str] | None:
    """Measured/calculated SLD for a formula, if the database has one."""
    row = session.execute(
        select(PropertyValue.value, PropertyValue.material_id, PropertyValue.method)
        .join(Material, Material.id == PropertyValue.material_id)
        .where(
            Material.formula_reduced == formula,
            PropertyValue.property_key == "sld_xray",
            PropertyValue.value.isnot(None),
        )
        .order_by(PropertyValue.id)
        .limit(1)
    ).first()
    if row is None:
        return None
    value, material_id, method = row

    imag_row = session.execute(
        select(PropertyValue.value)
        .where(
            PropertyValue.material_id == material_id,
            PropertyValue.property_key == "sld_xray_imag",
        )
        .limit(1)
    ).first()
    return float(value), float(imag_row[0]) if imag_row else 0.0, material_id, f"database ({method})"


def _sld_for(session, formula: str, density_g_cm3: float | None = None) -> StackLayer:
    """Resolve an SLD, preferring the database over the calculation."""
    found = _lookup_sld(session, formula)
    if found is not None:
        real, imag, material_id, source = found
        return StackLayer(formula, 0.0, 0.0, real, imag, material_id, source)
    return StackLayer(
        formula,
        0.0,
        0.0,
        xray_sld_from_composition(formula, density_g_cm3),
        0.0,
        None,
        "computed from composition and bulk density (anomalous dispersion neglected)",
    )


def build_stack(
    session,
    recipe: dict,
    *,
    film_formula: str = "HfO2",
    substrate_formula: str = "Si",
    interfacial_formula: str = "SiO2",
    interfacial_thickness_ang: float = NATIVE_SIO2_THICKNESS_ANG,
) -> Stack:
    """Assemble air / film / interfacial oxide / substrate from a recipe.

    ``recipe`` supplies ``thickness_ang`` and ``roughness_ang``, and optionally
    ``dopant_fraction``. Doping changes the film's SLD roughly in proportion to
    the electron-density change; for the light dopants used to stabilise HfO2
    (Si, Al) the effect is a few percent and is applied linearly.
    """
    thickness = float(recipe["thickness_ang"])
    roughness = float(recipe["roughness_ang"])
    dopant = float(recipe.get("dopant_fraction", 0.0))

    film = _sld_for(session, film_formula)
    interfacial = _sld_for(session, interfacial_formula)
    substrate = _sld_for(session, substrate_formula)

    notes: list[str] = []
    if dopant > 0:
        #  Linear mixing of the film and dopant-oxide electron densities. Crude,
        #  and adequate at the few-percent doping that stabilises the high-k
        #  phases; it is not a substitute for measuring the doped film.
        dopant_sld = xray_sld_from_composition("Al2O3")
        film.sld_real = (1 - dopant) * film.sld_real + dopant * dopant_sld
        notes.append(
            f"Film SLD linearly mixed for dopant_fraction={dopant:.3f} "
            "(Al2O3 as the dopant oxide)."
        )

    film.thickness_ang = thickness
    film.roughness_ang = roughness
    interfacial.thickness_ang = interfacial_thickness_ang
    #  The buried interface is smoother than the free surface in practice.
    interfacial.roughness_ang = min(roughness, 3.0)
    substrate.roughness_ang = 2.0

    for layer in (film, interfacial, substrate):
        if "computed" in layer.sld_source:
            notes.append(f"{layer.name}: SLD {layer.sld_source}.")

    ambient = StackLayer("air", 0.0, 0.0, 0.0, 0.0, None, "vacuum reference")
    return Stack(
        layers=[ambient, film, interfacial, substrate],
        recipe=dict(recipe),
        notes=notes,
    )


def stack_to_nk_csv(session, stack: Stack, *, max_points: int = 400) -> str:
    """Emit the stack's optical dispersion as CSV, when the database has it.

    Columns: ``layer,wavelength_nm,n,k``. Layers with no ingested dispersion are
    omitted rather than filled with a guess — a missing curve is a data gap, and
    Sec. 2.3 says it stays one.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["layer", "wavelength_nm", "n", "k"])

    for layer in stack.layers:
        if layer.material_id is None:
            continue
        series_rows = session.execute(
            select(SpectralSeries.id, SpectralSeries.quantity).where(
                SpectralSeries.material_id == layer.material_id,
                SpectralSeries.quantity.in_(("n", "k")),
            )
        ).all()
        by_quantity = {quantity: series_id for series_id, quantity in series_rows}
        if "n" not in by_quantity:
            continue

        n_points = session.execute(
            select(SpectralPoint.x_value, SpectralPoint.y_value)
            .where(SpectralPoint.series_id == by_quantity["n"])
            .order_by(SpectralPoint.x_value)
        ).all()
        k_lookup = (
            dict(
                session.execute(
                    select(SpectralPoint.x_value, SpectralPoint.y_value).where(
                        SpectralPoint.series_id == by_quantity["k"]
                    )
                ).all()
            )
            if "k" in by_quantity
            else {}
        )

        stride = max(1, math.ceil(len(n_points) / max_points))
        for wavelength, n_value in n_points[::stride]:
            writer.writerow(
                [layer.name, f"{wavelength:.4f}", f"{n_value:.6f}", k_lookup.get(wavelength, "")]
            )

    return buffer.getvalue()


def export_stack(session, recipe: dict, **kwargs) -> tuple[Stack, str, str]:
    """Convenience: the stack, its JSON, and its n,k CSV in one call."""
    stack = build_stack(session, recipe, **kwargs)
    return stack, stack.to_json(), stack_to_nk_csv(session, stack)
