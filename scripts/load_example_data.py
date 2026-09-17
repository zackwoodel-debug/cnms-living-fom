#!/usr/bin/env python
"""Load illustrative materials so the platform can be exercised end to end.

READ THIS BEFORE USING THE OUTPUT FOR ANYTHING.

Every property value written by this script is tagged ``ProvenanceTier.MODELED``.
They are order-of-magnitude figures typical of these oxides, entered without a
DOI, a specimen, or a measurement context. They are *not* citable data.

That tagging is deliberate, and it is the point of the script. FOM_PROOF Sec. 2.3
permits a modeled-scenario analysis provided it is labelled modeled and kept
separate from measurement-based results, so these rows exercise exactly that
path: every score computed from them comes back with status ``illustrative`` and
an "ILLUSTRATIVE: modeled scenario" note. If you ever see one of these materials
in a ``scored`` (rather than ``illustrative``) ranking, the quarantine has broken
and that is a bug worth chasing.

To build a real dataset, enter values through ``POST /materials/{id}/properties``
with the DOI, page, and full measurement context, at tier ``measured`` or
``calculated``.

    python scripts/load_example_data.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from cnms_fom.db.base import session_scope  # noqa: E402
from cnms_fom.db.enums import ProvenanceTier, SpecimenForm  # noqa: E402
from cnms_fom.db.models import DescriptorValue, Material, PropertyValue  # noqa: E402

#  (formula, polymorph, specimen form, {property: value}, {descriptor: value})
#  Values are illustrative only — see the module docstring.
EXAMPLES = [
    (
        "SiO2",
        "amorphous",
        SpecimenForm.AMORPHOUS_FILM,
        {"k": 3.9, "Eg": 9.0, "dEc": 3.2, "Ebd": 10.0, "tan_delta": 1.0e-4, "kappa_th": 1.4},
        {"V_fu": 45.0, "rho": 2.2, "CN": 4.0, "d_M_O": 1.61, "delta_chi": 1.54},
    ),
    (
        "Al2O3",
        "amorphous",
        SpecimenForm.AMORPHOUS_FILM,
        {"k": 9.0, "Eg": 6.5, "dEc": 2.1, "Ebd": 8.0, "tan_delta": 3.0e-4, "kappa_th": 2.0},
        {"V_fu": 42.5, "rho": 3.2, "CN": 5.0, "d_M_O": 1.88, "delta_chi": 1.83},
    ),
    (
        "HfO2",
        "monoclinic",
        SpecimenForm.CRYSTALLINE_FILM,
        {"k": 25.0, "Eg": 5.7, "dEc": 1.5, "Ebd": 4.0, "tan_delta": 2.0e-3, "kappa_th": 1.1,
         "eps_inf": 4.5, "eps_ionic": 20.5},
        {"V_fu": 34.5, "rho": 9.7, "CN": 7.0, "d_M_O": 2.15, "delta_chi": 2.14,
         "Z_RMS_star": 4.8, "omega_TO_min": 135.0},
    ),
    (
        "ZrO2",
        "tetragonal",
        SpecimenForm.CRYSTALLINE_FILM,
        {"k": 30.0, "Eg": 5.5, "dEc": 1.4, "Ebd": 3.5, "tan_delta": 2.5e-3, "kappa_th": 2.0,
         "eps_inf": 4.8, "eps_ionic": 25.2},
        {"V_fu": 33.1, "rho": 6.1, "CN": 8.0, "d_M_O": 2.20, "delta_chi": 2.11,
         "Z_RMS_star": 5.1, "omega_TO_min": 128.0},
    ),
    (
        "TiO2",
        "rutile",
        SpecimenForm.BULK_SINGLE_CRYSTAL,
        {"k": 89.0, "Eg": 3.0, "dEc": 0.1, "Ebd": 1.0, "tan_delta": 5.0e-3, "kappa_th": 8.0,
         "eps_inf": 6.8, "eps_ionic": 82.2},
        {"V_fu": 31.2, "rho": 4.25, "CN": 6.0, "d_M_O": 1.96, "delta_chi": 1.90,
         "Z_RMS_star": 6.3, "omega_TO_min": 89.0, "A_eps": 1.6},
    ),
    (
        "Ta2O5",
        "orthorhombic",
        SpecimenForm.CRYSTALLINE_FILM,
        {"k": 25.0, "Eg": 4.4, "dEc": 0.4, "Ebd": 3.0, "tan_delta": 4.0e-3, "kappa_th": 1.0,
         "eps_inf": 4.9, "eps_ionic": 20.1},
        {"V_fu": 53.0, "rho": 8.2, "CN": 6.0, "d_M_O": 1.98, "delta_chi": 1.94,
         "Z_RMS_star": 5.5, "omega_TO_min": 110.0},
    ),
    (
        "La2O3",
        "hexagonal",
        SpecimenForm.CRYSTALLINE_FILM,
        {"k": 27.0, "Eg": 5.5, "dEc": 2.3, "Ebd": 2.5, "tan_delta": 6.0e-3, "kappa_th": 1.8,
         "eps_inf": 4.0, "eps_ionic": 23.0},
        {"V_fu": 47.9, "rho": 6.5, "CN": 7.0, "d_M_O": 2.45, "delta_chi": 2.34,
         "Z_RMS_star": 4.4, "omega_TO_min": 120.0},
    ),
    (
        "SrTiO3",
        "cubic perovskite",
        SpecimenForm.BULK_SINGLE_CRYSTAL,
        {"k": 300.0, "Eg": 3.2, "dEc": 0.0, "Ebd": 0.5, "tan_delta": 1.0e-3, "kappa_th": 11.0,
         "eps_inf": 5.2, "eps_ionic": 294.8},
        {"V_fu": 59.5, "rho": 5.11, "CN": 6.0, "d_M_O": 1.95, "delta_chi": 1.86,
         "Z_RMS_star": 6.9, "omega_TO_min": 42.0},
    ),
]


def main() -> int:
    created = 0
    now = datetime.now(timezone.utc)

    with session_scope() as db:
        for formula, polymorph, form, properties, descriptors in EXAMPLES:
            existing = (
                db.query(Material)
                .filter(
                    Material.formula_reduced == formula,
                    Material.polymorph == polymorph,
                    Material.specimen_form == form,
                )
                .one_or_none()
            )
            if existing is not None:
                print(f"  = {formula}/{polymorph} already present")
                continue

            material = Material(
                formula=formula,
                formula_reduced=formula,
                polymorph=polymorph,
                specimen_form=form,
                notes="ILLUSTRATIVE modeled data from scripts/load_example_data.py. Not citable.",
            )
            db.add(material)
            db.flush()

            for key, value in properties.items():
                db.add(
                    PropertyValue(
                        material_id=material.id,
                        property_key=key,
                        value=value,
                        #  MODELED, so every score built on these is ILLUSTRATIVE.
                        provenance_tier=ProvenanceTier.MODELED,
                        method="illustrative placeholder — no source",
                        ingested_at=now,
                        #  Context filled so the rows survive the eligibility
                        #  filter; the values remain modeled regardless.
                        temperature_k=300.0,
                        frequency_hz=1.0e4 if key in ("k", "tan_delta") else None,
                        tensor_component="iso" if key == "k" else None,
                        thickness_nm=10.0 if key == "Ebd" else None,
                        electrode="TiN" if key == "Ebd" else None,
                        area_cm2=1.0e-4 if key == "Ebd" else None,
                        failure_criterion="1 mA/cm^2" if key == "Ebd" else None,
                        field_amplitude_v_per_cm=1.0e4 if key == "tan_delta" else None,
                        substrate="Si(001)" if key == "dEc" else None,
                        interface=f"{formula}/SiO2/Si" if key == "dEc" else None,
                        xc_functional="illustrative" if key in ("Eg", "eps_inf", "eps_ionic") else None,
                    )
                )

            for key, value in descriptors.items():
                db.add(
                    DescriptorValue(
                        material_id=material.id,
                        descriptor_key=key,
                        value=value,
                        provenance_tier=ProvenanceTier.MODELED,
                        method="illustrative placeholder — no source",
                    )
                )
            created += 1
            print(f"  + {formula}/{polymorph} ({form.value})")

    print(
        f"\nLoaded {created} illustrative material(s). Every value is tier MODELED, so scores "
        "computed from them return status 'illustrative' and must never be reported as results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
