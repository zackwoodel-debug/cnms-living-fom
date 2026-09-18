"""/materials — the S and P layers, and the descriptor dictionary."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, selectinload

from cnms_fom.db.base import get_db
from cnms_fom.db.enums import SpecimenForm
from cnms_fom.db.models import DescriptorValue, Material, PropertyValue, StructureRecord
from cnms_fom.descriptors.registry import (
    PHYSICAL_PROPERTIES,
    STRUCTURAL_DESCRIPTORS,
    descriptor_dictionary,
)
from cnms_fom.schemas.materials import (
    DescriptorComputeOut,
    MaterialDetailOut,
    MaterialIn,
    MaterialOut,
    PropertyValueIn,
    PropertyValueOut,
    StructureIn,
)

router = APIRouter(prefix="/materials", tags=["materials"])


@router.get("/dictionary")
def get_dictionary() -> dict:
    """The descriptor dictionary required by FOM_PROOF Sec. 13.1.

    Ships with every released analysis: name, formula, units, source method,
    physical interpretation, declared transform, and missing-value policy.
    """
    return descriptor_dictionary()


@router.get("", response_model=list[MaterialOut])
def list_materials(
    db: Session = Depends(get_db),
    formula: str | None = Query(default=None, description="Substring match on the reduced formula."),
    polymorph: str | None = None,
    specimen_form: SpecimenForm | None = None,
    limit: int = Query(default=100, le=1000),
    offset: int = 0,
) -> list[Material]:
    query = db.query(Material)
    if formula:
        query = query.filter(Material.formula_reduced.ilike(f"%{formula}%"))
    if polymorph:
        query = query.filter(Material.polymorph == polymorph)
    if specimen_form:
        query = query.filter(Material.specimen_form == specimen_form)
    return query.order_by(Material.id).offset(offset).limit(limit).all()


@router.post("", response_model=MaterialOut, status_code=status.HTTP_201_CREATED)
def create_material(payload: MaterialIn, db: Session = Depends(get_db)) -> Material:
    """Create a material-context record.

    The reduced formula is derived with pymatgen when it is installed; otherwise
    the supplied formula is used verbatim. The uniqueness constraint is on
    (reduced formula, polymorph, specimen form) — Eq. (3)'s identity, not the
    formula alone.
    """
    reduced = payload.formula
    try:
        from pymatgen.core import Composition

        reduced = Composition(payload.formula).reduced_formula
    except Exception:  # noqa: BLE001 - pymatgen optional, or an exotic formula string
        pass

    existing = (
        db.query(Material)
        .filter(
            Material.formula_reduced == reduced,
            Material.polymorph == payload.polymorph,
            Material.specimen_form == payload.specimen_form,
        )
        .one_or_none()
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Material {reduced}/{payload.polymorph}/{payload.specimen_form.value} already "
                f"exists as id {existing.id}."
            ),
        )

    material = Material(
        **payload.model_dump(exclude={"formula"}),
        formula=payload.formula,
        formula_reduced=reduced,
    )
    db.add(material)
    db.commit()
    db.refresh(material)
    return material


@router.get("/{material_id}", response_model=MaterialDetailOut)
def get_material(material_id: int, db: Session = Depends(get_db)) -> Material:
    material = (
        db.query(Material)
        .options(selectinload(Material.descriptors), selectinload(Material.properties))
        .filter(Material.id == material_id)
        .one_or_none()
    )
    if material is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No material with id {material_id}.")
    return material


@router.post(
    "/{material_id}/properties",
    response_model=PropertyValueOut,
    status_code=status.HTTP_201_CREATED,
)
def add_property(
    material_id: int, payload: PropertyValueIn, db: Session = Depends(get_db)
) -> PropertyValue:
    """Attach a property value, with its full measurement or calculation context.

    Rejects a value missing any context field the registry declares for that
    property. FOM_PROOF Sec. 16: a breakdown field without thickness, electrode,
    area, and failure criterion cannot be compared with another one, so storing
    it would only add a row that no analysis can use.
    """
    if db.get(Material, material_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No material with id {material_id}.")

    spec = PHYSICAL_PROPERTIES.get(payload.property_key)
    if spec is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Unknown property {payload.property_key!r}. Known: {sorted(PHYSICAL_PROPERTIES)}.",
        )

    data = payload.model_dump()
    if payload.value is not None:
        missing = [f for f in spec.required_context if not data.get(f)]
        if missing:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                {
                    "error": f"Missing required context for {payload.property_key!r}: {missing}.",
                    "why": spec.caveat
                    or "FOM_PROOF Sec. 16: context is what makes a value comparable.",
                },
            )

    value = PropertyValue(material_id=material_id, units=data.pop("units", None) or spec.units, **{
        k: v for k, v in data.items() if k != "units"
    })
    db.add(value)
    db.commit()
    db.refresh(value)
    return value


@router.post("/{material_id}/structure", status_code=status.HTTP_201_CREATED)
def add_structure(material_id: int, payload: StructureIn, db: Session = Depends(get_db)) -> dict:
    """Attach a CIF structure that descriptors can be computed from."""
    if db.get(Material, material_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No material with id {material_id}.")

    n_sites = volume = None
    try:
        from cnms_fom.descriptors.structural import structure_from_cif

        structure = structure_from_cif(payload.cif)
        n_sites, volume = len(structure), float(structure.volume)
    except ImportError:
        pass  # pymatgen absent: store the CIF, derive geometry when the extra is installed
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unparseable CIF: {exc}") from exc

    record = StructureRecord(
        material_id=material_id,
        cif=payload.cif,
        n_sites=n_sites,
        volume_ang3=volume,
        method=payload.method,
        xc_functional=payload.xc_functional,
        provenance_tier=payload.provenance_tier,
        doi=payload.doi,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"structure_id": record.id, "n_sites": n_sites, "volume_ang3": volume}


@router.post("/{material_id}/descriptors/compute", response_model=DescriptorComputeOut)
def compute_descriptors(material_id: int, db: Session = Depends(get_db)) -> DescriptorComputeOut:
    """Compute the geometry-derived descriptors of Table 2 from the stored structure.

    Z*_RMS, omega_TO,min, S_osc, and A_eps are *not* computed here — they come
    from DFPT or spectroscopy and must be ingested with their own method and
    XC-functional metadata (Sec. 16 items 8-9). They are listed in the response
    so completeness is visible rather than assumed.
    """
    material = (
        db.query(Material)
        .options(selectinload(Material.structures))
        .filter(Material.id == material_id)
        .one_or_none()
    )
    if material is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No material with id {material_id}.")
    if not material.structures:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No structure on record. POST a CIF to /materials/{id}/structure first.",
        )

    try:
        from cnms_fom.descriptors.structural import (
            compute_geometric_descriptors,
            missing_from_structure,
            structure_from_cif,
        )
    except ImportError as exc:
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "Structural descriptors need the 'descriptors' extra: pip install -e '.[descriptors]'",
        ) from exc

    record = material.structures[-1]  # most recently added
    structure = structure_from_cif(record.cif)
    results = compute_geometric_descriptors(structure)

    warnings: list[str] = []
    #  One query for every descriptor already on this material, rather than one
    #  lookup per computed descriptor inside the loop. The unique constraint is
    #  (material_id, descriptor_key, method), so that tuple is the natural key.
    existing_by_key = {
        (row.descriptor_key, row.method): row
        for row in db.query(DescriptorValue)
        .filter(DescriptorValue.material_id == material_id)
        .all()
    }

    for result in results:
        if result.note:
            warnings.append(f"{result.key}: {result.note}")
        existing = existing_by_key.get((result.key, result.method))
        if existing is None:
            db.add(
                DescriptorValue(
                    material_id=material_id,
                    structure_id=record.id,
                    descriptor_key=result.key,
                    value=result.value,
                    units=result.units,
                    method=result.method,
                    provenance_tier=result.provenance_tier,
                )
            )
        else:
            existing.value = result.value
    db.commit()

    absent = [
        key
        for key in missing_from_structure()
        if key in STRUCTURAL_DESCRIPTORS
        and not any(
            d.descriptor_key == key and d.value is not None for d in material.descriptors
        )
    ]
    return DescriptorComputeOut(
        material_id=material_id,
        computed=[r.as_dict() for r in results],  # type: ignore[arg-type]
        not_derivable_from_structure=absent,
        warnings=warnings,
    )
