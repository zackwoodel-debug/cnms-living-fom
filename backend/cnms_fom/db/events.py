"""ORM event hooks that enforce protocol invariants a CHECK constraint cannot.

Two rules live here:

1. **Context digests stay correct.**  ``PropertyValue.context_digest`` and
   ``SpectralSeries.context_digest`` are derived columns.  Recomputing them on
   every insert and update means the uniqueness constraint cannot be defeated by
   editing a context field after the fact.

2. **A frozen FOM definition is immutable.**  Sec. 5.3 makes the bounds,
   transform, direction, floor, and weights part of the score's identity, so
   editing a frozen definition silently changes the meaning of every score
   already published against it.  There is no SQL constraint for "this row was
   writable yesterday and is not today", so it is enforced here.

Scope, stated plainly: these fire on ORM flushes.  ``Session.bulk_*`` and raw
``insert()`` Core statements bypass them.  That is an acceptable trade for a
research platform — the ingestion scripts compute digests explicitly — and
``scripts/check_protocol_compliance.py`` recomputes every digest so a bypass
shows up as a reported mismatch rather than as silent corruption.
"""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.orm import Session

from .context import context_digest
from .models import FomDefinition, PropertyValue, SpectralSeries

#  Columns that may still be edited on a frozen definition: administrative
#  metadata that does not change what the score means.
_FROZEN_EDITABLE = {"frozen", "approved", "approved_by", "approved_at", "updated_at"}


def _assign_digest(target) -> None:
    target.context_digest = context_digest(target)


@event.listens_for(PropertyValue, "before_insert")
@event.listens_for(PropertyValue, "before_update")
def _property_value_digest(mapper, connection, target: PropertyValue) -> None:  # noqa: ARG001
    _assign_digest(target)


@event.listens_for(SpectralSeries, "before_insert")
@event.listens_for(SpectralSeries, "before_update")
def _spectral_series_digest(mapper, connection, target: SpectralSeries) -> None:  # noqa: ARG001
    _assign_digest(target)


@event.listens_for(SpectralSeries, "before_insert")
@event.listens_for(SpectralSeries, "before_update")
def _spectral_series_extent(mapper, connection, target: SpectralSeries) -> None:  # noqa: ARG001
    """Keep the denormalised point count and x-range in step with the points.

    Only updated when the points are loaded; a bulk point insert should set them
    explicitly. Cheap enough to be worth having, since every listing of a
    spectrum wants the range without touching 10^5 rows.
    """
    if "points" not in target.__dict__:
        return  # relationship not loaded — leave whatever the caller set
    points = target.points or []
    target.n_points = len(points)
    if points:
        xs = [p.x_value for p in points]
        target.x_min, target.x_max = min(xs), max(xs)


@event.listens_for(Session, "before_flush")
def _block_frozen_definition_edits(session: Session, flush_context, instances) -> None:  # noqa: ARG001
    """Refuse to modify or delete a frozen ``FomDefinition``."""
    for obj in session.dirty:
        if not isinstance(obj, FomDefinition) or not session.is_modified(obj):
            continue
        state = obj._sa_instance_state  # noqa: SLF001 - the supported inspection path
        #  Read the pre-modification value: obj.frozen may itself be the edit.
        history = state.attrs["frozen"].history
        was_frozen = history.deleted[0] if history.deleted else obj.frozen
        if not was_frozen:
            continue

        changed = {
            name
            for name in state.attrs.keys()
            if state.attrs[name].history.has_changes()
        } - _FROZEN_EDITABLE
        if changed:
            raise PermissionError(
                f"FOM definition {obj.name!r} v{obj.version} is frozen; refusing to change "
                f"{sorted(changed)}. FOM_PROOF Sec. 5.3: bounds, transform, direction, floor, "
                "and weights are part of the score's identity, so every score already published "
                "against this version would silently change meaning. Create a new version instead."
            )

    for obj in session.deleted:
        if isinstance(obj, FomDefinition) and obj.frozen:
            raise PermissionError(
                f"FOM definition {obj.name!r} v{obj.version} is frozen and cannot be deleted; "
                "scores reference it as the record of how they were computed."
            )
