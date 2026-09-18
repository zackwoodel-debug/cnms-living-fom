"""Canonical measurement-context identity.

FOM_PROOF Eq. (3) makes the unit of analysis

    (composition, polymorph, specimen form, temperature, direction, method)

Composition, polymorph, and specimen form live on ``Material``. The rest lives
across ~20 columns of ``PropertyValue``, which leaves the database unable to
answer the one question the protocol cares about most: *are these two rows the
same measurement?*

This module answers it. ``context_digest`` reduces the context-defining columns
to a stable 32-character fingerprint, and ``PropertyValue`` carries it in an
indexed column with a uniqueness constraint. Two rows with the same
(material, property, digest) are the same measurement reported twice — a true
duplicate, which the database now rejects.

What is deliberately *not* in the digest is as important as what is:

  * ``value`` and ``uncertainty`` — a second reading of the same context is a
    duplicate, not a new record. If they disagree, that is a data-quality
    problem to resolve, not two rows to average.
  * source identity (``doi``, ``database_identifier``, ``source_locator``) **is**
    included. Two papers reporting the same quantity under the same conditions
    are independent measurements, and Sec. 2.1 forbids merging them without an
    explicit aggregation rule — it does not forbid *storing* both. Keeping both
    preserves the evidence, and ``eligibility.build_analysis_table`` surfaces the
    ambiguity at analysis time instead of the database silently picking one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#  Columns that define *which measurement this is*. Order is irrelevant (the
#  payload is sorted) but the membership of this tuple is part of the schema:
#  changing it changes every digest, so it needs a migration that rebuilds them.
CONTEXT_FIELDS: tuple[str, ...] = (
    # exact quantity (Sec. 3.2 — a tensor component is not an isotropic average)
    "tensor_component",
    "direction",
    "reduction_rule",
    # experimental context (Table 1)
    "temperature_k",
    "frequency_hz",
    "field_amplitude_v_per_cm",
    "thickness_nm",
    "electrode",
    "substrate",
    "interface",
    "area_cm2",
    "failure_criterion",
    "processing_route",
    # calculation context (Table 1)
    "method",
    "software",
    "xc_functional",
    "pseudopotential",
    # provenance — measured and calculated values are never the same record
    "provenance_tier",
    "doi",
    "database_identifier",
    "source_locator",
)

#  Significant figures retained for floats. Guards against a temperature of
#  300.0 and 300.00000000000006 fingerprinting as different measurements
#  because of float round-tripping, without merging 300 K and 301 K.
_FLOAT_PRECISION = 12


def _canonical(value: Any) -> Any:
    """Normalise one field value so equivalent inputs hash identically."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value:  # NaN is not a context, it is missing data
            return None
        return float(f"{value:.{_FLOAT_PRECISION}g}")
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}

    #  Enum members (ProvenanceTier and friends) reduce to their value.
    inner = getattr(value, "value", value)
    text = str(inner).strip()
    #  Empty string and NULL both mean "not recorded" and must not split a group.
    return text or None


def context_payload(source: Any) -> dict[str, Any]:
    """Extract the canonical context dict from an ORM row or a plain mapping."""
    getter = source.get if isinstance(source, dict) else lambda k, d=None: getattr(source, k, d)
    return {field: _canonical(getter(field)) for field in CONTEXT_FIELDS}


def context_digest(source: Any) -> str:
    """Stable 32-hex-character fingerprint of a measurement context.

    Accepts an ORM ``PropertyValue`` or a dict with the same field names, so
    ingestion scripts can compute a digest before building the row.
    """
    payload = json.dumps(context_payload(source), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def describe_context(source: Any) -> str:
    """Human-readable one-line context summary, for error messages and audits.

    A duplicate-key violation that says only "digest abc123 already exists" is
    useless to whoever has to fix the data.
    """
    payload = context_payload(source)
    parts = [f"{key}={value!r}" for key, value in payload.items() if value is not None]
    return ", ".join(parts) if parts else "no context recorded"
