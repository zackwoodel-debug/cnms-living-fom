"""The canonical envelope this platform accepts pySEA records in.

**This is our contract, not pySEA's.** pySEA (Walker, Pfeifer, Lupini, Hachtel,
Pantelides, Hoglund, M&M 2026) is described in its abstracts as three packages: a
FAIR data architecture for signal storage and calibration, a ray-optics digital
twin of the microscope, and a multislice framework for elastic and inelastic
scattering with MLIP-driven molecular dynamics for vEELS prediction via TACAW. We
have read those abstracts. We have not seen the container format, the field names,
or the API.

So this module defines a shape *we* can validate and promote from, and a boundary
we can re-map when the real format arrives. Everything here is provisional:

* ``CONTRACT_VERSION`` is stamped on every row, so rows written under a guessed
  mapping stay findable after the guess is corrected.
* Unknown keys are preserved under ``extensions`` rather than dropped, because the
  fields we failed to anticipate are the ones worth keeping.
* No field name below should be quoted to the pySEA team as something we require
  of them. They are names we chose; the mapping onto theirs is future work.

What is *not* provisional is the discipline. A record says whether it came off an
instrument or out of a simulation, in a field, not by inference. A scalar names the
signals it came from. Absent stays absent: this module never fills a missing unit,
angle or uncertainty with a nominal value.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#  Bumped when the shape changes in a way that makes an older envelope
#  un-ingestable. The minor version moves for additive changes.
CONTRACT_VERSION = "pysea-canonical/0.1"
SUPPORTED_MAJOR = "pysea-canonical/0"

RECORD_KINDS: frozenset[str] = frozenset({"experimental", "simulation", "hybrid"})
DERIVATIONS: frozenset[str] = frozenset({"measured", "fitted", "calculated", "simulated"})
AXIS_KINDS: frozenset[str] = frozenset({"spatial", "energy", "momentum", "time", "other"})

#  Envelope sections we know about. Anything else a container carries is kept
#  under "extensions" so that an unmapped pySEA field survives ingestion and can
#  be found later by whoever does the real mapping.
KNOWN_SECTIONS: frozenset[str] = frozenset({
    "contract_version",
    "record_id",
    "record_kind",
    "acquired_at",
    "operator",
    "sample",
    "proposal",
    "instrument",
    "instrument_state",
    "calibrations",
    "signals",
    "simulation",
    "derived_scalars",
    "software",
    "datafed",
    "extensions",
})


def enum_value(value) -> str:
    """The stored string for a column that may come back as a str-Enum member.

    ``PySeaRecordKind(str, Enum)`` compares equal to ``"experimental"`` but formats
    as ``PySeaRecordKind.EXPERIMENTAL``, so comparisons were right while every
    label and JSON payload showed the member name. One helper rather than a
    ``.value`` at each site, because the next person to format one of these will
    not remember.
    """
    return getattr(value, "value", value)


class ContractError(ValueError):
    """The envelope cannot be read under any supported version of this contract."""


def supported_version(version: str | None) -> bool:
    """Whether this build can read an envelope stamped ``version``.

    Major version only. ``pysea-canonical/0.2`` is readable by a 0.1 build because
    0.x changes are additive by construction; ``pysea-canonical/1.0`` is not, and
    guessing at it would mean interpreting fields that may have been redefined.
    """
    if not version or not isinstance(version, str):
        return False
    return version.startswith(SUPPORTED_MAJOR + ".")


def assert_supported(envelope: dict) -> str:
    """Return the envelope's contract version, or raise.

    Raises before anything else looks at the payload. A 1.x envelope read by a 0.x
    parser would produce plausible-looking rows from fields that mean something
    else, which is worse than refusing.
    """
    version = envelope.get("contract_version")
    if not supported_version(version):
        raise ContractError(
            f"Unsupported contract version {version!r}. This build reads "
            f"{SUPPORTED_MAJOR}.x only. The envelope was not parsed: reading it under the "
            "wrong version would interpret fields that may have been redefined."
        )
    return str(version)


def _canonical(value: Any) -> Any:
    """A JSON-serialisable form with deterministic ordering.

    Dict keys sorted at every depth, so the same record exported twice with
    different key order hashes the same. Lists keep their order: axis order *is*
    the signal's dimension order, and sorting it would destroy the meaning.
    """
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    #  Datetimes, Decimals, numpy scalars. str() rather than a silent drop, so the
    #  hash still covers the field.
    return str(value)


#  Keys whose values are bulk arrays rather than metadata. Stripped before
#  storage and before hashing: a spectrum's counts belong in DataFed, and
#  including megabytes of them in a content hash makes re-import fragile against
#  a float being re-serialised one ulp differently.
BULK_KEYS: frozenset[str] = frozenset({"data", "array", "values", "counts", "intensity"})


def strip_bulk(value: Any) -> Any:
    """``value`` with bulk array payloads removed, recursively.

    The reference survives, the array does not. A caller that wants the data
    fetches it through ``data_ref``.
    """
    if isinstance(value, dict):
        return {
            key: strip_bulk(inner)
            for key, inner in value.items()
            if key not in BULK_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [strip_bulk(item) for item in value]
    return value


def canonical_envelope(envelope: dict) -> dict:
    """The envelope in canonical form: bulk stripped, keys sorted, JSON-safe."""
    return _canonical(strip_bulk(envelope))


def content_sha256(envelope: dict) -> str:
    """Content hash over the canonical envelope.

    Canonical form rather than file bytes, matching ``modalfit.records``: the same
    container re-exported with different indentation is the same container, and
    treating it as new would put two copies of one acquisition in the table.
    """
    blob = json.dumps(
        canonical_envelope(envelope), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def split_extensions(envelope: dict) -> dict:
    """Move unrecognised top-level keys into ``extensions``.

    A pySEA container will carry fields this contract has not anticipated. Dropping
    them would lose exactly the information needed to finish the mapping, so they
    are collected rather than discarded, and nothing downstream reads them.
    """
    known = {key: value for key, value in envelope.items() if key in KNOWN_SECTIONS}
    unknown = {key: value for key, value in envelope.items() if key not in KNOWN_SECTIONS}
    if unknown:
        extensions = dict(known.get("extensions") or {})
        extensions.setdefault("_unmapped", {}).update(unknown)
        known["extensions"] = extensions
    return known


def load_envelope(source: dict | str) -> dict:
    """An envelope from a dict or a path to a JSON file."""
    if isinstance(source, dict):
        envelope = source
    else:
        from pathlib import Path

        text = Path(source).read_text(encoding="utf-8")
        try:
            envelope = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(envelope, dict):
        raise ContractError("A pySEA envelope must be a JSON object at the top level.")
    return envelope
