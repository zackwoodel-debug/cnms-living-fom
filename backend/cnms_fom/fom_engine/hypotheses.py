"""Pre-registered directional hypotheses (FOM_PROOF Sec. 4.2, Table 3).

    "The sign hypotheses in Table 3 are declared before calculating correlations.
     Results that contradict the prediction are reported as contradictions rather
     than retroactively explained away."

Pre-registration only means something if the declaration is fixed and
checkable, so this module exposes ``registry_fingerprint()``: a hash of the
table that gets stamped onto every analysis run.  If someone edits a predicted
sign after seeing results, the fingerprint on the new run will not match the one
on the old, and the change is visible in git.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

#  "+" expects positive, "-" expects negative, "test" is pre-registered as
#  genuinely open — no directional claim is made or permitted.
Sign = str


@dataclass(frozen=True)
class Hypothesis:
    x_key: str
    y_key: str
    expected_sign: Sign
    mechanism: str
    conditional_on: tuple[str, ...] = ()
    scope: str = ""


#  FOM_PROOF Table 3, transcribed.  Add rows by PR with the mechanism stated;
#  never by editing an existing sign after an analysis has been run.
PREREGISTERED: tuple[Hypothesis, ...] = (
    Hypothesis(
        x_key="Z_RMS_star",
        y_key="eps_ionic",
        expected_sign="+",
        mechanism="Ionic response increases approximately with the square of effective charge.",
    ),
    Hypothesis(
        x_key="omega_TO_min",
        y_key="eps_ionic",
        expected_sign="-",
        mechanism="Softer polar modes yield larger lattice response.",
    ),
    Hypothesis(
        x_key="inv_omega_TO_min",
        y_key="eps_ionic",
        expected_sign="+",
        mechanism="Equivalent expression of the same mechanism using a softness descriptor.",
    ),
    Hypothesis(
        x_key="V_fu",
        y_key="eps_ionic",
        expected_sign="-",
        mechanism="Greater volume reduces polarization density.",
        conditional_on=("Z_RMS_star", "omega_TO_min"),
    ),
    Hypothesis(
        x_key="k",
        y_key="Eg",
        expected_sign="test",
        mechanism=(
            "May be negative for high-polarizability oxides, but is not assumed universal. "
            "Test empirically within a chemically controlled family."
        ),
        scope="chemically controlled family",
    ),
    Hypothesis(
        x_key="k",
        y_key="Ebd",
        expected_sign="test",
        mechanism=(
            "A trade-off may occur, but theory cannot replace measured breakdown values. "
            "Test empirically within a matched specimen context."
        ),
        scope="matched specimen context",
    ),
)


def hypothesis_map() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(x_key, y_key): (expected_sign, mechanism)}`` for the correlation engine.

    Registered in both orders: a correlation is symmetric, and which variable a
    caller passes first is an implementation detail, not a scientific claim.
    """
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for h in PREREGISTERED:
        out[(h.x_key, h.y_key)] = (h.expected_sign, h.mechanism)
        out[(h.y_key, h.x_key)] = (h.expected_sign, h.mechanism)
    return out


def registry_fingerprint() -> str:
    """Stable SHA-256 over the pre-registered table, for stamping on a run."""
    payload = json.dumps([asdict(h) for h in PREREGISTERED], sort_keys=True, default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def as_records() -> list[dict]:
    """Serialisable form for the API and the released report."""
    return [
        {**asdict(h), "conditional_on": list(h.conditional_on)} for h in PREREGISTERED
    ]
