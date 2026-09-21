"""The typed shapes the research loop passes around.

Dataclasses rather than Pydantic models, and rather than ORM rows, because these
are the *domain* contracts: they are built by ``brief`` from tool output, validated
here, and only then persisted or serialised.  Keeping them free of SQLAlchemy means
the invariants below are testable without a database, which is the same reason
``fom_engine`` depends on nothing heavier than numpy.

Three invariants are enforced by construction, not by convention:

``an extracted claim cannot present itself as a measurement``
    :class:`ExtractedClaim` has no field that could be read as an analysis-table
    tier.  ``tier`` is a :class:`ClaimTier` — what the *source* said about its own
    number — and ``status`` has no ``accepted`` member.  There is no method on this
    class that produces a ``PropertyValue``.

``a claim without evidence is not a claim``
    :meth:`ExtractedClaim.__post_init__` refuses one with no
    :class:`EvidenceItem`, and an evidence item refuses to exist without a document
    hash, a page, and the quote it rests on.  FOM_PROOF Sec. 2.2 wants provenance a
    reader can follow; an extraction that cannot be checked against its page is
    worse than no extraction, because it looks the same as a good one.

``a proposed context change cannot loosen anything``
    :meth:`ProposedBOContext.widening_violations` reports every bound that would
    widen a live campaign's. The bridge refuses on a non-empty list. A literature
    claim may narrow a search space — that is a scientist choosing to trust a
    paper — but widening one past its instrument envelope is a physical claim about
    a tool, which no paper can make.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cnms_fom.db.enums import (
    BriefStatus,
    ClaimStatus,
    ClaimTier,
    ContextStatus,
    StatementKind,
)


class ResearchContractError(ValueError):
    """A research object cannot be built as described."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """One retrieved passage, with everything needed to find it again.

    ``content_sha256`` is here as well as ``document_id`` on purpose: an id is
    only meaningful inside one database, and a brief outlives the row it was built
    from. The hash identifies the document itself, so a claim remains checkable
    after a re-ingest renumbers everything.
    """

    document_id: int | None
    document_title: str
    page: int | None
    quote: str
    content_sha256: str | None = None
    chunk_id: int | None = None
    doi: str | None = None
    source_url: str | None = None
    technique: str | None = None
    #  How this passage was found: "vector", "lexical", "vector+lexical", "card",
    #  "record". Kept because a passage only lexical search could reach says
    #  something about the query that a relevance grade does not.
    retrieval_method: str = ""
    retrieval_rank: int | None = None
    #  The 0-3 relevance grade, when grading ran.
    grade: int | None = None
    grade_reason: str = ""

    def __post_init__(self) -> None:
        if not (self.document_title or "").strip():
            raise ResearchContractError("An evidence item needs a document title.")
        if not (self.quote or "").strip():
            raise ResearchContractError(
                "An evidence item needs the quote it rests on. A citation without the "
                "supporting text cannot be checked against its page, which makes a wrong "
                "extraction indistinguishable from a right one."
            )

    @property
    def citation(self) -> str:
        parts = [self.document_title]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.doi:
            parts.append(f"doi:{self.doi}")
        return ", ".join(parts)

    @property
    def is_locatable(self) -> bool:
        """Whether a reader could actually go and check this.

        A page number is the minimum. A title alone points at a document and not
        at a claim, and a document is where a disagreement hides.
        """
        return self.page is not None and bool(self.document_id or self.content_sha256)

    def as_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "content_sha256": self.content_sha256,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "doi": self.doi,
            "source_url": self.source_url,
            "technique": self.technique,
            "retrieval_method": self.retrieval_method,
            "retrieval_rank": self.retrieval_rank,
            "grade": self.grade,
            "grade_reason": self.grade_reason,
            "citation": self.citation,
            "locatable": self.is_locatable,
        }


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

#  Context fields a claim carries when the source records them. Mirrors the
#  PropertyValue columns of FOM_PROOF Table 1, because a claim that is ever going
#  to be compared with a stored value has to be describable in the same terms.
CLAIM_CONTEXT_FIELDS: tuple[str, ...] = (
    "material",
    "polymorph",
    "specimen_form",
    "technique",
    "substrate",
    "electrode",
    "interface",
    "temperature_k",
    #  Sources state growth temperatures in Celsius almost without exception. The
    #  extractor records what the source wrote and never converts; the conversion to
    #  temperature_k happens here, in code that can be audited and cannot arithmetic
    #  its way to a plausible-looking wrong number.
    "temperature_c",
    "pressure_torr",
    "frequency_hz",
    "thickness_nm",
    "precursor",
    "oxidant",
    "chamber",
    "anneal",
    "failure_criterion",
    "method",
    "software",
    "xc_functional",
)

#  Context fields whose name declares a unit. A value in one of these has to be a
#  number in *that* unit or the field is lying: a ``temperature_k`` holding
#  "200 to 300 degC" would be read as 200-300 kelvin by anything that trusted the
#  name, and a source stating a range in Celsius is exactly what an extractor
#  produces. Observed from a live extraction, hence the guard.
UNIT_BEARING_CONTEXT: dict[str, str] = {
    "temperature_k": "kelvin",
    "temperature_c": "degrees Celsius",
    "pressure_torr": "torr",
    "frequency_hz": "hertz",
    "thickness_nm": "nanometres",
    "area_cm2": "square centimetres",
}

#  The physical dimension each registry key must have, and how a unit string is
#  classified into one. This exists because of a real fabrication: asked a compound
#  question, the extractor filed a passage's ALD cycle timings — "0.2 s TDMAH dose,
#  6 s purge" — as four ``growth_per_cycle_ang`` claims, and the narrative then
#  reported "growth per cycle values vary widely (0.2 s, 6.0 s, 0.1 s, 6.0 s),
#  indicating a lack of consistency in the literature". A dose time is not a growth
#  rate, and a field named ``_ang`` holding seconds is not a value that should reach
#  a reader at all.
#
#  Deliberately a dimension check rather than a list of acceptable spellings: real
#  sources write "A/cycle", "Å/cy", "Ang per cycle" and worse, and rejecting a
#  legitimate value for an unrecognised spelling would lose data. So a unit is only
#  rejected when its dimension is *recognised* and wrong. An unclassifiable unit is
#  kept — act on knowledge, abstain on ignorance.
_DIMENSION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    #  Order matters: the compound forms must be tested before their parts, or
    #  "angstrom per cycle" classifies as a bare length.
    ("growth_per_cycle", ("/cycle", "percycle", "/cy", "/pulse")),
    ("rate", ("/s", "/sec", "/min", "/hour", "/h")),
    ("density", ("g/cm3", "g/cc", "kg/m3", "gcm-3")),
    ("sld", ("a-2", "a^-2", "1/a2", "a**-2")),
    ("energy", ("ev", "mev", "kev", "joule", "kj/mol")),
    ("field", ("mv/cm", "kv/cm", "v/nm", "kv/mm", "v/m")),
    ("thermal_conductivity", ("w/mk", "w/m-k", "w/(mk)", "w/cmk")),
    ("time", ("s", "sec", "secs", "second", "seconds", "ms", "min", "minute",
              "minutes", "h", "hr", "hour", "hours", "cycle", "cycles")),
    ("temperature", ("degc", "c", "k", "degk", "celsius", "kelvin", "degf")),
    ("pressure", ("torr", "mtorr", "pa", "kpa", "mbar", "bar", "atm", "psi")),
    ("length", ("a", "angstrom", "angstroms", "nm", "um", "micron", "mm", "cm", "m")),
)

#  The dimension a registry key's units must have. A key absent here is unconstrained,
#  which is the right default for a descriptive name the extractor invented.
FIELD_DIMENSION: dict[str, str] = {
    "growth_per_cycle_ang": "growth_per_cycle",
    "growth_rate_nm_min": "rate",
    "rho": "density",
    "sld_xray": "sld",
    "sld_neutron": "sld",
    "Eg": "energy",
    "dEc": "energy",
    "Ebd": "field",
    "kappa_th": "thermal_conductivity",
    "thickness_nm": "length",
    "roughness_ang": "length",
    "temperature_c": "temperature",
}

#  A registry key whose *name* declares a unit, and the factor that converts each
#  accepted spelling into it. Keys are the normalised form produced by
#  ``_normalised_unit`` below.
#
#  This exists because the dimensional guard checks dimension and never magnitude, so a
#  field named ``_nm`` could hold a value in angstrom and still be marked comparable.
#  Observed on the real corpus: the extractor emitted
#  ``{"field": "thickness_nm", "value": 12, "units": "nm"}`` from the quote "the
#  interfacial oxide measured 12 angstrom". 12 angstrom is 1.2 nm, so the stored number
#  was ten times the truth and flagged ``ok``. A legitimate "0.098 nm/cycle" has the same
#  shape: it is exactly 0.98 A/cycle, and comparing 0.098 against 1.42 would report a
#  93% disagreement where the real one is 31%.
#
#  Code does the arithmetic, never the model — the same division of labour as
#  ``_derive_kelvin_from_celsius``. Factors are exact where the definition is exact and
#  are the conventional values otherwise.
CANONICAL_UNIT: dict[str, str] = {
    "growth_per_cycle_ang": "a/cycle",
    "growth_rate_nm_min": "nm/min",
    "thickness_nm": "nm",
    "roughness_ang": "a",
    "pressure_torr": "torr",
    "temperature_c": "degc",
}

UNIT_FACTORS: dict[str, dict[str, float]] = {
    "a/cycle": {"a/cycle": 1.0, "nm/cycle": 10.0, "pm/cycle": 0.01},
    #  1 A/s = 0.1 nm/s = 6 nm/min.
    "nm/min": {"nm/min": 1.0, "nm/s": 60.0, "a/s": 6.0, "a/min": 0.1, "um/min": 1000.0},
    "nm": {"nm": 1.0, "a": 0.1, "um": 1000.0, "mm": 1e6, "cm": 1e7, "m": 1e9, "pm": 0.001},
    "a": {"a": 1.0, "nm": 10.0, "pm": 0.01, "um": 10_000.0},
    #  760 torr = 1 atm exactly; 1 torr = 101325/760 Pa.
    "torr": {
        "torr": 1.0, "mtorr": 0.001, "pa": 760.0 / 101_325.0,
        "kpa": 760_000.0 / 101_325.0, "mbar": 76.0 / 101.325,
        "bar": 76_000.0 / 101.325, "atm": 760.0, "psi": 760.0 / 14.695_948_775_5,
    },
    #  Temperature is an offset scale, so it is handled in code rather than by a factor.
    "degc": {"degc": 1.0},
}

#  Context fields that name a category, not a quantity. A *numeric* claim filed under
#  one of these is the extractor confusing the field slot with the context slot, and it
#  is not harmless: an extraction of ``material = 2`` was read by the interpretation
#  step as "a growth per cycle of 2.0 was reported for the hot-wall chamber", inventing
#  a disagreement between sources out of a category label.
#
#  Only the text-valued context fields are listed. ``thickness_nm``, ``temperature_c``,
#  ``pressure_torr`` and ``frequency_hz`` are legitimately both context and claim, so
#  they must not appear here.
CATEGORICAL_CONTEXT_FIELDS: frozenset[str] = frozenset({
    "material", "polymorph", "specimen_form", "technique", "substrate", "electrode",
    "interface", "precursor", "oxidant", "chamber", "anneal", "failure_criterion",
    "method", "software", "xc_functional",
})

#  Dimensionless by definition: a number with any unit attached is suspect.
DIMENSIONLESS_FIELDS: frozenset[str] = frozenset(
    {"k", "eps_inf", "eps_ionic", "tan_delta"}
)


#  Wording that makes a value_text a range, a bound or an estimate rather than a single
#  number. Any of these present means no scalar is recovered: inventing 0.98 from
#  "0.9 to 1.1" or from "< 1.0" would be a fabrication with a citation attached.
_NOT_A_SCALAR = (
    " to ", "\u2013", "\u2014", "\u00b1", "+/-", "between", "<", ">", "~",
    "approx", "approximately", " or ", "least", "most", "above", "below", "over",
    "under", "up to", "range",
)
#  A number, allowing a leading sign, decimals and exponents — but never a digit that
#  belongs to a chemical formula. The lookbehind is load-bearing: without it "HfO2"
#  yields the single number 2, and recovering that for a `material` claim manufactures
#  exactly the `material = 2` fabrication that §10d's bug 17 was written to stop. A
#  trailing letter is still allowed so "12nm" reads as 12.
_NUMERIC_TOKEN = re.compile(
    r"(?<![A-Za-z0-9.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
#  A hyphen *between digits* is a range ("0.9-1.1"); a leading one is a sign, and one
#  inside a word is part of a name like Nevot-Croce.
_DIGIT_RANGE = re.compile(r"\d\s*-\s*\d")


def convert_to_canonical(
    field_name: str, value: float | None, units: str | None
) -> tuple[float | None, str | None, bool]:
    """``(value, units)`` expressed in the unit ``field_name`` declares.

    Returns ``(value, units, converted)``. ``converted`` is False both when nothing
    needed doing and when the units are the right dimension but no conversion is known —
    the caller distinguishes those, because only the second is a reason to refuse a
    comparison.

    Shared by ``ExtractedClaim`` and by the benchmark's matcher on purpose. A gold value
    written in the source's units ("oxygen_pressure 100 mTorr") has to be compared
    against a claim already normalised to torr, and two copies of this arithmetic would
    eventually disagree.
    """
    canonical = CANONICAL_UNIT.get(field_name)
    if canonical is None or value is None:
        return value, units, False
    stated = _normalised_unit(units)
    if not stated or stated == canonical:
        return value, units, False
    if canonical == "degc" and stated in {"k", "degk", "kelvin"}:
        return round(value - 273.15, 4), "degC", True
    factor = UNIT_FACTORS.get(canonical, {}).get(stated)
    if factor is None:
        return value, units, False
    return value * factor, canonical, True


def _normalised_unit(units: str | None) -> str:
    """A unit string reduced to the spelling the factor tables are keyed on."""
    text = (units or "").strip().lower().replace(" per ", "/").replace("\u00b7", "")
    text = "".join(text.split())
    text = text.lstrip("0123456789.,+-\u00b1")
    for variant in ("\u00e5ngstr\u00f6m", "\u00c5", "\u212b", "angstroms", "angstrom"):
        text = text.replace(variant.lower(), "a")
    text = text.replace("\u00b0c", "degc").replace("celsius", "degc")
    text = text.replace("degreec", "degc").replace("degc.", "degc")
    text = text.replace("micron", "um").replace("\u00b5", "u")
    text = text.replace("/cy", "/cycle").replace("/cyclecle", "/cycle")
    text = text.replace("/sec", "/s").replace("/minute", "/min").replace("/hour", "/h")
    return text


def classify_unit(units: str | None) -> str | None:
    """The physical dimension a unit string denotes, or None if unrecognised.

    A compound marker (one containing "/") matches as a substring, because the unit
    may carry a prefix: "1.42 A/cycle" is still a growth per cycle. A bare marker must
    match the *whole* remaining unit, because substring matching on short tokens is
    indefensible — "s" for seconds appears inside "angstrom", which classified a length
    as a time and would have rejected the very claims this guard exists to protect.
    """
    text = (units or "").strip().lower().replace(" per ", "/").replace("·", "")
    text = "".join(text.split())
    #  Drop a leading magnitude so "0.2s" and "6s" reduce to the unit itself.
    text = text.lstrip("0123456789.,+-±")
    if not text:
        return None
    for dimension, markers in _DIMENSION_PATTERNS:
        for marker in markers:
            compound = "/" in marker
            if (marker in text) if compound else (text == marker):
                return dimension
    return None


#  Which context a claim needs before it is worth comparing with anything, per
#  property key. Absence is reported, never filled in: Sec. 16 makes context part
#  of what a value *is*, so a permittivity with no frequency is not a permittivity
#  with a gap, it is an incomparable number.
REQUIRED_CONTEXT: dict[str, tuple[str, ...]] = {
    "k": ("temperature_k", "frequency_hz"),
    "eps_inf": ("temperature_k",),
    "eps_ionic": ("temperature_k",),
    "Eg": ("method",),
    "dEc": ("substrate", "method"),
    "Ebd": ("thickness_nm", "electrode", "failure_criterion"),
    "tan_delta": ("temperature_k", "frequency_hz"),
    "kappa_th": ("temperature_k",),
    "sld_xray": ("method",),
    "sld_neutron": ("method",),
    "growth_per_cycle_ang": ("technique", "temperature_k", "precursor", "chamber"),
    "growth_rate_nm_min": ("technique", "temperature_k", "chamber"),
}


@dataclass
class ExtractedClaim:
    """One number (or statement) pulled out of a document, with its context.

    Not a measurement. Not a candidate for arithmetic. A record that a source said
    something, wired to the page where it said it.
    """

    #  A registry key when it maps onto one ("k", "Eg", "sld_xray"), otherwise a
    #  descriptive name. Kept as free text rather than constrained to the registry
    #  because a paper's most useful number is often one this platform has no key
    #  for yet, and refusing to record it would lose it.
    field_name: str
    evidence: list[EvidenceItem]
    value: float | None = None
    units: str | None = None
    value_text: str | None = None
    #  Only set when a conversion was performed, and then always alongside the
    #  original. A normalised value with no record of what it was normalised from
    #  is a number nobody can audit.
    normalized_value: float | None = None
    normalized_units: str | None = None
    normalization_note: str = ""
    tier: ClaimTier = ClaimTier.REPORTED
    status: ClaimStatus = ClaimStatus.CANDIDATE
    context: dict[str, Any] = field(default_factory=dict)
    #  The extracting model's own confidence. A model output, recorded as one, and
    #  never a substitute for review: a confident extraction of a misread table is
    #  the failure mode this field must not be allowed to mask.
    model_confidence: float | None = None
    extracted_by_model: str = ""
    extracted_by_provider: str = ""
    prompt_version: str = ""
    extracted_at: datetime = field(default_factory=_utcnow)
    notes: str = ""
    #  True when the units are the right *kind* of quantity for this field but their
    #  magnitude relative to the unit the field name declares could not be established.
    #  Such a value is readable and not comparable: the number's scale is unknown.
    magnitude_unverified: bool = False

    def __post_init__(self) -> None:
        if not (self.field_name or "").strip():
            raise ResearchContractError("A claim needs a field name.")
        if not self.evidence:
            raise ResearchContractError(
                f"Claim {self.field_name!r} has no evidence. An extraction with no page behind "
                "it cannot be checked, and an unverifiable claim looks exactly like a verified "
                "one once it is a number in a table (FOM_PROOF Sec. 2.2)."
            )
        if self.value is None and not (self.value_text or "").strip():
            raise ResearchContractError(
                f"Claim {self.field_name!r} has neither a numeric value nor a text value."
            )
        if self.normalized_value is not None and not self.normalization_note:
            raise ResearchContractError(
                f"Claim {self.field_name!r} carries a normalised value with no note saying how it "
                "was converted. An unexplained conversion is not auditable."
            )
        unknown = set(self.context) - set(CLAIM_CONTEXT_FIELDS)
        if unknown:
            #  Kept rather than dropped, but flagged: an unrecognised context key
            #  is usually a typo that would silently stop matching.
            self.notes = (
                f"{self.notes} [unrecognised context keys: {sorted(unknown)}]".strip()
            )
        self._recover_scalar_from_value_text()
        self._reject_dimensionally_impossible_units()
        self._reconcile_units_with_the_quote()
        self._normalise_to_the_unit_its_name_declares()
        self._quarantine_mislabelled_units()
        self._derive_kelvin_from_celsius()

    def _reconcile_units_with_the_quote(self) -> None:
        """When the declared units contradict the claim's own quote, the quote wins.

        Observed on the real corpus: the extractor emitted
        ``{"field": "thickness_nm", "value": 12, "units": "nm"}`` from the quote **"the
        interfacial oxide measured 12 angstrom"**. Nothing caught it, because the
        declared unit and the field name agreed with each other — they were just both
        wrong about the source. 12 angstrom is 1.2 nm, and the claim read ``ok``.

        The quote is already verified to appear verbatim in the passage, so it is the
        better authority than a units field the model filled in separately. Deliberately
        narrow: it acts only when the value appears exactly once in the quote and is
        directly followed by a unit of the same dimension that is convertible. A quote
        holding the number twice, or an unrecognised trailing token, is left alone.
        """
        if self.value is None or not self.evidence:
            return
        quote = (self.evidence[0].quote or "").strip()
        declared = _normalised_unit(self.units)
        if not quote or not declared:
            return

        rendered = f"{self.value:g}"
        if quote.count(rendered) != 1:
            #  Ambiguous or absent: two occurrences give two candidate units, and a
            #  whole-passage quote (used when the model's own quote failed verification)
            #  routinely has both.
            return
        after = quote[quote.index(rendered) + len(rendered):]
        match = re.match(r"\s*([A-Za-z\u00c5\u212b\u00b0][A-Za-z\u00c5\u212b\u00b0/.^\-]*"
                         r"(?:\s+per\s+\w+|/\w+)?)", after)
        if not match:
            return
        from_quote = _normalised_unit(match.group(1))
        if not from_quote or from_quote == declared:
            return

        #  Only when both are the same kind of quantity and the source's unit is one this
        #  code can convert. Anything else is a disagreement to report, not to resolve.
        canonical = CANONICAL_UNIT.get(self.field_name)
        if canonical is None or from_quote not in UNIT_FACTORS.get(canonical, {}):
            return
        if classify_unit(from_quote) != classify_unit(declared):
            return

        self.notes = (
            f"{self.notes} [declared units {self.units!r} contradict the quote, which "
            f"reads {rendered} {match.group(1).strip()!r}; the quote was taken as "
            f"authoritative]"
        ).strip()
        self.units = match.group(1).strip()

    def _normalise_to_the_unit_its_name_declares(self) -> None:
        """Convert a value into the unit the field name declares, or refuse to compare it.

        A field called ``thickness_nm`` holding 12 with units "angstrom" is not a unit
        problem, it is a *wrong number*: 12 angstrom is 1.2 nm, and the dimensional guard
        passes it because both are lengths. Three outcomes, and no silent fourth:

        * already canonical -> untouched;
        * a known variant -> converted here, in code, with the original recorded;
        * the right dimension but an unrecognised spelling -> kept, flagged, and
          **not comparable**, because the alternative is trusting a number whose scale
          nobody has established.

        ``units`` becomes the canonical spelling so the claim is internally consistent.
        The quote and the note preserve what the source actually wrote.
        """
        canonical = CANONICAL_UNIT.get(self.field_name)
        if canonical is None or self.value is None:
            return
        stated = _normalised_unit(self.units)
        if not stated or stated == canonical:
            return

        original, original_units = self.value, self.units
        value, units, converted = convert_to_canonical(
            self.field_name, self.value, self.units
        )
        if converted:
            self.value, self.units = value, units
            self.notes = (
                f"{self.notes} [converted {original} {original_units!r} to {self.value} "
                f"{self.units!r}, the unit {self.field_name} declares]"
            ).strip()
            return

        #  Same dimension (the guard already checked) but an unknown spelling, so the
        #  magnitude cannot be established. Reported, not guessed at.
        self.magnitude_unverified = True
        self.notes = (
            f"{self.notes} [units {self.units!r} are the right kind of quantity for "
            f"{self.field_name}, which declares {canonical!r}, but no conversion is "
            f"known; the value has NOT been rescaled and cannot be compared]"
        ).strip()

    def _recover_scalar_from_value_text(self) -> None:
        """Fill ``value`` from a ``value_text`` that holds exactly one number.

        Observed on a real extraction: the model emitted
        ``{"value": null, "value_text": "0.98 angstrom per cycle"}``. A claim with no
        numeric ``value`` is invisible to ``is_comparable``, to the benchmark's matcher
        and to contradiction detection, so the number was read from the source and then
        silently discarded.

        Deliberately narrow. The scalar is recovered only when the text is unambiguously
        one number with a unit that can belong to this field; a range, a bound, an
        estimate or two numbers leaves ``value`` as ``None``, because a citation
        attached to a number nobody wrote is worse than a missing value.
        ``value_text`` and ``units`` are both preserved exactly as the model gave them.
        """
        if self.value is not None or not (self.value_text or "").strip():
            return
        value_text = self.value_text or ""
        if self.field_name in CATEGORICAL_CONTEXT_FIELDS:
            #  A category can never carry a number, so there is nothing here to
            #  recover and every candidate is a misreading. Belt and braces with the
            #  formula lookbehind above: either alone would have let `material = 2`
            #  through from "monoclinic HfO2".
            return
        text = value_text.strip()
        lowered = f" {text.lower()} "
        if any(marker in lowered for marker in _NOT_A_SCALAR):
            return
        if _DIGIT_RANGE.search(text):
            return
        numbers = _NUMERIC_TOKEN.findall(text)
        if len(numbers) != 1:
            return

        #  The unit to check: what the model declared, else whatever follows the number.
        remainder = _NUMERIC_TOKEN.sub("", text, count=1).strip()
        unit_text = (self.units or "").strip() or remainder
        expected = FIELD_DIMENSION.get(self.field_name)
        if expected is not None and classify_unit(unit_text) != expected:
            #  Either the unit is wrong for this field or it is unrecognised. Recovery
            #  *adds* a number, so it only happens where the unit can be verified —
            #  stricter than the rejection guard, which abstains on ignorance.
            return
        if self.field_name in DIMENSIONLESS_FIELDS and classify_unit(unit_text) is not None:
            return

        try:
            recovered = float(numbers[0])
        except ValueError:  # pragma: no cover - the regex guarantees a float
            return
        self.value = recovered
        recorded_units = ""
        if not (self.units or "").strip() and remainder:
            #  The unit came from the source's own value_text and was just validated
            #  against this field's dimension. Recording it is what lets the magnitude
            #  normalisation below act: without it a recovered "12 angstrom" keeps
            #  value 12 under a field named _nm, which is the ten-times error this
            #  whole area exists to stop.
            self.units = remainder
            recorded_units = f"; units taken from value_text as {remainder!r}"
        self.notes = (
            f"{self.notes} [value {recovered} recovered from value_text "
            f"{text!r}, which the model left non-numeric{recorded_units}]"
        ).strip()

    def _reject_dimensionally_impossible_units(self) -> None:
        """Refuse a registry-keyed claim whose units cannot belong to that quantity.

        Raises rather than flagging: ``extract.py`` records the rejection as a problem
        on the brief, so the claim is dropped *and* the drop is visible. A mislabelled
        claim is worse than a missing one here, because downstream it is
        indistinguishable from a real one and the interpretation step will reason over
        it — which is exactly what happened, reporting four purge and dose times as a
        literature disagreement about growth per cycle.
        """
        if self.field_name in CATEGORICAL_CONTEXT_FIELDS and self.value is not None:
            raise ResearchContractError(
                f"{self.field_name} names a category, not a quantity, so it cannot have the "
                f"numeric value {self.value}. If the passage states it, it belongs in this "
                f"claim's context rather than as a claim of its own."
            )
        expected = FIELD_DIMENSION.get(self.field_name)
        actual = classify_unit(self.units)
        if expected and actual and actual != expected:
            raise ResearchContractError(
                f"{self.field_name} must be a {expected.replace('_', ' ')}, but its units "
                f"{self.units!r} are a {actual.replace('_', ' ')}. A quantity filed under the "
                f"wrong key cannot be compared with anything and would be read as real."
            )
        if self.field_name in DIMENSIONLESS_FIELDS and actual in {
            "time", "temperature", "pressure", "length", "density", "rate",
            "growth_per_cycle",
        }:
            raise ResearchContractError(
                f"{self.field_name} is dimensionless, but its units {self.units!r} are a "
                f"{actual.replace('_', ' ')}."
            )

    def _derive_kelvin_from_celsius(self) -> None:
        """Fill ``temperature_k`` from ``temperature_c``, in code rather than in a prompt.

        ``REQUIRED_CONTEXT`` demands ``temperature_k`` for every growth claim, while the
        extraction prompt forbids converting anything — correctly, because a model that
        converts units confidently is the worst failure mode available to it. Taken
        together those two rules made a claim from any source stating °C permanently
        incomparable, and essentially every ALD and PLD paper states °C. So the
        conversion happens here: it is exact, it is one line, and it is auditable.

        ``temperature_c`` is kept, not consumed. What the source said is the record; the
        kelvin value is derived from it and says so.
        """
        celsius = self.context.get("temperature_c")
        if celsius is None or "temperature_k" in self.context:
            return
        if isinstance(celsius, bool) or not isinstance(celsius, (int, float)):
            return
        self.context["temperature_k"] = round(float(celsius) + 273.15, 2)
        self.notes = (
            f"{self.notes} [temperature_k {self.context['temperature_k']} derived from "
            f"the stated {celsius} degC; the source did not state kelvin]"
        ).strip()

    def _quarantine_mislabelled_units(self) -> None:
        """Move a non-numeric value out of a unit-bearing context field.

        ``temperature_k="200 to 300 degC"`` is worse than a missing temperature: the
        field name asserts kelvin, so anything that trusts it reads 200-300 K. The
        value is preserved under ``<field>_as_stated`` — the source did say it, and
        discarding it would lose information — and the unit-bearing field is left
        absent, so ``missing_context`` reports it honestly.
        """
        for field_name, unit in UNIT_BEARING_CONTEXT.items():
            if field_name not in self.context:
                continue
            value = self.context[field_name]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                continue
            #  A bare numeric string is fine; anything else is not a value in `unit`.
            try:
                self.context[field_name] = float(str(value).strip())
                continue
            except (TypeError, ValueError):
                pass
            self.context[f"{field_name}_as_stated"] = value
            del self.context[field_name]
            self.notes = (
                f"{self.notes} [{field_name} held {value!r}, which is not a number in "
                f"{unit}; moved to {field_name}_as_stated and the field left absent]"
            ).strip()

    @property
    def missing_context(self) -> list[str]:
        """Context this claim's field needs and does not have."""
        required = REQUIRED_CONTEXT.get(self.field_name, ())
        return [key for key in required if self.context.get(key) in (None, "", [])]

    @property
    def is_context_complete(self) -> bool:
        return not self.missing_context

    @property
    def is_comparable(self) -> bool:
        """Whether this claim could be set beside a stored value at all.

        Needs a number, a unit, complete context, and at least one locatable piece
        of evidence. Anything less is worth reading and not worth comparing.
        """
        return (
            self.value is not None
            and bool(self.units)
            and not self.magnitude_unverified
            and self.is_context_complete
            and any(item.is_locatable for item in self.evidence)
        )

    def as_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "units": self.units,
            "value_text": self.value_text,
            "normalized_value": self.normalized_value,
            "normalized_units": self.normalized_units,
            "normalization_note": self.normalization_note,
            "tier": self.tier.value,
            "status": self.status.value,
            "context": self.context,
            "missing_context": self.missing_context,
            "context_complete": self.is_context_complete,
            "comparable": self.is_comparable,
            "model_confidence": self.model_confidence,
            "extracted_by_model": self.extracted_by_model,
            "extracted_by_provider": self.extracted_by_provider,
            "prompt_version": self.prompt_version,
            "extracted_at": self.extracted_at.isoformat(),
            "notes": self.notes,
            "evidence": [item.as_dict() for item in self.evidence],
            "is_measurement": False,
        }


@dataclass
class Contradiction:
    """Two claims that cannot both hold, kept as two.

    There is deliberately no ``resolved_value``. FOM_PROOF Sec. 2.1 forbids merging
    records without a declared aggregation rule, and two sources disagreeing do not
    have a mean worth reporting — they have a discrepancy someone has to explain.
    """

    field_name: str
    left: ExtractedClaim
    right: ExtractedClaim
    basis: str
    #  What differs in their context, which is usually the explanation.
    differing_context: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.basis or "").strip():
            raise ResearchContractError(
                "A contradiction needs a basis saying what conflicts and how it was judged. "
                "Recording that two sources disagree while discarding what they disagree about "
                "leaves nobody able to settle it."
            )

    @property
    def relative_spread(self) -> float | None:
        left, right = self.left.value, self.right.value
        if left is None or right is None:
            return None
        midpoint = (left + right) / 2.0
        return abs(left - right) / abs(midpoint) if midpoint else None

    def as_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "basis": self.basis,
            "relative_spread": self.relative_spread,
            "differing_context": {k: list(v) for k, v in self.differing_context.items()},
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
        }


@dataclass
class DataGap:
    """Something the corpus was asked for and does not contain.

    A first-class object rather than a sentence in prose, because the gap list is
    the actionable output of a brief: it is the reading list, and it is what
    distinguishes "we do not know" from "we did not look".
    """

    question: str
    what_was_searched: str
    what_would_resolve_it: str
    field_name: str | None = None

    def __post_init__(self) -> None:
        if not (self.what_would_resolve_it or "").strip():
            raise ResearchContractError(
                f"Data gap {self.question!r} does not say what would resolve it. A gap with no "
                "route out of it is a complaint, not a finding."
            )

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "field_name": self.field_name,
            "what_was_searched": self.what_was_searched,
            "what_would_resolve_it": self.what_would_resolve_it,
        }


# ---------------------------------------------------------------------------
# Proposed BO context
# ---------------------------------------------------------------------------


@dataclass
class BoundProposal:
    """A proposed narrowing of one numeric parameter."""

    parameter: str
    lower: float
    upper: float
    rationale: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    card_slug: str | None = None

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ResearchContractError(
                f"{self.parameter}: proposed upper ({self.upper}) must exceed lower ({self.lower})."
            )
        if not (self.rationale or "").strip():
            raise ResearchContractError(f"{self.parameter}: a bound proposal needs a rationale.")

    def as_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "lower": self.lower,
            "upper": self.upper,
            "rationale": self.rationale,
            "card_slug": self.card_slug,
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass
class ProposedBOContext:
    """A proposed change to a campaign's configuration, awaiting review.

    The hard/soft split is the design. ``recommended_bounds`` and
    ``excluded_choices`` can change what the optimizer is allowed to propose, so
    they are reviewed and applied explicitly. ``soft_priors``,
    ``process_window_hints`` and ``uncertainty_notes`` are advisory: they are
    written to the campaign's ``notes`` for a human to read and never enter the
    acquisition function, because a literature prior silently steering a GP is a
    result nobody can attribute afterwards.
    """

    bo_run_id: int
    #  Narrowings only. Checked against the live space in `widening_violations`.
    recommended_bounds: list[BoundProposal] = field(default_factory=list)
    #  {parameter: [choices to remove]} — categorical combinations known infeasible.
    excluded_choices: dict[str, list[Any]] = field(default_factory=dict)
    #  Advisory. Never numerical inputs.
    soft_priors: list[str] = field(default_factory=list)
    process_window_hints: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)
    rationale: str = ""
    #  Card slugs this proposal rests on. The bridge re-checks each one is
    #  reviewed, non-stale, and sourced at apply time, not just at propose time.
    supporting_card_slugs: list[str] = field(default_factory=list)
    status: ContextStatus = ContextStatus.PROPOSED
    proposed_by: str = "assistant"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    @property
    def is_empty(self) -> bool:
        return not (
            self.recommended_bounds
            or self.excluded_choices
            or self.soft_priors
            or self.process_window_hints
            or self.uncertainty_notes
        )

    @property
    def changes_search_behaviour(self) -> bool:
        """Whether applying this would change what the optimizer may propose.

        False means the proposal is entirely advisory, which is a materially
        different review: nobody needs to check a note as carefully as a bound.
        """
        return bool(self.recommended_bounds or self.excluded_choices)

    def widening_violations(self, search_space: dict) -> list[str]:
        """Bounds that would widen the live space, or name a parameter it lacks.

        A paper may persuade a scientist to *narrow* a search — that is what
        evidence is for. Widening one past its instrument envelope is a claim about
        what a tool can physically do, and no paper is a source for that.
        """
        parameters = {
            spec.get("name"): spec for spec in (search_space or {}).get("parameters", [])
        }
        problems: list[str] = []

        for proposal in self.recommended_bounds:
            spec = parameters.get(proposal.parameter)
            if spec is None:
                problems.append(
                    f"{proposal.parameter!r} is not a parameter of this campaign "
                    f"(has: {sorted(n for n in parameters if n)})."
                )
                continue
            if spec.get("kind") == "categorical":
                problems.append(
                    f"{proposal.parameter!r} is categorical; propose excluded_choices for it "
                    "rather than numeric bounds."
                )
                continue
            low, high = spec.get("lower"), spec.get("upper")
            if low is None or high is None:
                problems.append(f"{proposal.parameter!r} has no bounds in the live space.")
                continue
            if proposal.lower < float(low):
                problems.append(
                    f"{proposal.parameter}: proposed lower {proposal.lower} is below the live "
                    f"lower bound {low}. A proposal may narrow a search space, never widen it — "
                    "widening is a claim about what the instrument can reach."
                )
            if proposal.upper > float(high):
                problems.append(
                    f"{proposal.parameter}: proposed upper {proposal.upper} is above the live "
                    f"upper bound {high}. A proposal may narrow a search space, never widen it."
                )

        for parameter, choices in (self.excluded_choices or {}).items():
            spec = parameters.get(parameter)
            if spec is None:
                problems.append(f"{parameter!r} is not a parameter of this campaign.")
                continue
            if spec.get("kind") != "categorical":
                problems.append(f"{parameter!r} is not categorical; excluded_choices does not apply.")
                continue
            available = list(spec.get("choices") or [])
            unknown = [c for c in choices if c not in available]
            if unknown:
                problems.append(f"{parameter}: {unknown} are not choices of this parameter.")
            if available and not [c for c in available if c not in choices]:
                problems.append(
                    f"{parameter}: excluding {choices} would leave no choices at all."
                )
        return problems

    def as_constraint_patch(self) -> dict:
        """The ``ConstraintSet``-shaped dict this proposal would contribute.

        Matches ``bo_engine.constraints.ConstraintSet.from_dict``, so the bridge
        hands the BO engine a shape it already understands rather than a parallel
        one. Advisory content goes into ``notes``, which that class already
        documents as "rules a human must check".
        """
        notes: list[str] = []
        for prior in self.soft_priors:
            notes.append(f"[soft prior, advisory] {prior}")
        for hint in self.process_window_hints:
            notes.append(f"[process window, advisory] {hint}")
        for note in self.uncertainty_notes:
            notes.append(f"[uncertainty, advisory] {note}")
        for proposal in self.recommended_bounds:
            notes.append(
                f"[bound rationale] {proposal.parameter} narrowed to "
                f"[{proposal.lower}, {proposal.upper}]: {proposal.rationale}"
            )
        return {
            "bounds": {p.parameter: [p.lower, p.upper] for p in self.recommended_bounds},
            "allowed_choices": {},  # filled by the bridge, which knows the live choices
            "notes": notes,
        }

    def as_dict(self) -> dict:
        return {
            "bo_run_id": self.bo_run_id,
            "status": self.status.value,
            "changes_search_behaviour": self.changes_search_behaviour,
            "recommended_bounds": [p.as_dict() for p in self.recommended_bounds],
            "excluded_choices": self.excluded_choices,
            "soft_priors": self.soft_priors,
            "process_window_hints": self.process_window_hints,
            "uncertainty_notes": self.uncertainty_notes,
            "rationale": self.rationale,
            "supporting_card_slugs": self.supporting_card_slugs,
            "proposed_by": self.proposed_by,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------


@dataclass
class LabelledStatement:
    """One sentence, with what kind of sentence it is."""

    kind: StatementKind
    text: str
    evidence: list[EvidenceItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind is StatementKind.EVIDENCE and not self.evidence:
            raise ResearchContractError(
                f"A statement labelled EVIDENCE must carry its evidence: {self.text[:80]!r}"
            )

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass
class ResearchBrief:
    """What the assistant found, what it could not find, and what it suggests.

    Read-only with respect to every scientific table. Producing one changes
    nothing: it is a document about the state of the evidence, and the only thing
    it can cause is a person deciding to act.
    """

    research_question: str
    bo_run_id: int | None = None
    experiment_id: int | None = None
    material: str | None = None
    specimen_form: str | None = None
    target_property: str | None = None
    fom_definition: str | None = None

    evidence: list[EvidenceItem] = field(default_factory=list)
    claims: list[ExtractedClaim] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    data_gaps: list[DataGap] = field(default_factory=list)
    statements: list[LabelledStatement] = field(default_factory=list)
    proposed_actions: list[str] = field(default_factory=list)
    proposed_context: ProposedBOContext | None = None
    proposed_card_slugs: list[str] = field(default_factory=list)

    #  Warnings the loop is obliged to surface: an unapproved FOM definition, a
    #  stalled campaign, suggestions on a bound, a clamped fit parameter, a
    #  proposed card used as context.
    warnings: list[str] = field(default_factory=list)

    status: BriefStatus = BriefStatus.PROPOSED
    model: str = ""
    provider: str = ""
    policy_version: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    #  True when the assistant declined for want of evidence. A brief that abstains
    #  is a successful brief; it is the gap list that makes it useful.
    abstained: bool = False
    #  Non-empty when one or more model calls failed outright. A failed grade drops
    #  its passage and a failed extraction yields no claims, so an unreachable model
    #  produces a brief that looks like a thin corpus. Kept separate from
    #  ``abstained`` because "we found no evidence" and "we could not look" are
    #  different findings, and only the first is a finding at all.
    degraded_reason: str = ""

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reason)

    def __post_init__(self) -> None:
        if not (self.research_question or "").strip():
            raise ResearchContractError("A brief needs a research question.")

    @property
    def comparable_claims(self) -> list[ExtractedClaim]:
        return [claim for claim in self.claims if claim.is_comparable]

    @property
    def incomplete_claims(self) -> list[ExtractedClaim]:
        return [claim for claim in self.claims if not claim.is_context_complete]

    @property
    def unsupported_statements(self) -> list[LabelledStatement]:
        """Interpretations and proposals resting on no evidence anywhere in the brief.

        Not an error — an interpretation is allowed to reason over several pieces of
        evidence without citing one per sentence — but the count is the headline
        number when judging whether a brief is grounded.
        """
        if self.evidence or self.claims:
            return []
        return [s for s in self.statements if s.kind is not StatementKind.EVIDENCE]

    def fingerprint(self) -> str:
        """Stable hash of the brief's substantive content.

        Excludes timestamps and the tool trace, so re-running the same question
        against an unchanged corpus produces the same fingerprint — which is how a
        benchmark tells a policy change from noise.
        """
        payload = {
            "question": self.research_question,
            "bo_run_id": self.bo_run_id,
            "claims": sorted(
                f"{c.field_name}={c.value}{c.units or ''}|{sorted(c.context.items(), key=str)}"
                for c in self.claims
            ),
            "gaps": sorted(g.question for g in self.data_gaps),
            "contradictions": sorted(c.field_name for c in self.contradictions),
            "evidence": sorted(e.citation for e in self.evidence),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        return {
            "research_question": self.research_question,
            "bo_run_id": self.bo_run_id,
            "experiment_id": self.experiment_id,
            "material": self.material,
            "specimen_form": self.specimen_form,
            "target_property": self.target_property,
            "fom_definition": self.fom_definition,
            "status": self.status.value,
            "abstained": self.abstained,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "evidence": [item.as_dict() for item in self.evidence],
            "claims": [claim.as_dict() for claim in self.claims],
            "contradictions": [c.as_dict() for c in self.contradictions],
            "data_gaps": [g.as_dict() for g in self.data_gaps],
            "statements": [s.as_dict() for s in self.statements],
            "proposed_actions": self.proposed_actions,
            "proposed_context": self.proposed_context.as_dict() if self.proposed_context else None,
            "proposed_card_slugs": self.proposed_card_slugs,
            "warnings": self.warnings,
            "model": self.model,
            "provider": self.provider,
            "policy_version": self.policy_version,
            "tool_calls": self.tool_calls,
            "created_at": self.created_at.isoformat(),
            "fingerprint": self.fingerprint(),
            "n_comparable_claims": len(self.comparable_claims),
            "n_incomplete_claims": len(self.incomplete_claims),
            "disclaimer": (
                "Every claim here is a literature extraction, not a measurement. Nothing in this "
                "brief has entered property_values, descriptor_values, or any FOM score, and "
                "there is no code path by which it could. A value reaches the analysis tables "
                "through PropertyValue with its DOI, page, and full context, entered or reviewed "
                "by a person."
            ),
        }
