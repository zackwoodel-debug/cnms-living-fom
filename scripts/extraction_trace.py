#!/usr/bin/env python3
"""Where does an expected claim get lost?

``extraction_f1 = 0.397`` is a single number over a pipeline with seven places a
claim can vanish, and "extraction recall is the binding constraint" is a hypothesis
that number cannot settle: a unit- or key-normalisation mismatch in the *grader*
produces exactly the same score as a genuine extraction miss. This walks one
expected claim through every stage and names the first one that loses it.

Stages, in the order a claim passes them:

    S0  the value is in the stored chunk text at all      (PDF text layer)
    S1  the value, its unit and its subject are together  (chunking)
    S2  that chunk is retrieved and survives grading      (retrieval)
    S3  the extractor emitted something for it            (model)
    S4  the emitted claim survived the guards             (validation)
    S5  the grader matched it against the gold claim      (evaluation)
    S6  it reached the reader                             (brief assembly)

Diagnostics only. This module imports production code and must never modify it.

Usage::

    DATABASE_URL=postgresql+psycopg2://... PYTHONPATH=backend \\
        python scripts/extraction_trace.py                 # all five claims
    ... python scripts/extraction_trace.py --only gpc_098  # one of them
    ... python scripts/extraction_trace.py --markdown      # the audit table
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from cnms_fom.research.contracts import EvidenceItem  # noqa: E402

CONTEXT_WINDOW = 200

#  Glyph variants of Angstrom, which is the unit most likely to survive a PDF text
#  layer mangled. Checked explicitly because "the number is present but its unit is
#  mojibake" is a PDF-layer bug that looks exactly like an extraction miss.
ANGSTROM_VARIANTS: dict[str, str] = {
    "Å": "U+00C5 LATIN CAPITAL LETTER A WITH RING ABOVE",
    "Å": "U+212B ANGSTROM SIGN",
    "Å": "A + U+030A COMBINING RING ABOVE",
    "A˚": "A + U+02DA RING ABOVE (spacing)",
    "˚A": "U+02DA RING ABOVE + A (transposed)",
    "°A": "U+00B0 DEGREE SIGN + A (mis-mapped ring)",
    "A°": "A + U+00B0 DEGREE SIGN (mis-mapped ring)",
}


@dataclass
class ClaimSpec:
    """An expected claim, and how to look for it without cheating.

    ``number_pattern`` matches the NUMBER ONLY. Searching for "0.98 A/cycle" would
    make a mangled unit glyph indistinguishable from a missing value, which is the
    exact confusion this script exists to resolve.
    """

    name: str
    field_name: str
    value: float
    units: str
    number_pattern: str
    #  Words a human would use to identify what the number is *of*. Used to decide
    #  whether chunking separated the value from its subject.
    subject_terms: tuple[str, ...]
    unit_terms: tuple[str, ...]
    question: str
    corpus: str  # "real" or "fixture"
    note: str = ""


@dataclass
class StageResult:
    ok: bool | None = None  # None = not reached
    reason: str = ""
    detail: list[str] = field(default_factory=list)

    def line(self) -> str:
        mark = "n/a" if self.ok is None else ("PASS" if self.ok else "FAIL")
        return f"{mark:<5} {self.reason}"


@dataclass
class Trace:
    spec: ClaimSpec
    stages: dict[str, StageResult] = field(default_factory=dict)

    def first_failure(self) -> str | None:
        for key in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
            result = self.stages.get(key)
            if result is not None and result.ok is False:
                return key
        return None


CLAIMS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        name="gpc_098",
        field_name="growth_per_cycle_ang",
        value=0.98,
        units="A/cycle",
        number_pattern=r"0\.98",
        subject_terms=("growth per cycle", "growth-per-cycle", "gpc"),
        unit_terms=("angstrom", "Å", "a/cycle", "per cycle"),
        question=(
            "What growth per cycle and film density are reported for HfO2 ALD, "
            "and do the sources agree?"
        ),
        corpus="real",
        note="the missing one",
    ),
    ClaimSpec(
        name="gpc_142",
        field_name="growth_per_cycle_ang",
        value=1.42,
        units="A/cycle",
        number_pattern=r"1\.42",
        subject_terms=("growth per cycle", "growth-per-cycle", "gpc", "grew at"),
        unit_terms=("angstrom", "Å", "a/cycle", "per cycle"),
        question=(
            "What growth per cycle and film density are reported for HfO2 ALD, "
            "and do the sources agree?"
        ),
        corpus="real",
        note="CONTROL - this one works",
    ),
    ClaimSpec(
        name="decomposition_onset",
        field_name="decomposition_onset_c",
        value=320.0,
        units="degC",
        number_pattern=r"\b320\b",
        subject_terms=("decomposition", "decompos", "onset"),
        unit_terms=("degc", "°c", "celsius"),
        question="At what temperature does TDMAH begin to decompose?",
        corpus="fixture",
    ),
    ClaimSpec(
        name="substrate_temperature",
        field_name="substrate_temperature",
        value=700.0,
        units="degC",
        number_pattern=r"\b700\b",
        subject_terms=("substrate_temperature", "substrate temperature"),
        unit_terms=("degc", "°c", "celsius"),
        question="What substrate temperature and oxygen pressure were used for the LSMO growth?",
        corpus="fixture",
    ),
    ClaimSpec(
        name="oxygen_pressure",
        field_name="oxygen_pressure",
        value=100.0,
        units="mTorr",
        number_pattern=r"\b100\b",
        subject_terms=("oxygen_pressure", "oxygen pressure", "oxygen partial"),
        unit_terms=("mtorr", "torr"),
        question="What substrate temperature and oxygen pressure were used for the LSMO growth?",
        corpus="fixture",
    ),
)


# --- glyph forensics -------------------------------------------------------


def describe_glyphs(text: str) -> list[str]:
    """Angstrom variants and non-ASCII oddities present in a span."""
    found = []
    for variant, description in ANGSTROM_VARIANTS.items():
        if variant in text:
            found.append(f"contains {description}")
    exotic = {
        ch for ch in text
        if ord(ch) > 127 and not unicodedata.category(ch).startswith("L")
    }
    if exotic:
        rendered = ", ".join(
            f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}" for ch in sorted(exotic)
        )
        found.append(f"non-letter non-ASCII: {rendered}")
    #  A number and its unit split by a newline is a classic PDF two-column artefact.
    if re.search(r"\d\s*\n\s*(?:Å|Å|A|nm|s)\b", text):
        found.append("NUMBER AND UNIT SEPARATED BY A LINE BREAK")
    return found


def looks_tabular(text: str) -> bool:
    """Whether a span reads like a table row rather than prose."""
    return bool(
        re.search(r"\|", text)
        or re.search(r"\S {3,}\S", text)
        or re.search(r"(?:^|\n)\s*Table\s+\d", text, re.IGNORECASE)
    )


# --- the stages ------------------------------------------------------------


def stage_s0(db, spec: ClaimSpec) -> tuple[StageResult, list]:
    """Is the literal number in the stored chunk text at all?"""
    from cnms_fom.db.models import Document, DocumentChunk

    pattern = re.compile(spec.number_pattern)
    rows = (
        db.query(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        .all()
    )

    hits = []
    for chunk, document in rows:
        for match in pattern.finditer(chunk.text):
            hits.append((chunk, document, match.start()))

    result = StageResult()
    if not hits:
        result.ok = False
        result.reason = (
            f"the number {spec.value} does not appear in any stored chunk "
            f"(searched {len(rows)} chunks, number only, no unit)"
        )
        return result, []

    result.ok = True
    result.reason = f"found {len(hits)} occurrence(s) of the bare number"
    for chunk, document, offset in hits:
        lo = max(0, offset - CONTEXT_WINDOW)
        hi = min(len(chunk.text), offset + CONTEXT_WINDOW)
        span = chunk.text[lo:hi]
        result.detail.append(
            f"chunk {chunk.id} (doc {document.id} {document.title!r} p{chunk.page}) "
            f"offset {offset}"
        )
        result.detail.append(f"  span repr: {span!r}")
        glyphs = describe_glyphs(span)
        result.detail.append(
            "  glyphs: " + ("; ".join(glyphs) if glyphs else "clean ASCII, no variants")
        )
        result.detail.append(f"  tabular region: {looks_tabular(span)}")
    return result, hits


def stage_s1(spec: ClaimSpec, hits: list) -> StageResult:
    """Are the number, its unit and its subject in the same chunk?"""
    result = StageResult()
    if not hits:
        return result

    verdicts = []
    for chunk, _document, offset in hits:
        lowered = chunk.text.lower()
        unit_at = next(
            (lowered.find(term) for term in spec.unit_terms if term in lowered), -1
        )
        subject_at = next(
            (lowered.find(term) for term in spec.subject_terms if term in lowered), -1
        )
        verdicts.append((chunk, offset, unit_at, subject_at))
        result.detail.append(
            f"chunk {chunk.id}: len={len(chunk.text)} number@{offset} "
            f"unit@{unit_at if unit_at >= 0 else 'ABSENT'} "
            f"subject@{subject_at if subject_at >= 0 else 'ABSENT'}"
        )

    #  The claim is chunk-intact if any single chunk holds all three.
    intact = [v for v in verdicts if v[2] >= 0 and v[3] >= 0]
    if intact:
        result.ok = True
        result.reason = (
            f"number, unit and subject co-located in chunk {intact[0][0].id}"
        )
    else:
        has_unit = any(v[2] >= 0 for v in verdicts)
        has_subject = any(v[3] >= 0 for v in verdicts)
        missing = []
        if not has_unit:
            missing.append("unit token")
        if not has_subject:
            missing.append("subject noun")
        result.ok = False
        result.reason = (
            f"no single chunk holds number + unit + subject; missing in-chunk: "
            f"{', '.join(missing) or 'co-location only'}"
        )
    return result


def stage_s2(db, spec: ClaimSpec, hits: list, provider) -> tuple[StageResult, list]:
    """Is the chunk retrieved, and does it survive grading?"""
    from cnms_fom.rag_backend.grading import retrieve_with_correction
    from cnms_fom.rag_backend.hybrid import hybrid_search

    result = StageResult()
    if not hits:
        return result, []

    wanted = {chunk.id for chunk, _d, _o in hits}
    dialect = db.get_bind().dialect.name
    result.detail.append(f"backend: {dialect}")

    for label, kwargs in (
        ("lexical", {"use_vector": False}),
        ("dense", {"use_lexical": False}),
        ("fusion", {}),
    ):
        try:
            found = hybrid_search(db, spec.question, k=12, depth=12, **kwargs)
            ranks = [
                i + 1 for i, h in enumerate(found) if h.hit.chunk_id in wanted
            ]
            result.detail.append(
                f"  {label:<8} {len(found)} hits; target at rank(s) "
                f"{ranks or 'NOT RETRIEVED'}"
            )
        except Exception as exc:  # noqa: BLE001 - a diagnostic reports, never raises
            result.detail.append(f"  {label:<8} FAILED {type(exc).__name__}: {exc}")

    survivors = []
    try:
        outcome = retrieve_with_correction(
            db, spec.question, k=6, depth=12, provider=provider
        )
        for graded in outcome.hits:
            if graded.fused.hit.chunk_id in wanted:
                result.detail.append(
                    f"  graded: grade={graded.grade} useful={graded.useful} "
                    f"reason={graded.reason[:90]!r}"
                )
        survivors = [
            g for g in outcome.useful if g.fused.hit.chunk_id in wanted
        ]
        if survivors:
            result.ok = True
            result.reason = f"retrieved and survived grading ({len(survivors)} passage(s))"
        else:
            retrieved_at_all = any(
                g.fused.hit.chunk_id in wanted for g in outcome.hits
            )
            result.ok = False
            result.reason = (
                "retrieved but graded out below the useful threshold"
                if retrieved_at_all
                else "not retrieved into the graded window at all"
            )
    except Exception as exc:  # noqa: BLE001
        result.ok = False
        result.reason = f"retrieval raised {type(exc).__name__}: {exc}"
    return result, survivors


def stage_s3(spec: ClaimSpec, survivors: list, provider, policy) -> tuple[StageResult, list]:
    """What did the extractor actually emit for the passage holding the value?"""
    from cnms_fom.research.extract import _call_extractor, extraction_provider

    result = StageResult()
    if not survivors:
        return result, []

    extractor = extraction_provider(provider, policy)
    result.detail.append(
        f"prompt_version={policy.extraction_prompt_version} "
        f"model={getattr(extractor, 'model', '?')} "
        f"provider={getattr(extractor, 'name', '?')}"
    )

    emitted: list[dict] = []
    for graded in survivors:
        hit = graded.fused.hit
        item = EvidenceItem(
            document_id=hit.document_id,
            document_title=hit.document_title,
            page=hit.page,
            quote=hit.text,
            chunk_id=hit.chunk_id,
        )
        payload = _call_extractor(extractor, item)
        if payload.get("refused"):
            result.detail.append(f"chunk {hit.chunk_id}: MODEL REFUSED")
            continue
        if payload.get("unparseable"):
            result.detail.append(f"chunk {hit.chunk_id}: MALFORMED STRUCTURED OUTPUT")
            continue
        claims = payload.get("claims") or []
        result.detail.append(f"chunk {hit.chunk_id}: {len(claims)} raw claim(s) emitted")
        for raw in claims:
            emitted.append({"raw": raw, "item": item, "payload": payload})
            result.detail.append(f"  {json.dumps(raw, ensure_ascii=False)[:300]}")

    #  Classify, on the number rather than the key: the whole point is to separate
    #  "never saw it" from "saw it and called it something else".
    target = f"{spec.value:g}"
    matching = [
        e for e in emitted
        if target in str(e["raw"].get("value"))
        or target in str(e["raw"].get("value_text") or "")
    ]
    right_key = [e for e in matching if str(e["raw"].get("field", "")).strip() == spec.field_name]

    if not emitted:
        result.ok = False
        result.reason = "EMITTED NOTHING - no claims at all from the passage"
    elif not matching:
        result.ok = False
        result.reason = (
            f"EMITTED NOTHING FOR THIS VALUE - {len(emitted)} other claim(s) emitted, "
            f"none carrying {spec.value}"
        )
    elif right_key:
        result.ok = True
        result.reason = f"EMITTED CORRECTLY under {spec.field_name!r}"
    else:
        keys = sorted({str(e["raw"].get("field")) for e in matching})
        result.ok = False
        result.reason = f"WRONG REGISTRY KEY - value emitted under {keys}"
    return result, emitted


def stage_s4(spec: ClaimSpec, emitted: list, policy) -> tuple[StageResult, list]:
    """Which guard killed each emitted claim?"""
    from cnms_fom.research.extract import _claim_from_extraction

    result = StageResult()
    if not emitted:
        return result, []

    survived = []
    target = f"{spec.value:g}"
    for entry in emitted:
        raw, item, payload = entry["raw"], entry["item"], entry["payload"]
        problems: list[str] = []
        claim = _claim_from_extraction(
            raw, item, payload.get("model", ""), payload.get("provider", ""),
            policy, problems,
        )
        label = f"{raw.get('field')}={raw.get('value')} {raw.get('units')!r}"
        if claim is None:
            #  Every drop must name a guard. An unexplained disappearance is a bug of
            #  the same class as bug 14 and is reported as such.
            reason = problems[-1] if problems else "*** DROPPED WITH NO NAMED REASON ***"
            result.detail.append(f"KILLED  {label}: {reason[:200]}")
        else:
            survived.append(claim)
            flags = []
            if claim.missing_context:
                flags.append(f"missing_context={claim.missing_context}")
            if not claim.is_comparable:
                flags.append("not comparable")
            result.detail.append(
                f"SURVIVED {label} -> {claim.field_name}={claim.value} "
                f"{claim.units!r} {' '.join(flags)}"
            )

    #  The None check must precede the format: in a comprehension the `if` clauses
    #  evaluate left to right, and a value-less claim is legitimate (value_text only).
    ours = [c for c in survived if c.value is not None and target in f"{c.value:g}"]
    if ours:
        result.ok = True
        result.reason = f"the target claim survived validation ({len(ours)})"
    elif any(target in str(e["raw"].get("value")) for e in emitted):
        result.ok = False
        result.reason = "the target was emitted and a guard rejected it"
    else:
        result.reason = "target never emitted; nothing for the guards to reject"
    return result, survived


def stage_s5(spec: ClaimSpec, survived: list) -> StageResult:
    """Would the benchmark's matcher pair the surviving claim with the gold claim?"""
    from cnms_fom.research.benchmark.evaluate import normalise_units

    result = StageResult()
    if not survived:
        return result

    gold_units = normalise_units(spec.units)
    for claim in survived:
        if claim.value is None:
            continue
        key_ok = claim.field_name == spec.field_name
        value_ok = abs(claim.value - spec.value) <= max(abs(spec.value) * 0.05, 1e-12)
        unit_norm = normalise_units(claim.units)
        unit_ok = unit_norm == gold_units
        if value_ok:
            result.detail.append(
                f"candidate {claim.field_name}={claim.value} {claim.units!r}: "
                f"key_match={key_ok} value_match={value_ok} "
                f"units {unit_norm!r} vs gold {gold_units!r} match={unit_ok} "
                f"page={claim.evidence[0].page if claim.evidence else None}"
            )
            if key_ok and unit_ok:
                result.ok = True
                result.reason = "matcher pairs it with the gold claim"
                return result
            result.ok = False
            result.reason = (
                f"NORMALISATION MISMATCH, not a recall failure: "
                f"key_match={key_ok} unit_match={unit_ok}"
            )
            return result
    result.reason = "no surviving claim carries the gold value"
    return result


def stage_s6(db, spec: ClaimSpec, provider, policy) -> StageResult:
    """Does the claim reach the reader, or die in brief assembly?"""
    from cnms_fom.research.brief import generate_brief

    result = StageResult()
    try:
        brief = generate_brief(
            db, spec.question, provider=provider, policy=policy,
            interpret=False, include_cards=False,
        )
    except Exception as exc:  # noqa: BLE001
        result.ok = False
        result.reason = f"brief generation raised {type(exc).__name__}: {exc}"
        return result

    target = f"{spec.value:g}"
    present = [
        c for c in brief.claims
        if c.value is not None and target in f"{c.value:g}"
    ]
    result.detail.append(
        f"brief has {len(brief.claims)} claim(s), {len(brief.contradictions)} "
        f"contradiction(s), abstained={brief.abstained}, degraded={brief.degraded}"
    )
    for claim in brief.claims:
        result.detail.append(
            f"  {claim.field_name}={claim.value} {claim.units!r} "
            f"p{claim.evidence[0].page if claim.evidence else '?'}"
        )
    if present:
        result.ok = True
        result.reason = "the claim reaches the brief"
    else:
        result.ok = False
        result.reason = "absent from the brief"
    return result


# --- substring hazard check (§10d) ----------------------------------------


def check_substring_hazard() -> list[str]:
    """Does the dimensional guard use boundary-aware matching, or substrings?

    §10d records that `"s" in "angstrom"` classified every length as a time. This
    asserts the fix is still in place rather than trusting the note.
    """
    from cnms_fom.research.contracts import classify_unit

    cases = {
        "angstrom": "length", "angstroms": "length", "A": "length", "nm": "length",
        "s": "time", "0.2 s": "time", "6 s": "time", "ms": "time",
        "A/cycle": "growth_per_cycle", "angstrom per cycle": "growth_per_cycle",
        "Å/cy": "growth_per_cycle", "widgets": None, "1": None,
    }
    lines = []
    for units, expected in cases.items():
        got = classify_unit(units)
        mark = "ok  " if got == expected else "FAIL"
        lines.append(f"  {mark} classify_unit({units!r}) = {got} (expected {expected})")
    return lines


# --- driver ---------------------------------------------------------------


def open_corpus(kind: str):
    """A session over the named corpus, on Postgres."""
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from cnms_fom.db import models  # noqa: F401 - registers mappers
    from cnms_fom.db.base import Base

    url = os.environ["DATABASE_URL"]
    if kind == "fixture":
        #  The fixture corpus, seeded into its own Postgres database so the
        #  benchmark claims are traced on the real backend too. Bug 14 existed
        #  because a leg had only ever run on SQLite.
        url = url.replace("/cnms_fom?", "/cnms_fom_fixture?")
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, future=True)()
    if kind == "fixture":
        from cnms_fom.db.models import Document
        from cnms_fom.research.benchmark.cases import seed_corpus

        if not session.query(Document).count():
            seed_corpus(session, embed=True)
            session.commit()
    return session


def run(spec: ClaimSpec, provider, policy) -> Trace:
    trace = Trace(spec=spec)
    db = open_corpus(spec.corpus)
    try:
        s0, hits = stage_s0(db, spec)
        trace.stages["S0"] = s0
        trace.stages["S1"] = stage_s1(spec, hits)
        s2, survivors = stage_s2(db, spec, hits, provider)
        trace.stages["S2"] = s2
        s3, emitted = stage_s3(spec, survivors, provider, policy)
        trace.stages["S3"] = s3
        s4, survived = stage_s4(spec, emitted, policy)
        trace.stages["S4"] = s4
        trace.stages["S5"] = stage_s5(spec, survived)
        trace.stages["S6"] = stage_s6(db, spec, provider, policy)
    finally:
        db.close()
    return trace


FIX_CATEGORY = {
    "S0": "A - PDF text layer / unicode normalisation",
    "S1": "B - chunk boundaries",
    "S2": "retrieval or grading, not extraction",
    "S3": "F - model capability (recovery pass / candidate-first / multi-pass)",
    "S4": "C - guard over-rejection",
    "S5": "D - grader normalisation; f1 was understated",
    "S6": "E - brief assembly",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="trace one claim by name")
    parser.add_argument("--markdown", action="store_true", help="emit the audit table")
    args = parser.parse_args()

    from cnms_fom.rag_backend.providers import get_provider
    from cnms_fom.research.policy import BASELINE

    provider = get_provider("ollama", None)
    specs = [c for c in CLAIMS if not args.only or c.name == args.only]

    print("=" * 78)
    print("SUBSTRING HAZARD CHECK (§10d: 's' inside 'angstrom')")
    print("=" * 78)
    for line in check_substring_hazard():
        print(line)

    traces = []
    for spec in specs:
        print()
        print("=" * 78)
        print(f"{spec.name}  {spec.field_name} = {spec.value} {spec.units}"
              + (f"   [{spec.note}]" if spec.note else ""))
        print(f"corpus={spec.corpus}  question={spec.question!r}")
        print("=" * 78)
        trace = run(spec, provider, BASELINE)
        traces.append(trace)
        for key in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
            result = trace.stages[key]
            print(f"\n{key}  {result.line()}")
            for line in result.detail:
                print(f"     {line}")
        first = trace.first_failure()
        print(f"\n>>> FIRST FAILING STAGE: {first or 'none - claim survives end to end'}")
        if first:
            print(f">>> FIX CATEGORY: {FIX_CATEGORY[first]}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = f"{'claim':<24}" + "".join(f"{k:>5}" for k in
                                        ("S0", "S1", "S2", "S3", "S4", "S5", "S6"))
    print(header + "   first failure")
    for trace in traces:
        cells = ""
        for key in ("S0", "S1", "S2", "S3", "S4", "S5", "S6"):
            ok = trace.stages[key].ok
            cells += f"{'-' if ok is None else ('ok' if ok else 'XX'):>5}"
        print(f"{trace.spec.name:<24}{cells}   {trace.first_failure() or '-'}")

    if args.markdown:
        print()
        print("| claim | S0 | S1 | S2 | S3 | S4 | S5 | S6 | first failure | reason | fix |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for trace in traces:
            cells = " | ".join(
                "-" if trace.stages[k].ok is None else ("ok" if trace.stages[k].ok else "**XX**")
                for k in ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
            )
            first = trace.first_failure()
            reason = trace.stages[first].reason if first else "survives end to end"
            print(
                f"| `{trace.spec.name}` | {cells} | {first or '-'} | {reason} | "
                f"{FIX_CATEGORY.get(first, '-')} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
