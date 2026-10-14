"""Answers that rest on atlas values: the values are written by code, the model frames them.

A local 8B model restates structured tool output unreliably — it drops ids, misreads a
grade, invents a phase from a material's name. So when an answer rests on atlas lookups,
the part that carries the facts is not left to it: every value is quoted verbatim from
the lookup's ``summary`` lines (name and id, scope, value, grade or kind with its
meaning, source), under a heading that says what they are. The model writes only a short
framing from those lines, and that framing is checked (``answer_checks``); if it asserts
a number or a claim no tool returned, it is withheld and a template framing stands in.

When the evidence is mixed — atlas values *and* corpus passages or platform records —
the model's answer is needed to connect them, so it stays, checked, with the verbatim
atlas block appended.
"""

from __future__ import annotations

from collections.abc import Sequence

ATLAS_VALUE_TOOLS = frozenset({"atlas_lookup", "atlas_compare", "atlas_coverage"})

FRAMING_SYSTEM = """\
You write the opening two or three sentences of an answer for a materials scientist. \
The reference values are shown to the reader verbatim right after your sentences, so \
do not repeat any number. Use only the lines you are given: do not name a material, \
phase, axis, dataset, source or grade that is not in them, and never say data exists \
that the lines do not show; a phase in a material's name is not data. If several \
materials match, say so and do not pick one. Say what kind of values they are \
(measured, model fit, calculated) and what their grades mean for how far to trust \
them. If a line says a value is absent, say it is absent."""


def _lookups(steps: Sequence) -> list[dict]:
    """atlas_lookup results, including each lookup inside an atlas_compare."""
    out: list[dict] = []
    for step in steps:
        result = step.result if isinstance(step.result, dict) else {}
        if step.tool not in ATLAS_VALUE_TOOLS or result.get("error"):
            continue
        if step.tool == "atlas_compare":
            out.extend(r for r in result.get("materials", []) or [] if isinstance(r, dict))
        else:
            out.append(result)
    return out


def atlas_reference(steps: Sequence) -> tuple[list[str], list[str], str | None]:
    """(verbatim value lines, the atlas's own notes, the origin) from this turn's lookups."""
    lines: list[str] = []
    notes: list[str] = []
    origin = None
    for result in _lookups(steps):
        origin = origin or result.get("origin")
        for line in result.get("summary", []) or []:
            if line not in lines:
                lines.append(line)
        note = result.get("note")
        if note and result.get("matches") and note not in notes:
            notes.append(note)
    return lines, notes, origin


def rests_only_on_atlas(steps: Sequence, has_content) -> bool:
    """Every tool call that returned evidence was an atlas tool."""
    contributing = [step for step in steps if has_content(step.result)]
    return bool(contributing) and all(str(step.tool).startswith("atlas_") for step in contributing)


def framing_messages(question: str, lines: list[str], notes: list[str]) -> list[dict]:
    body = [f"Question: {question}", "", "Reference values (shown verbatim after your sentences):"]
    body += [f"- {line}" for line in lines]
    if notes:
        body += ["", "Notes from the atlas:"] + [f"- {note}" for note in notes]
    return [{"role": "user", "content": "\n".join(body)}]


def template_framing(steps: Sequence) -> str:
    """A framing written from the lookups alone, for when the model's cannot be used."""
    sentences: list[str] = []
    for result in _lookups(steps):
        matches = result.get("matches") or []
        if not matches:
            continue
        names = ", ".join(f"{m.get('material')} (id {m.get('material_id')})" for m in matches)
        query = result.get("material")
        if len(matches) > 1:
            sentences.append(
                f"{len(matches)} materials in the atlas match {query!r}: {names}. A formula is not a "
                "material identity, so each is listed separately."
            )
        else:
            sentences.append(f"The atlas record for {query!r} is {names}.")
    sentences.append(
        "Its stored values are listed below exactly as the release holds them, each with its grade "
        "or kind and its source."
    )
    return " ".join(sentences)


def reference_block(lines: list[str], origin: str | None) -> str:
    source = origin or "materials-db (reference)"
    heading = f"**Reference values** — quoted verbatim from the atlas, {source}; not CNMS measurements:"
    return "\n".join([heading, ""] + [f"- {line}" for line in lines])
