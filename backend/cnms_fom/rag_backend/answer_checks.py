"""What an answer says, checked against what the tools returned.

Two checks, both deterministic and both run on the model's own words:

* ``unsupported_numbers`` — a number no tool returned, nor rounds from one.
* ``unsupported_claims`` — a material id, an evidence grade, an agreement class or a
  crystal phase / optical axis the answer asserts and no tool returned.

They exist because a local model restates tool output unreliably. Measured on the
atlas evals with llama3.1:8b: it gave values to the right digits yet dropped ids,
reported "anatase data" for a material whose datasets are all rutile (the phase is in
its *name*), and called a consensus the release classes suspicious "agreeing". Each of
those is caught here. The checks cannot read meaning: a phase mentioned with a negation
near it ("there is no anatase dataset") or as part of a name passes, and a claim no
check covers passes too. They narrow what can go wrong; they do not certify an answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence

from cnms_fom.atlas.vocabulary import AXIS_WORDS, PHASE_PHRASES, PHASE_WORDS

# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

#  A number as written in an answer: dotted versions ("0.22.0") stay one token.
_ANSWER_NUMBER = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)*(?:[eE][-+]?\d+)?(?![\w])")
#  Every number inside the evidence, wherever it sits ("n_633nm", "JOSA.55.001205").
_EVIDENCE_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _evidence_text(steps: Sequence) -> str:
    return " ".join(json.dumps(step.result, default=str, ensure_ascii=False) for step in steps)


def unsupported_numbers(answer: str, steps: Sequence, question: str = "") -> list[str]:
    """Numbers in the answer that no tool result (and not the question) contains.

    Supported: a version or an integer that appears anywhere in the evidence (a page number
    inside a DOI, the 633 in n_633nm), and a decimal equal to a returned value or a rounding
    of one at the precision written ("1.33" for 1.331790355). Small integers (counts, "2
    sources") are left alone. What remains is what a model invented, the failure FOM_PROOF
    Sec. 15.2 is written against.
    """
    evidence = _evidence_text(steps) + " " + question
    known: list[float] = []
    for token in _EVIDENCE_NUMBER.findall(evidence):
        try:
            known.append(float(token))
        except ValueError:
            continue
    out: list[str] = []
    for match in _ANSWER_NUMBER.finditer(answer):
        token = match.group(0).replace("−", "-")
        if token in out:
            continue
        bare = token.lstrip("-")
        if bare.count(".") > 1:
            if bare not in evidence:
                out.append(token)
            continue
        if "." not in bare and "e" not in bare.lower():
            if int(bare) < 100 or bare in evidence:
                continue
        elif bare in evidence:
            continue
        value = float(token)
        decimals = len(bare.split(".")[1].split("e")[0].split("E")[0]) if "." in bare else 0
        half = 0.5 * 10 ** (-decimals)
        if any(abs(candidate - abs(value)) <= half * (1 + 1e-9) for candidate in known):
            continue
        out.append(token)
    return out


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

#  Keys of an atlas result that describe *data* (labels, scopes, keys), as opposed to a
#  material's display name. A phase in a name is not a dataset.
DATA_KEYS = frozenset({"label", "scope", "dataset_label", "key", "property_name", "phase", "axis", "status"})
NEGATIONS = frozenset(
    {
        "no", "not", "none", "without", "lacks", "lack", "lacking", "absent", "neither", "nor", "never",
        "only", "isn't", "aren't", "doesn't", "don't", "didn't", "cannot", "can't", "zero",
        #  A mention of the material's name is not a claim about its data.
        "name", "named", "names", "called",
    }
)


def _leaves(value, key: str | None = None) -> Iterable[tuple[str | None, object]]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(v, k)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item, key)
    else:
        yield key, value


#  Returned strings an answer may repeat word for word without making a claim about data: a
#  material's name ("Titanium dioxide (rutile / anatase)"), a source's citation or title
#  ("Refractive indices of rutile and sphalerite"), a verbatim reference line.
QUOTABLE_KEYS = frozenset({"material", "title", "citation", "summary", "name"})


class _Evidence:
    """What the tools returned, sorted into the fields the claim checks compare against."""

    def __init__(self, steps: Sequence) -> None:
        self.ids: set[int] = set()
        self.grades: set[str] = set()
        self.classes: set[str] = set()
        self.quotable: set[str] = set()
        data: list[str] = []
        for step in steps:
            result = step.result if isinstance(step.result, dict) else {}
            atlas = str(getattr(step, "tool", "")).startswith("atlas_")
            if not atlas:
                #  Other tools' text has no name/data split; all of it counts as data.
                data.append(json.dumps(result, default=str, ensure_ascii=False))
            for key, value in _leaves(result):
                if key == "material_id" and isinstance(value, int):
                    self.ids.add(value)
                elif key == "grade" and isinstance(value, str):
                    self.grades.add(value)
                elif key == "grades" and isinstance(value, str):
                    self.grades.add(value)
                elif key == "classification" and isinstance(value, str):
                    self.classes.add(value)
                elif atlas and key in DATA_KEYS and isinstance(value, str):
                    data.append(value)
                if key in QUOTABLE_KEYS and isinstance(value, str) and len(value) >= 8:
                    self.quotable.add(value)
        self.data_text = " ".join(data).casefold()

    def mask_quotes(self, text: str) -> str:
        """Blank out returned names, titles and lines the answer repeats, keeping offsets."""
        masked = text
        for quote in sorted(self.quotable, key=len, reverse=True):
            for match in re.finditer(re.escape(quote), masked, re.IGNORECASE):
                masked = masked[: match.start()] + " " * len(match.group(0)) + masked[match.end() :]
        return masked


def _words(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).casefold(), m.start(), m.end()) for m in re.finditer(r"[\w'’-]+", text)]


def _negated(text: str, start: int, end: int, window: int = 8) -> bool:
    """A negation or a name-mention within ``window`` words either side, inside the same sentence."""
    sentence_start = max((m.end() for m in re.finditer(r"[.!?;:]\s", text[:start])), default=0)
    following = re.search(r"[.!?;:](\s|$)", text[end:])
    sentence_end = end + following.start() if following else len(text)
    sentence = text[sentence_start:sentence_end]
    words = _words(sentence)
    offset = start - sentence_start
    index = next((i for i, (_, s, e) in enumerate(words) if s <= offset < e), None)
    if index is None:
        return False
    nearby = words[max(0, index - window) : index] + words[index + 1 : index + 1 + window]
    return any(w.replace("’", "'") in NEGATIONS or w.endswith("n't") for w, _, _ in nearby)


def _affirmed(text: str, pattern: str) -> bool:
    """Whether ``pattern`` occurs at least once without a negation in the three words before it."""
    for match in re.finditer(pattern, text, re.IGNORECASE):
        before = _words(text[: match.start()])[-3:]
        if not any(w in NEGATIONS or w.endswith("n't") for w, _, _ in before):
            return True
    return False


def unsupported_claims(answer: str, steps: Sequence, question: str = "") -> list[str]:
    """Ids, grades, agreement classes and phases or axes the answer asserts and no tool returned."""
    evidence = _Evidence(steps)
    problems: list[str] = []

    def add(message: str) -> None:
        if message not in problems:
            problems.append(message)

    for match in re.finditer(r"\b(?:material[_ ])?id\s*[:=]?\s*(\d+)\b", answer, re.IGNORECASE):
        number = int(match.group(1))
        if evidence.ids and number not in evidence.ids and match.group(1) not in question:
            add(f"material id {number}: no tool returned it")

    for match in re.finditer(r"\bgrade(?:d)?\s*(?:of|is|:|=)?\s*\**\s*([ABCDST])\b", answer):
        if match.group(1) not in evidence.grades:
            add(f"grade {match.group(1)}: no returned value has that grade")

    for word in ("excellent", "suspicious"):
        if re.search(rf"\b{word}\b", answer, re.IGNORECASE) and evidence.classes and word not in evidence.classes:
            add(f"'{word}': no returned consensus is classed {word}")

    graded = evidence.classes - {"single_source"}
    if graded == {"suspicious"} and _affirmed(answer, r"\bagree(?:s|d|ment)?\b"):
        add("says the sources agree; the release classes them suspicious, i.e. they disagree")

    #  A phase inside a repeated name, title or reference line is a quotation, not a claim.
    scanned = evidence.mask_quotes(answer)
    lowered = scanned.casefold()
    for word in sorted(PHASE_WORDS | PHASE_PHRASES | AXIS_WORDS):
        if word in evidence.data_text:
            continue
        for match in re.finditer(rf"(?<![\w-]){re.escape(word)}(?![\w-])", lowered):
            if not _negated(scanned, match.start(), match.end()):
                add(f"'{word}': no returned dataset or value is labelled {word}")
                break
    return problems
