"""Who wrote a source, and how many independent author groups a set of sources spans.

Rubric 6.4 of materials-atlas/CLAUDE.md, made conservative:

* ``authors`` is parsed into (surname, first initial) pairs. Both formats in the
  release are handled — "J. Walling, O. Peterson" (initials first, commas and
  "and") and "Vasylechko, L.; Kodama, N." (surname first, semicolons) — plus
  what real strings do: initials without spaces ("G.E. Jellison"), without
  periods ("J Köser"), glued to the surname ("D.Maniglio"), hyphenated
  ("H.-J. Chang"), surname particles ("E. W. Van Stryland"), suffixes ("Jr."),
  a missing comma between two names, and an author list followed by the rest of
  a citation ("W. S. Rodney and R. J. Spindler. Refractive index of ...").
* Organisations — Materials Project, NCBI/PubChem and the databases it relays,
  the periodictable package, glass, resin and index-liquid makers — are their
  own group, labelled organisation. They are recognised from a fixed list,
  never inferred. A handbook that compiles other people's data (Palik, Weber,
  the Handbook of Mineralogy) is not an organisation: whose measurement it
  relays is unknown, so its authors are unknown.
* Two sources are in the same group if they share any (surname, initial) pair;
  groups are the connected components of that relation.
* The answer is ``unknown`` when any compared source has no parseable authors,
  or when two groups would be joined only by a surname with no initial on one
  side: that may or may not be one person, and guessing either way would be a
  guess.

The rule errs toward fewer groups. Two different people with the same surname
and first initial are merged, which can understate independence; it never
overstates it. Matching on surname alone, which merges every "Wang", is the
failure this avoids.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Organisations: a fixed, tested list. (pattern, canonical name)
# ---------------------------------------------------------------------------

ORGANISATIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), name)
    for pattern, name in (
        (r"^\s*(?:the\s+)?materials project\b|^\s*MP bulk DFT\b", "Materials Project"),
        (r"^\s*National Center for Biotechnology Information\b|^\s*NCBI\b|^\s*PubChem\b", "NCBI PubChem"),
        (r"^\s*Hazardous Substances Data Bank\b", "Hazardous Substances Data Bank (via PubChem)"),
        (r"^\s*PAC Chemical Database\b", "PAC Chemical Database, U.S. Department of Energy (via PubChem)"),
        (r"^\s*CAMEO Chemicals\b", "CAMEO Chemicals"),
        (r"^\s*periodictable\b", "periodictable package"),
        (r"^\s*(?:\d\)\s*)?SCHOTT\b|\bschott\.com\b", "SCHOTT"),
        (r"^\s*OHARA\b", "OHARA"),
        (r"^\s*HIKARI\b", "HIKARI"),
        (r"^\s*CDGM\b", "CDGM"),
        (r"^\s*HOYA\b", "HOYA"),
        (r"^\s*SUMITA\b", "SUMITA"),
        (r"^\s*NIKON\b", "NIKON"),
        (r"^\s*LZOS\b", "LZOS"),
        (r"^\s*Corning\b", "Corning"),
        (r"^\s*Cargille\b|\bcargille\.com\b|Matching Liquid", "Cargille Laboratories"),
        (r"^\s*Norland\b", "Norland Products"),
        (r"^\s*Microchem\b|^\s*MicroChem\b", "MicroChem"),
        (r"^\s*Dow Corning\b", "Dow Corning"),
        (r"^\s*Amorphous Materials,? Inc\b", "Amorphous Materials, Inc."),
        (r"^\s*Micro resist\b", "micro resist technology"),
    )
)

_PARTICLES = {"van", "von", "de", "der", "den", "del", "della", "di", "da", "du", "le", "la", "dos", "das", "ter", "ten", "mac", "st."}
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}
#  "J.", "J", "G.E.", "H.-J.", "Yu.", "Th." — tokens that are initials, not surnames.
#  Every part carries a period except a lone capital ("J Köser"): "Li" and "Wu"
#  are surnames, and reading them as initials would lose the author.
_INITIALS = re.compile(r"^(?:[A-Z][a-z]?\.-?)+$|^[A-Z]$")
#  "D.Maniglio": initials glued to a surname.
_GLUED = re.compile(r"^((?:[A-Z]\.)+)([A-Z][\w'’\-]+)$")
_ETAL = re.compile(r",?\s*\bet\s+al\.?", re.IGNORECASE)


@dataclass(frozen=True)
class Person:
    """One author, normalised: surname without accents, case, spaces or hyphens."""

    surname: str
    initial: str | None

    def label(self) -> str:
        return f"{self.surname}, {self.initial}." if self.initial else self.surname


@dataclass(frozen=True)
class AuthorParse:
    """What a source's ``authors`` field yields.

    ``kind`` is ``people``, ``organisation`` or ``missing``. ``missing`` covers an
    empty field and a field from which no name could be read; both make any
    comparison involving the source ``unknown``.
    """

    kind: str
    people: tuple[Person, ...] = ()
    organisation: str | None = None
    note: str = ""


def _fold(text: str) -> str:
    """Lower case, accents removed: 'Schröter' -> 'schroter'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _surname_key(words: list[str]) -> str:
    return re.sub(r"[^a-z]", "", _fold(" ".join(words)))


def _ascii(token: str) -> str:
    """Accents removed, case kept: 'Á.' -> 'A.', so initials match one pattern."""
    decomposed = unicodedata.normalize("NFKD", token)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _initial_of(token: str) -> str | None:
    letters = re.sub(r"[^A-Za-z]", "", _fold(token))
    return letters[0] if letters else None


def _looks_like_surname_word(word: str) -> bool:
    bare = word.strip(".,;")
    if not bare:
        return False
    if bare.lower() in _PARTICLES:
        return True
    return bare[0].isupper() and any(ch.islower() for ch in bare)


def _people_initials_first(segment: str) -> list[Person] | None:
    """'G. E. Jellison Jr.' / 'L. A. Ramey D. J. Singh' -> people; None if not names."""
    tokens: list[str] = []
    for raw in segment.split():
        glued = _GLUED.match(_ascii(raw))
        if glued:
            tokens.extend([glued.group(1), glued.group(2)])
        else:
            tokens.append(raw)

    people: list[Person] = []
    initials: list[str] = []
    surname: list[str] = []

    def flush() -> bool:
        if not surname:
            return not initials
        if not initials and len(surname) > 1:
            #  Several capitalised words and no initial is a title or a
            #  publisher ("Academic Press"), not a person.
            return False
        people.append(Person(_surname_key(surname), _initial_of(initials[0]) if initials else None))
        return True

    for token in tokens:
        bare = token.strip(",;")
        if bare.lower() in _SUFFIXES:
            continue
        if _INITIALS.match(_ascii(bare)):
            if surname:
                if not flush():
                    return None
                initials, surname = [], []
            initials.append(bare)
            continue
        if not _looks_like_surname_word(bare):
            return None
        surname.append(bare.rstrip("."))
    if not flush():
        return None
    return people or None


def _people_surname_first(text: str) -> list[Person] | None:
    """'Vasylechko, L.; Kodama, N.' -> people; None if any entry is not 'Surname, I.'."""
    people = []
    for entry in (part.strip() for part in text.split(";")):
        if not entry:
            continue
        if "," not in entry:
            return None
        surname, given = (part.strip() for part in entry.split(",", 1))
        if not surname or not all(_looks_like_surname_word(w) for w in surname.split()):
            return None
        people.append(Person(_surname_key(surname.split()), _initial_of(given) if given else None))
    return people or None


def organisation_of(text: str | None) -> str | None:
    """The canonical organisation a string names, from the fixed list, or None."""
    if not text:
        return None
    for pattern, name in ORGANISATIONS:
        if pattern.search(text):
            return name
    return None


#  Organisations recognisable in a source URL ("download/data/1993/Schott_B270.pdf").
_URL_ORGANISATIONS: dict[str, str] = {
    "schott": "SCHOTT",
    "ohara": "OHARA",
    "hoya": "HOYA",
    "cdgm": "CDGM",
    "cargille": "Cargille Laboratories",
    "norland": "Norland Products",
    "materialsproject": "Materials Project",
}


def _organisation_in_url(url: str | None) -> str | None:
    words = set(re.split(r"[^a-z0-9]+", (url or "").lower()))
    for word, name in _URL_ORGANISATIONS.items():
        if word in words:
            return name
    return None


def parse_authors(authors: str | None, *, title: str | None = None, url: str | None = None) -> AuthorParse:
    """Parse a source's ``authors`` field; fall back to its title and URL for an organisation.

    The title and URL are consulted only when ``authors`` yields no person, and
    only to recognise an organisation from the fixed lists: 64 sources in v0.22.0
    have no authors, and for a manufacturer datasheet or a database entry the
    title or the URL is where the organisation is named. They are never parsed
    for people.
    """
    text = (authors or "").strip()
    org = organisation_of(text)
    if org:
        return AuthorParse("organisation", organisation=org)
    people = _parse_people(text) if text else []
    if people:
        return AuthorParse("people", tuple(people))
    org = organisation_of(title)
    if org:
        return AuthorParse("organisation", organisation=org, note="organisation named in the title")
    org = _organisation_in_url(url)
    if org:
        return AuthorParse("organisation", organisation=org, note="organisation named in the URL")
    if not text:
        return AuthorParse("missing", note="the source has no authors")
    return AuthorParse("missing", note=f"no author name could be read from {text[:60]!r}")


def _parse_people(text: str) -> list[Person]:
    """People from an authors string in either format; empty when none can be read."""

    text = _ETAL.sub("", text).strip().rstrip(",")
    if ";" in text:
        people = _people_surname_first(text)
        if people:
            return people

    #  Initials-first: split on commas and " and ", and stop at the first segment
    #  that is not a name. That is where an author list followed by the rest of a
    #  citation ("H. H. Li. Refractive index of ...") turns into a title.
    people: list[Person] = []
    for chunk in re.split(r",\s*|\s+and\s+|\s*&\s*", text):
        #  "N. Umemura, K. Miyata, and K. Kato": the serial comma leaves "and K. Kato".
        chunk = re.sub(r"^and\s+", "", chunk.strip())
        if not chunk:
            continue
        #  "M. R. Querry. Optical constants of ..." — a name, then a sentence.
        head = re.match(r"^((?:\S+\s+)*?\S*[a-z’'\-]{2,})\.\s+(\S.*)$", chunk)
        candidate = head.group(1) if head and not _INITIALS.match(_ascii(head.group(1).split()[-1])) else chunk
        parsed = _people_initials_first(candidate.rstrip("."))
        if not parsed:
            break
        people.extend(parsed)
        if candidate is not chunk:
            break
    return people


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupResult:
    """How many independent author groups a set of sources spans.

    ``n_groups`` is None exactly when ``status`` is ``unknown``. ``components``
    lists the source ids in each group (sorted), for showing the reader which
    sources were joined.
    """

    status: str
    n_groups: int | None
    reason: str
    components: tuple[tuple[int, ...], ...] = field(default_factory=tuple)


def _canonical(components: list[set[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(group)) for group in components))


def author_groups(parses: Mapping[int, AuthorParse]) -> GroupResult:
    """Group sources (id -> parse) by shared authors. See the module docstring."""
    ids = sorted(parses)
    if not ids:
        return GroupResult("unknown", None, "no sources to compare")
    missing = [sid for sid in ids if parses[sid].kind == "missing"]
    if missing:
        return GroupResult(
            "unknown",
            None,
            "source(s) " + ", ".join(str(s) for s in missing) + " have no parseable authors",
        )

    parent = {sid: sid for sid in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pa, pb = parses[a], parses[b]
            if pa.kind == "organisation" or pb.kind == "organisation":
                if pa.kind == pb.kind and pa.organisation == pb.organisation:
                    union(a, b)
                continue
            full_a = {(p.surname, p.initial) for p in pa.people if p.initial}
            full_b = {(p.surname, p.initial) for p in pb.people if p.initial}
            if full_a & full_b:
                union(a, b)

    #  A surname with no initial cannot be told apart from a namesake. If it is
    #  the only thing that could join two otherwise separate groups, the count
    #  is unknown rather than either answer.
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if find(a) == find(b):
                continue
            pa, pb = parses[a], parses[b]
            if pa.kind != "people" or pb.kind != "people":
                continue
            bare_a = {p.surname for p in pa.people if not p.initial}
            bare_b = {p.surname for p in pb.people if not p.initial}
            all_a = {p.surname for p in pa.people}
            all_b = {p.surname for p in pb.people}
            clash = (bare_a & all_b) | (bare_b & all_a)
            if clash:
                return GroupResult(
                    "unknown",
                    None,
                    f"sources {a} and {b} share the surname {sorted(clash)[0]!r} with no initial on "
                    "one side, so whether they are one person cannot be decided",
                )

    components: dict[int, set[int]] = {}
    for sid in ids:
        components.setdefault(find(sid), set()).add(sid)
    groups = _canonical(list(components.values()))
    n = len(groups)
    if n == 1 and len(ids) > 1:
        reason = "all sources are connected by shared authors or a shared organisation"
    elif n == 1:
        reason = "one source"
    else:
        reason = f"{n} groups with no author or organisation in common"
    return GroupResult("determined", n, reason, groups)
