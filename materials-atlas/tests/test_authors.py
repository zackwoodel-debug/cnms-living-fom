"""Rubric 6.4: author parsing and author-group independence."""

from __future__ import annotations

import pytest
from atlas.authors import AuthorParse, Person, author_groups, organisation_of, parse_authors


def names(parse: AuthorParse) -> list[tuple[str, str | None]]:
    return [(p.surname, p.initial) for p in parse.people]


# --- the two formats --------------------------------------------------------


def test_initials_first_with_commas_and_and():
    parse = parse_authors("J. Walling, O. Peterson and H. Jenssen.")
    assert parse.kind == "people"
    assert names(parse) == [("walling", "j"), ("peterson", "o"), ("jenssen", "h")]


def test_surname_first_with_semicolons():
    parse = parse_authors("Vasylechko, L.; Kodama, N.")
    assert names(parse) == [("vasylechko", "l"), ("kodama", "n")]


def test_serial_comma_keeps_the_last_author():
    parse = parse_authors("N. Umemura, K. Miyata, and K. Kato. New data on the optical properties")
    assert names(parse) == [("umemura", "n"), ("miyata", "k"), ("kato", "k")]


# --- what real strings do ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("G.E. Jellison Jr.", [("jellison", "g")]),
        ("J. Rheims, J Köser, T Wriedt.", [("rheims", "j"), ("koser", "j"), ("wriedt", "t")]),
        ("A. Bucciarelli, V. Mulloni, D.Maniglio.", [("bucciarelli", "a"), ("mulloni", "v"), ("maniglio", "d")]),
        ("H.-J. Chang, C. Lopez-Zelaya.", [("chang", "h"), ("lopezzelaya", "c")]),
        ("E. W. Van Stryland, L. De Stefano.", [("vanstryland", "e"), ("destefano", "l")]),
        ("E. Sani and A. Dell'Oro.", [("sani", "e"), ("delloro", "a")]),
        ("H. Schröter.", [("schroter", "h")]),
        ("Á. Barroso, B. Kemper.", [("barroso", "a"), ("kemper", "b")]),
        ("H. H. Li.", [("li", "h")]),
        ("S. T. Wu, C. Wu.", [("wu", "s"), ("wu", "c")]),
        ("J. H McFee.", [("mcfee", "j")]),
        #  A missing comma between two names.
        ("J. A. Kolopus, L. A. Ramey D. J. Singh.", [("kolopus", "j"), ("ramey", "l"), ("singh", "d")]),
        ("A. Smith et al.", [("smith", "a")]),
    ],
)
def test_real_formats(text, expected):
    assert names(parse_authors(text)) == expected


def test_an_author_list_followed_by_the_rest_of_the_citation_stops_at_the_title():
    parse = parse_authors(
        "W. S. Rodney and R. J. Spindler. Refractive index of cesium bromide for ultraviolet, visible, and infrared"
    )
    assert names(parse) == [("rodney", "w"), ("spindler", "r")]
    parse = parse_authors("M. R. Querry. Optical constants of minerals and other materials, Contractor Report")
    assert names(parse) == [("querry", "m")]
    parse = parse_authors("I. H. Malitson, as cited in Handbook of Optics, 3rd edition")
    assert names(parse) == [("malitson", "i")]


# --- organisations ------------------------------------------------------------


@pytest.mark.parametrize(
    ("authors", "title", "organisation"),
    [
        ("periodictable Python package", None, "periodictable package"),
        ("National Center for Biotechnology Information", "PubChem", "NCBI PubChem"),
        (None, "SCHOTT Zemax catalog 2017-01-20b (obtained from http://www.schott.com)", "SCHOTT"),
        (None, "OHARA Zemax catalog 2017-11-30", "OHARA"),
        (None, "MP bulk DFT density used as a film approximation (batch-2 materials)", "Materials Project"),
        (None, "Hazardous Substances Data Bank (HSDB) via PubChem CID 6348: density", "Hazardous Substances Data Bank (via PubChem)"),
        ("Microchem PMMA Datasheet, 2001", None, "MicroChem"),
        (None, "BK-7 Matching Liquid Data Sheet 27-March-2023", "Cargille Laboratories"),
        ("Amorphous Materials, Inc.", "AMTIR-6 (As2S3) product datasheet", "Amorphous Materials, Inc."),
    ],
)
def test_organisations_are_recognised_from_the_fixed_list(authors, title, organisation):
    parse = parse_authors(authors, title=title)
    assert parse.kind == "organisation"
    assert parse.organisation == organisation


def test_a_title_is_never_parsed_for_people():
    parse = parse_authors(None, title="J. Smith. A paper about glass")
    assert parse.kind == "missing"


def test_organisation_list_does_not_swallow_a_person_called_dow():
    assert organisation_of("D. Dow, A. Smith.") is None


# --- missing authors -> unknown, never independent ---------------------------


@pytest.mark.parametrize(
    "authors",
    [
        None,
        "",
        "Handbook of Optical Constants of Solids, Edward D. Palik, ed. Academic Press, Boston, 1985",
        "B 270 datasheet, August 1993",
    ],
)
def test_missing_or_unreadable_authors_are_missing(authors):
    #  (The B 270 datasheet is SCHOTT's only by its URL; without one it stays missing.)
    assert parse_authors(authors, title="Literature density estimate").kind == "missing"


def test_a_missing_author_makes_the_comparison_unknown_not_independent():
    result = author_groups(
        {
            1: parse_authors("A. Smith."),
            2: parse_authors(None),
        }
    )
    assert result.status == "unknown"
    assert result.n_groups is None
    assert "2" in result.reason


# --- groups -------------------------------------------------------------------


def test_shared_surname_and_initial_is_one_group():
    result = author_groups(
        {
            1: parse_authors("S. T. Wu, C. Wu, W. Warenghem."),
            2: parse_authors("J. Li, C. H. Wen, S. Gauza, R. Lu, S. T. Wu."),
        }
    )
    assert (result.status, result.n_groups) == ("determined", 1)


def test_groups_are_connected_components():
    #  1 and 2 share Kato, 2 and 3 share Badikov: one group, though 1 and 3 share nobody.
    result = author_groups(
        {
            1: parse_authors("K. Kato, H. Shirahata."),
            2: parse_authors("K. Kato, V. V. Badikov."),
            3: parse_authors("V. Badikov, D. Badikov."),
            4: parse_authors("G. D. Boyd, H. Kasper."),
        }
    )
    assert result.n_groups == 2
    assert result.components == ((1, 2, 3), (4,))


def test_same_surname_different_initial_is_not_merged():
    result = author_groups({1: parse_authors("J. Wang."), 2: parse_authors("L. Wang.")})
    assert result.n_groups == 2


def test_surname_only_collision_is_unknown():
    result = author_groups(
        {
            1: AuthorParse("people", (Person("querry", None),)),
            2: parse_authors("M. R. Querry, A. Smith."),
        }
    )
    assert result.status == "unknown"
    assert "querry" in result.reason


def test_organisations_are_their_own_group():
    result = author_groups(
        {
            1: parse_authors(None, title="SCHOTT Zemax catalog 2017"),
            2: parse_authors("SCHOTT glass data sheet"),
            3: parse_authors("S. Tsuda, S. Yamaguchi."),
        }
    )
    assert result.n_groups == 2
    assert result.components == ((1, 2), (3,))


def test_one_source_is_one_group():
    result = author_groups({7: parse_authors("H. R. Philipp.")})
    assert (result.status, result.n_groups, result.reason) == ("determined", 1, "one source")


# --- named real cases (release) -------------------------------------------------


def test_every_release_source_parses_and_only_handbooks_and_bare_notes_are_missing(release):
    connection = release.connect()
    kinds = {}
    for sid, authors, title, url in connection.execute("SELECT source_id, authors, title, url FROM sources"):
        kinds[sid] = parse_authors(authors, title=title, url=url).kind
    missing = sorted(sid for sid, kind in kinds.items() if kind == "missing")
    #  A bare density note (4) and three compiled handbooks: Handbook of Mineralogy (81),
    #  Palik (182), Weber (402). Their original measurers are unknown, so they stay unknown.
    assert missing == [4, 81, 182, 402]
    assert parse_authors("B 270 datasheet, August 1993", url="https://refractiveindex.info/download/data/1993/Schott_B270.pdf").organisation == "SCHOTT"
