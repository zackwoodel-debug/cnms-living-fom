"""Stable dot-path ids and safe file names.

Ids come from materials-db's stable keys, which never change for a material, so
an atlas id is stable across releases too: ``oxides_50:SiO2@amorphous`` ->
``material.oxides_50.sio2-amorphous``. File names come from display names,
which may change between releases; nothing links by file name alone (links
carry the full vault path), and a collision of either kind stops the build.
"""

from __future__ import annotations

import re
import unicodedata

#  Characters Obsidian refuses in a file name, or that break a wikilink.
_FILENAME_FORBIDDEN = re.compile(r'[\\/:*?"<>|#^\[\]]')


class IdCollision(RuntimeError):
    """Two records would get the same id or the same file path."""


def slug(text: str) -> str:
    """Lower-case ASCII, words joined by '-'. Dots become '_' so ids stay dot-paths.

    'SiO2@amorphous' -> 'sio2-amorphous'; 'AlGaAs@x0.097' -> 'algaas-x0_097';
    'SCHOTT-optical/N-BK7HT' -> 'schott-optical-n-bk7ht'.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()
    folded = folded.replace(".", "_")
    folded = re.sub(r"[^a-z0-9_]+", "-", folded)
    folded = re.sub(r"-{2,}", "-", folded).strip("-_")
    if not folded:
        raise ValueError(f"{text!r} has no characters an id can use")
    return folded


def material_id_for(key: str) -> str:
    family, selection = key.split(":", 1)
    return f"material.{family}.{slug(selection)}"


def source_id_for(source_id: int) -> str:
    return f"source.{source_id}"


def family_id_for(family: str) -> str:
    return f"family.{family}"


def property_id_for(name: str) -> str:
    return f"property.{slug(name)}"


def dataset_id_for(material_note_id: str, dataset_label: str) -> str:
    return f"{material_note_id}#dataset.{slug(dataset_label)}"


def safe_filename(name: str) -> str:
    """A display name made safe for a file name; the original goes into aliases."""
    cleaned = _FILENAME_FORBIDDEN.sub("-", name)
    cleaned = re.sub(r"\s*-\s*-\s*", " - ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .")
    if not cleaned:
        raise ValueError(f"{name!r} has no characters a file name can use")
    return cleaned


def short_title(title: str | None, words: int = 6) -> str:
    """The first few words of a title, for a source note's file name."""
    if not title:
        return ""
    tokens = re.findall(r"[^\s]+", re.sub(r"<[^>]+>", " ", title))
    return safe_filename(" ".join(tokens[:words])) if tokens else ""


def check_unique(pairs: list[tuple[str, str]], what: str) -> None:
    """Raise IdCollision if two owners share a value (case-insensitively, as macOS files do)."""
    seen: dict[str, str] = {}
    for owner, value in pairs:
        key = value.casefold()
        if key in seen and seen[key] != owner:
            raise IdCollision(f"{what} collision: {seen[key]!r} and {owner!r} both map to {value!r}")
        seen[key] = owner
