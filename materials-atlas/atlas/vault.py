"""Write notes to an Obsidian vault folder, deterministically.

The build renders into a fresh folder and then syncs it over the target: files
whose bytes changed are rewritten, files the new build does not produce are
removed, and Obsidian's own state in ``.obsidian/`` (workspace, plugins) is
left alone apart from the graph settings the build owns.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import yamlfm
from .render import Note

#  Files under .obsidian/ the build writes. Everything else there is Obsidian's.
OWNED_OBSIDIAN = {".obsidian/graph.json"}

GRAPH_SETTINGS = {
    "collapse-filter": False,
    "search": '-path:"Materials/Glass catalogs"',
    "showTags": False,
    "showAttachments": False,
    "hideUnresolved": True,
    "showOrphans": True,
}


def render_note(note: Note) -> str:
    parts = [yamlfm.dump(note.frontmatter), f"# {note.title}\n"]
    if note.intro:
        parts.append(f"\n{note.intro}\n")
    for section in note.sections:
        parts.append(f"\n## {section.heading}\n\n{section.body.rstrip()}\n")
    if note.footer:
        parts.append(f"\n---\n\n{note.footer}\n")
    return "".join(parts)


def render_files(notes: list[Note]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for note in notes:
        if note.path in files:
            raise RuntimeError(f"two notes write {note.path}")
        files[note.path] = render_note(note).encode("utf-8")
    files[".obsidian/graph.json"] = (json.dumps(GRAPH_SETTINGS, indent=2, sort_keys=True) + "\n").encode()
    return files


def write_tree(root: Path, files: dict[str, bytes]) -> dict:
    """Make ``root`` hold exactly ``files`` (plus Obsidian's own .obsidian state)."""
    root.mkdir(parents=True, exist_ok=True)
    written = unchanged = removed = 0
    for rel, data in sorted(files.items()):
        target = root / rel
        if target.is_file() and target.read_bytes() == data:
            unchanged += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written += 1
    keep = set(files)
    for path in sorted(root.rglob("*"), reverse=True):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".obsidian") and rel not in OWNED_OBSIDIAN:
            continue
        if path.is_file() and rel not in keep:
            path.unlink()
            removed += 1
        elif path.is_dir() and not any(path.iterdir()) and rel != ".obsidian":
            path.rmdir()
    return {"written": written, "unchanged": unchanged, "removed": removed, "files": len(files)}


def replace_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
