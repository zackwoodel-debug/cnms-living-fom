"""Open the atlas index, read-only, and refuse one that is missing or stale.

A missing or stale index is an error, not "no data": an assistant told "the atlas has
nothing on GaAs" when the atlas was never built has been told something false about the
literature (RESEARCH_ASSISTANT.md Sec. 2, "a failed search is an error").
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from cnms_fom.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
#  The index layout this reader understands (materials-atlas/atlas/index.py INDEX_SCHEMA).
INDEX_SCHEMA = 1


class AtlasUnavailable(RuntimeError):
    """The atlas index cannot be used: not built, unreadable, or built from another release."""


@dataclass
class AtlasIndex:
    path: Path
    connection: sqlite3.Connection
    meta: dict

    @property
    def version(self) -> str:
        return str(self.meta["release_version"])

    @property
    def sha256(self) -> str:
        return str(self.meta["db_sha256"])

    @property
    def origin(self) -> str:
        return f"materials-db v{self.version} (reference)"

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.connection.execute(sql, params).fetchall()

    def close(self) -> None:
        self.connection.close()


def index_path() -> Path:
    configured = Path(get_settings().materials_atlas_index).expanduser()
    return configured if configured.is_absolute() else REPO_ROOT / configured


def open_atlas(path: Path | None = None, *, expected_version: str | None = None) -> AtlasIndex:
    """Open the index read-only. Raises AtlasUnavailable when it is missing or stale."""
    path = path or index_path()
    expected = expected_version if expected_version is not None else get_settings().materials_atlas_release
    if not path.is_file():
        raise AtlasUnavailable(
            f"The materials atlas index is not built ({path} does not exist). Build it with "
            "`cnms-fom atlas build`. Until then the atlas cannot be searched, which is not the same "
            "as the literature having no value."
        )
    try:
        connection = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, check_same_thread=False)
        connection.execute("PRAGMA query_only = ON")
        connection.row_factory = sqlite3.Row
        meta = {row["key"]: json.loads(row["value"]) for row in connection.execute("SELECT key, value FROM meta")}
    except sqlite3.Error as exc:
        raise AtlasUnavailable(f"The atlas index at {path} cannot be read: {exc}") from exc
    if meta.get("index_schema") != INDEX_SCHEMA:
        connection.close()
        raise AtlasUnavailable(
            f"The atlas index at {path} has layout {meta.get('index_schema')!r}; this platform reads layout "
            f"{INDEX_SCHEMA}. Rebuild it with `cnms-fom atlas build`."
        )
    built = str(meta.get("release_version"))
    if expected and built != str(expected):
        connection.close()
        raise AtlasUnavailable(
            f"The atlas index is stale: it was built from materials-db v{built}, and this platform is "
            f"configured for v{expected} (MATERIALS_ATLAS_RELEASE). Rebuild it with `cnms-fom atlas build`."
        )
    if meta.get("sample"):
        connection.close()
        raise AtlasUnavailable(
            f"The atlas index at {path} is a five-material sample build, not the full atlas. Rebuild it "
            "with `cnms-fom atlas build`."
        )
    return AtlasIndex(path=path, connection=connection, meta=meta)


def status(path: Path | None = None) -> dict:
    """For /health/ready and the CLI: whether the atlas can be used, and what it holds."""
    try:
        atlas = open_atlas(path)
    except AtlasUnavailable as exc:
        return {"ok": False, "index": str(path or index_path()), "error": str(exc)}
    try:
        counts = {
            "materials": atlas.query("SELECT COUNT(*) FROM notes WHERE type = 'material'")[0][0],
            "values": atlas.query('SELECT COUNT(*) FROM "values"')[0][0],
            "chunks": atlas.query("SELECT COUNT(*) FROM chunks")[0][0],
        }
    finally:
        atlas.close()
    return {
        "ok": True,
        "index": str(atlas.path),
        "release": atlas.version,
        "db_sha256": atlas.sha256,
        "retrieval": "lexical (SQLite FTS5); no dense leg",
        **counts,
    }
