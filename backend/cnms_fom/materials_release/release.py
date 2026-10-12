"""Open a materials-db release read-only and prove it is the release it claims to be.

A release folder holds ``materials-db-v<version>.sqlite`` plus ``MANIFEST.json``
(build facts and row counts), ``SHA256SUMS``, ``material_registry.csv`` (the
permanent id -> stable key map) and ``family_tables/``. Everything this platform
says about a release value rests on three checks made here:

* the sqlite's SHA-256 is the one ``SHA256SUMS`` lists, so the file is the one
  that was published;
* every table count, the dataset count and every per-family count equal the
  MANIFEST, so nothing was dropped or added on the way in;
* the registry covers exactly the ``materials`` table, so a material id means
  the same material everywhere.

Any mismatch raises ``ReleaseError``. A partial read of a reference dataset that
reports itself as complete is worse than no read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from urllib.parse import quote


class ReleaseError(RuntimeError):
    """The release on disk is not the release its MANIFEST describes."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RegistryEntry:
    """One row of ``material_registry.csv``: a permanent id and its stable key."""

    material_id: int
    key: str
    name: str

    @property
    def family(self) -> str:
        return self.key.split(":", 1)[0]

    @property
    def selection(self) -> str:
        return self.key.split(":", 1)[1]

    @property
    def polymorph(self) -> str | None:
        """The phase named in the key (``oxides_50:TiO2@rutile`` -> ``rutile``), if any.

        Only the earliest families encode a polymorph this way; for every other
        key there is none, and None is the answer rather than a guess.
        """
        return self.selection.split("@", 1)[1] if "@" in self.selection else None


class Release:
    """A materials-db release folder. Construction reads only the MANIFEST."""

    def __init__(self, root: Path | str) -> None:
        path = Path(root).expanduser()
        if path.is_file() and path.suffix == ".sqlite":
            path = path.parent
        manifest_path = path / "MANIFEST.json"
        if not manifest_path.is_file():
            raise ReleaseError(f"{path} is not a materials-db release: no MANIFEST.json")
        self.root = path.resolve()
        self.manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("dataset") != "materials-db":
            raise ReleaseError(f"{manifest_path} does not describe a materials-db release")
        self.version: str = str(self.manifest["version"])
        self.sqlite_path = self.root / f"materials-db-v{self.version}.sqlite"
        if not self.sqlite_path.is_file():
            raise ReleaseError(f"{self.sqlite_path} is missing")

    @classmethod
    def detect(cls, path: Path | str) -> Release | None:
        """The release a sqlite file belongs to, or None when it is not part of one."""
        candidate = Path(path).expanduser()
        if candidate.is_file() and not (candidate.parent / "MANIFEST.json").is_file():
            return None
        try:
            release = cls(candidate)
        except ReleaseError:
            return None
        if candidate.is_file() and candidate.resolve() != release.sqlite_path:
            return None
        return release

    # ------------------------------------------------------------------
    # The database
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """A read-only connection: ``mode=ro`` at open, ``query_only`` on top."""
        uri = f"file:{quote(str(self.sqlite_path))}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only = ON")
        connection.row_factory = sqlite3.Row
        return connection

    @cached_property
    def sha256(self) -> str:
        return _sha256_file(self.sqlite_path)

    def verify_checksums(self) -> str:
        """Check every file ``SHA256SUMS`` lists; return the sqlite's SHA-256.

        All of them, not just the database: the family tables, the registry and
        the MANIFEST are read too, and a family table edited after release would
        change quoted flags without changing a single database row.
        """
        sums = self.root / "SHA256SUMS"
        if not sums.is_file():
            raise ReleaseError(f"{sums} is missing, so the release cannot be verified")
        listed: dict[str, str] = {}
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                listed[parts[1].lstrip("*").removeprefix("./")] = parts[0].lower()
        if self.sqlite_path.name not in listed:
            raise ReleaseError(f"SHA256SUMS does not list {self.sqlite_path.name}")
        problems = []
        for name, expected in sorted(listed.items()):
            path = self.root / name
            if not path.is_file():
                problems.append(f"{name}: listed but missing")
                continue
            actual = self.sha256 if path == self.sqlite_path else _sha256_file(path)
            if actual != expected:
                problems.append(f"{name}: sha256 {actual}, SHA256SUMS says {expected}")
        if problems:
            raise ReleaseError("release files do not match SHA256SUMS:\n  " + "\n  ".join(problems))
        return self.sha256

    def assert_manifest_counts(self, connection: sqlite3.Connection) -> dict:
        """Every count the MANIFEST states, checked against the data. Raises on any mismatch."""
        counts = self.manifest.get("counts", {})
        problems: list[str] = []
        observed: dict[str, object] = {"tables": {}}

        for table, expected in sorted(counts.get("tables", {}).items()):
            actual = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            observed["tables"][table] = actual
            if actual != expected:
                problems.append(f"{table}: MANIFEST {expected}, database {actual}")

        datasets = connection.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT material_id, dataset_label FROM optical_dispersion)"
        ).fetchone()[0]
        observed["datasets"] = datasets
        if "datasets" in counts and datasets != counts["datasets"]:
            problems.append(f"datasets: MANIFEST {counts['datasets']}, database {datasets}")

        registry = self.registry()
        ids = {row[0] for row in connection.execute("SELECT material_id FROM materials")}
        if set(registry) != ids:
            problems.append(
                f"material_registry.csv and materials differ: {len(set(registry) ^ ids)} ids"
            )
        names = dict(connection.execute("SELECT material_id, name FROM materials").fetchall())
        renamed = [mid for mid, entry in registry.items() if names.get(mid) != entry.name]
        if renamed:
            problems.append(f"registry names differ from materials.name for ids {renamed[:5]}")

        family_of = {mid: entry.family for mid, entry in registry.items()}
        per_family_materials = Counter(family_of.values())
        per_family_datasets: Counter = Counter()
        per_family_rows: Counter = Counter()
        for mid, n_rows in connection.execute(
            "SELECT material_id, COUNT(*) FROM optical_dispersion GROUP BY material_id"
        ):
            per_family_rows[family_of.get(mid)] += n_rows
        for (mid,) in connection.execute(
            "SELECT material_id FROM (SELECT DISTINCT material_id, dataset_label FROM optical_dispersion)"
        ):
            per_family_datasets[family_of.get(mid)] += 1
        per_family_density: Counter = Counter()
        for (mid,) in connection.execute(
            "SELECT DISTINCT material_id FROM physical_properties WHERE density_g_cm3 IS NOT NULL"
        ):
            per_family_density[family_of.get(mid)] += 1

        stated = counts.get("per_family", {})
        for family in sorted(set(stated) | set(per_family_materials)):
            expected = stated.get(family)
            if expected is None:
                problems.append(f"family {family} has materials but no MANIFEST entry")
                continue
            for field, actual in (
                ("materials", per_family_materials[family]),
                ("datasets", per_family_datasets[family]),
                ("optical_rows", per_family_rows[family]),
                ("with_density", per_family_density[family]),
            ):
                if field in expected and expected[field] != actual:
                    problems.append(
                        f"{family}.{field}: MANIFEST {expected[field]}, database {actual}"
                    )
        observed["families"] = len(per_family_materials)

        if problems:
            raise ReleaseError(
                f"materials-db v{self.version} does not match its MANIFEST:\n  "
                + "\n  ".join(problems)
            )
        return observed

    # ------------------------------------------------------------------
    # Files beside the database
    # ------------------------------------------------------------------

    @cached_property
    def _registry(self) -> dict[int, RegistryEntry]:
        path = self.root / "material_registry.csv"
        if not path.is_file():
            raise ReleaseError(f"{path} is missing")
        entries: dict[int, RegistryEntry] = {}
        keys: set[str] = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                entry = RegistryEntry(int(row["material_id"]), row["key"], row["name"])
                if entry.material_id in entries or entry.key in keys:
                    raise ReleaseError(f"material_registry.csv repeats {entry.material_id} / {entry.key}")
                if ":" not in entry.key:
                    raise ReleaseError(f"registry key {entry.key!r} has no family prefix")
                entries[entry.material_id] = entry
                keys.add(entry.key)
        return entries

    def registry(self) -> dict[int, RegistryEntry]:
        return dict(self._registry)

    def families(self) -> list[str]:
        """Family tables present in ``family_tables/`` (not the gap lists), sorted."""
        return sorted(
            path.stem
            for path in (self.root / "family_tables").glob("*.csv")
            if not path.stem.endswith("_gaps")
        )

    def family_table(self, family: str) -> list[dict[str, str]]:
        """A family table's rows as the strings the release wrote. Nothing is parsed here."""
        path = self.root / "family_tables" / f"{family}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def gap_tables(self) -> dict[str, list[dict[str, str]]]:
        """``*_gaps.csv`` rows keyed by the family each list belongs to.

        The file names are singular (``liquid_gaps.csv`` belongs to ``liquids``);
        each must map to exactly one family table or the release has a gap list
        nothing can be attached to, which is an error, not a skip.
        """
        families = set(self.families())
        out: dict[str, list[dict[str, str]]] = {}
        for path in sorted((self.root / "family_tables").glob("*_gaps.csv")):
            stem = path.stem.removesuffix("_gaps")
            matches = [name for name in (stem, stem + "s", stem + "es") if name in families]
            if len(matches) != 1:
                raise ReleaseError(f"{path.name} matches families {matches}, expected exactly one")
            with path.open(newline="", encoding="utf-8") as handle:
                out[matches[0]] = list(csv.DictReader(handle))
        return out

    def file_text(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")
