"""What kind of number a release value is: measured, a model fit, or a calculation.

Three rules, each read from the release rather than decided here:

* **Optical datasets** are measured or a model fit of the dielectric function,
  by materials-db's own ``scripts/dataset_kind.py`` (the rule the release used
  to pick primary datasets and to keep model fits out of the consensus). It is
  imported read-only from the materials-db checkout, and only after proving the
  checkout is the one that built the release: the refractiveindex.info commit
  must equal the MANIFEST's, that checkout must be clean, and the helper files
  must be unchanged since the release's git commit. Any difference stops the
  caller, because the same helper on a different checkout can classify a page
  differently and nothing downstream would notice.
* **Densities** by the origin in their label (``density_MP_DFT`` and so on).
* **SLDs and dielectric constants** are calculated, always: the release says
  "DFT densities, all SLDs and all descriptors are calculated" and that the
  dielectric constants are Materials Project DFPT values.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType

from .labels import LabelError, PhysicalLabel
from .release import Release, ReleaseError

#  density kind -> what it means, in the words the notes and tools use.
DENSITY_KINDS: dict[str, str] = {
    "literature": "a literature or datasheet value",
    "calculated_dft": "a Materials Project DFT calculation, not a measurement",
    "bulk_approximation": "a bulk value standing in for a film, not a film measurement",
    "calculated_from_lattice": "computed from experimental lattice parameters, not measured directly",
}


@dataclass(frozen=True)
class PhysicalKind:
    """The kind of one physical value and the measured/calculated class it falls in."""

    kind: str
    #  "measured" for a literature measurement, "calculated" for everything computed.
    #  This is the distinction FOM_PROOF Sec. 2.3 keeps apart.
    provenance: str
    reason: str


def density_kind(origin: str | None) -> PhysicalKind:
    """Classify a density by the origin its label states. From the label only."""
    label = f"density_{origin}" if origin else "density"
    if origin == "MP_DFT":
        return PhysicalKind("calculated_dft", "calculated", f"{label}: {DENSITY_KINDS['calculated_dft']}")
    if origin == "bulk_elemental_approximation":
        return PhysicalKind(
            "bulk_approximation", "calculated", f"{label}: {DENSITY_KINDS['bulk_approximation']}"
        )
    if origin == "experimental lattice params":
        return PhysicalKind(
            "calculated_from_lattice", "calculated", f"{label}: {DENSITY_KINDS['calculated_from_lattice']}"
        )
    if origin == "literature" or (origin or "").startswith("literature ("):
        return PhysicalKind("literature", "measured", f"{label}: {DENSITY_KINDS['literature']}")
    raise LabelError(f"density origin {origin!r} is not one the release documents")


def physical_kind(label: PhysicalLabel) -> PhysicalKind:
    """Classify any physical value. Raises ``LabelError`` for an origin nobody documented."""
    if label.quantity == "density":
        return density_kind(label.origin)
    if label.quantity.startswith(("xray_sld", "neutron_sld")):
        if not (label.method or "").startswith("periodictable"):
            raise LabelError(f"SLD label {label.raw!r} has an undocumented method")
        return PhysicalKind(
            "calculated_sld",
            "calculated",
            f"{label.quantity} | {label.method}: calculated with the periodictable package, not measured",
        )
    if label.quantity.startswith("dielectric_"):
        if label.method != "MP_DFPT":
            raise LabelError(f"dielectric label {label.raw!r} has an undocumented method")
        return PhysicalKind(
            "calculated_dfpt",
            "calculated",
            f"{label.quantity} | MP_DFPT: a 0 K Materials Project DFPT calculation, not a measurement",
        )
    raise LabelError(f"no kind rule for {label.raw!r}")


# ---------------------------------------------------------------------------
# materials-db's helpers, pinned to the release
# ---------------------------------------------------------------------------

#  Files whose content decides a dataset's kind or primacy. They are compared
#  against the release's git commit before either helper is imported.
HELPER_FILES: tuple[str, ...] = (
    "scripts/dataset_kind.py",
    "scripts/generate_ml_release_set.py",
    "scripts/fluoride_nitride_sulfide_material_list.py",
    "scripts/pure_element_material_list.py",
    "data/step1_selections*.json",
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    #  --no-optional-locks: a plain `git status` refreshes the index's stat cache
    #  and takes index.lock to do it, which is a write into a checkout this
    #  package promises only to read.
    return subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )


class DatasetKindHelper:
    """``dataset_kind.model_fit_reason`` and ``primary_pages``, verified before use."""

    def __init__(self, release: Release, materials_db_root: Path | str | None = None) -> None:
        self.release = release
        root = Path(materials_db_root).expanduser() if materials_db_root else release.root.parents[1]
        self.root = root.resolve()
        if not (self.root / "scripts" / "dataset_kind.py").is_file():
            raise ReleaseError(
                f"{self.root} is not a materials-db checkout (no scripts/dataset_kind.py); "
                "set MATERIALS_DB_ROOT"
            )
        self.provenance = self._verify()

    def _verify(self) -> dict:
        manifest = self.release.manifest
        release_commit = manifest.get("git_commit")
        ri_expected = manifest.get("refractiveindex_info_commit")
        if not release_commit or not ri_expected:
            raise ReleaseError("MANIFEST.json lacks git_commit or refractiveindex_info_commit")

        ri_checkout = self.root / "refractiveindex_db"
        head = _git(ri_checkout, "rev-parse", "HEAD")
        if head.returncode != 0:
            raise ReleaseError(f"cannot read the refractiveindex.info commit: {head.stderr.strip()}")
        ri_actual = head.stdout.strip()
        if ri_actual != ri_expected:
            raise ReleaseError(
                f"refractiveindex.info checkout is at {ri_actual}, the release was built from "
                f"{ri_expected}: dataset kinds could differ from the release's, so they are not read"
            )
        dirty = _git(ri_checkout, "status", "--porcelain")
        if dirty.returncode != 0 or dirty.stdout.strip():
            raise ReleaseError("the refractiveindex.info checkout has local changes")

        if _git(self.root, "cat-file", "-e", f"{release_commit}^{{commit}}").returncode != 0:
            raise ReleaseError(f"the release commit {release_commit} is not in {self.root}")
        changed = _git(self.root, "diff", "--name-only", release_commit, "--", *HELPER_FILES)
        if changed.returncode != 0:
            raise ReleaseError(f"cannot compare helper files: {changed.stderr.strip()}")
        if changed.stdout.strip():
            raise ReleaseError(
                "materials-db helper files differ from the release commit "
                f"{release_commit[:12]}: {changed.stdout.split()}"
            )
        return {
            "materials_db_release_commit": release_commit,
            "refractiveindex_info_commit": ri_actual,
            "helper_files_unchanged_since_release_commit": list(HELPER_FILES),
        }

    def _load(self, name: str) -> ModuleType:
        """Import a materials-db script by path without writing bytecode under it."""
        path = self.root / "scripts" / f"{name}.py"
        module_name = f"_materials_db_{name}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ReleaseError(f"cannot import {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        return module

    @cached_property
    def _dataset_kind(self) -> ModuleType:
        return self._load("dataset_kind")

    def model_fit_reason(self, raw_record_table: str) -> str | None:
        """Why a refractiveindex.info page is a model fit (quoted evidence), or None if measured."""
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            return self._dataset_kind.model_fit_reason(raw_record_table)
        finally:
            sys.dont_write_bytecode = previous

    def primary_pages(self) -> dict[str, tuple[str, str | None]]:
        """material name -> (family, primary data path), as the release's family tables name them."""
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            module = self._load("generate_ml_release_set")
            return dict(module.primary_pages(self.release.root))
        finally:
            sys.dont_write_bytecode = previous
