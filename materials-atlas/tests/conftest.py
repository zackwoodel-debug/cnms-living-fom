"""Shared fixtures. Unit tests need nothing; tests marked ``release`` read the real release.

The real release is not in CI and is 300+ MB, so those tests skip, loudly, when it
is absent: set MATERIALS_ATLAS_RELEASE to run them anywhere else.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ATLAS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ATLAS_ROOT.parent
for path in (ATLAS_ROOT, REPO_ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_RELEASE = Path("~/code/materials-db/release/materials-db-v0.22.0").expanduser()


def release_path() -> Path | None:
    candidate = Path(os.environ.get("MATERIALS_ATLAS_RELEASE", DEFAULT_RELEASE)).expanduser()
    return candidate if (candidate / "MANIFEST.json").is_file() else None


@pytest.fixture(scope="session")
def release():
    path = release_path()
    if path is None:
        pytest.skip("materials-db release not found; set MATERIALS_ATLAS_RELEASE to run release tests")
    from cnms_fom.materials_release import Release

    return Release(path)


@pytest.fixture(scope="session")
def records(release):
    """The full in-memory record set built from the real release (built once per session)."""
    from atlas.facts import build_records

    return build_records(release)
