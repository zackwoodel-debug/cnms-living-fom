"""Shared fixture loading for the pySEA tests.

A module rather than a conftest fixture: several tests want the envelope as a
plain dict they can mutate, and a function that re-reads from disk each time makes
mutation in one test invisible to the next.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "pysea"

EXPERIMENTAL = "example_experimental.json"
SIMULATION = "example_simulation.json"
INVALID = "example_invalid.json"


def fixture_envelope(name: str) -> dict:
    """A fresh copy of one fixture, safe to mutate."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
