"""The eval machinery: question files, the gate, and the baselines it reads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parents[1] / "evals"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("run_evals", EVALS / "run_evals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_question_files_load_for_one_release(runner):
    release, smoke = runner.load_questions("smoke")
    _, full = runner.load_questions("full")
    assert release == "0.22.0"
    assert len(smoke) == 40 and len(full) == 106
    assert len({q["id"] for q in full}) == len(full)
    assert sum(q["category"] == "adversarial" for q in full) == 10
    #  Generated questions name materials by id, from SQL on the release, not by atlas note id.
    generated = [q for q in full if q["suite"] == "full"]
    assert all("notes" not in q for q in generated)
    assert all(q.get("material_ids") or q["kind"] == "absent" for q in generated)


def _report(lookups=(10, 10), hit1=(5, 5), hit5=(5, 5), e2e=None):
    report = {
        "release": "0.22.0",
        "deterministic": {"summary": {"lookups_passed": list(lookups), "search_hit@1": list(hit1), "search_hit@5": list(hit5)}},
    }
    if e2e:
        report["end_to_end"] = {"summary": e2e}
    return report


def test_the_gate_demands_a_perfect_deterministic_run(runner):
    assert runner.gate(_report(), "smoke", 0.05) == []
    assert runner.gate(_report(lookups=(9, 10)), "smoke", 0.05) == ["deterministic lookups_passed: 9/10"]


def test_the_gate_holds_end_to_end_to_its_baseline(runner, tmp_path, monkeypatch):
    baselines = tmp_path / "baselines.json"
    baselines.write_text(json.dumps({"end_to_end": {"m:1b": {"smoke": {"pass_rate": 0.8, "framing_withheld_rate": 0.1}}}}))
    monkeypatch.setattr(runner, "BASELINES", baselines)
    ok = {"model": "m:1b", "passed": [8, 10], "framing_withheld": [1, 10]}
    assert runner.gate(_report(e2e=ok), "smoke", 0.05) == []
    worse = {"model": "m:1b", "passed": [6, 10], "framing_withheld": [4, 10]}
    failures = runner.gate(_report(e2e=worse), "smoke", 0.05)
    assert len(failures) == 2 and failures[0].startswith("end-to-end pass rate")
    unknown = {"model": "other:7b", "passed": [10, 10], "framing_withheld": [0, 10]}
    assert "no end-to-end baseline" in runner.gate(_report(e2e=unknown), "smoke", 0.05)[0]


def test_updating_a_baseline_records_rates(runner, tmp_path, monkeypatch):
    baselines = tmp_path / "baselines.json"
    monkeypatch.setattr(runner, "BASELINES", baselines)
    runner.update_baseline(_report(e2e={"model": "m:1b", "passed": [7, 10], "framing_withheld": [1, 4]}), "full")
    recorded = json.loads(baselines.read_text())["end_to_end"]["m:1b"]["full"]
    assert recorded == {"pass_rate": 0.7, "framing_withheld_rate": 0.25, "questions": 10, "release": "0.22.0"}
