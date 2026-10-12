#!/usr/bin/env python
"""Run the atlas retrieval evals (materials-atlas/CLAUDE.md section 10).

    python materials-atlas/evals/run_evals.py            # deterministic: the tools, no model
    python materials-atlas/evals/run_evals.py --e2e      # also end to end through the assistant (Ollama)

Deterministic mode calls the atlas tools directly on the built index and reports, per
question, whether the right notes and values came back, plus hit@1 / hit@5 for searches.
End-to-end mode asks the CNMS assistant each question (the same agent loop /rag/chat and
Quark use, with every tool offered) and checks whether the answer quotes the expected id,
label and value and avoids what it must not say. Results go to ``evals/results/``.

Never tune the data to pass an eval; tune the index, the tools or the prompt, and say which.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.dont_write_bytecode = True

import yaml  # noqa: E402

from cnms_fom.atlas import open_atlas  # noqa: E402
from cnms_fom.atlas import tools as atlas_tools  # noqa: E402


def _value_matches(expected: dict, got: dict) -> bool:
    for key, want in expected.items():
        if key == "label":
            if got.get("label") != want:
                return False
        elif got.get(key) != want:
            return False
    return True


def run_lookup(atlas, q: dict) -> dict:
    result = atlas_tools.lookup(atlas, q["material"], q.get("quantity", "all"))
    got_notes = sorted(m["note_id"] for m in result.get("matches", []))
    problems = []
    if q["kind"] == "absent":
        if got_notes:
            problems.append(f"expected no match, got {got_notes}")
        if q.get("reason") and not any(q["reason"] in (g.get("reason") or "") for g in result.get("absent_from_release", [])):
            problems.append("the release's reason for the absence is not quoted")
        return {"pass": not problems, "problems": problems, "notes": got_notes}
    if got_notes != sorted(q["notes"]):
        problems.append(f"notes {got_notes} != expected {sorted(q['notes'])}")
    if q.get("ambiguous") and not result.get("ambiguous"):
        problems.append("ambiguity not flagged")
    values = [v for m in result.get("matches", []) for v in m["values"]]
    for want in q.get("values", []):
        if not any(_value_matches(want, v) for v in values):
            problems.append(f"no value matching {want}")
    for v in values:
        if v.get("value") is not None and not v.get("origin", "").endswith("(reference)"):
            problems.append("a value without its reference origin")
    return {"pass": not problems, "problems": problems, "notes": got_notes}


def run_search(atlas, q: dict) -> dict:
    result = atlas_tools.search(atlas, q["query"], limit=8)
    ranked = []
    for hit in result["hits"]:
        if hit["note_id"] not in ranked:
            ranked.append(hit["note_id"])
    expected = set(q["notes"])
    hit1 = bool(ranked) and ranked[0] in expected
    hit5 = bool(expected & set(ranked[:5]))
    return {"pass": hit5, "hit@1": hit1, "hit@5": hit5, "ranked_notes": ranked[:5]}


def deterministic(atlas, questions: list[dict]) -> dict:
    rows = {}
    for q in questions:
        rows[q["id"]] = run_search(atlas, q) if q["kind"] == "search" else run_lookup(atlas, q)
    searches = [rows[q["id"]] for q in questions if q["kind"] == "search"]
    lookups = [rows[q["id"]] for q in questions if q["kind"] != "search"]
    return {
        "lookups_passed": f"{sum(r['pass'] for r in lookups)}/{len(lookups)}",
        "search_hit@1": f"{sum(r['hit@1'] for r in searches)}/{len(searches)}",
        "search_hit@5": f"{sum(r['hit@5'] for r in searches)}/{len(searches)}",
        "questions": rows,
    }


def end_to_end(questions: list[dict], model: str | None) -> dict:
    """Each non-search question through the assistant, as Quark asks it."""
    import tempfile

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from cnms_fom.db.base import Base
    from cnms_fom.rag_backend.agent import ask
    from cnms_fom.rag_backend.providers import get_provider

    tmp = tempfile.mkdtemp(prefix="atlas-e2e-")
    engine = create_engine(f"sqlite+pysqlite:///{tmp}/e2e.db", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, future=True)()
    provider = get_provider("ollama", model)
    rows = {}
    for q in questions:
        if q["kind"] == "search" or "e2e" not in q:
            continue
        started = time.monotonic()
        try:
            answer = ask(session, q["question"], provider=provider)
            text = answer.answer
            tools_used = answer.tools_used
            error = None
        except Exception as exc:  # noqa: BLE001 - an eval records failures, it does not stop on them
            text, tools_used, error = "", [], str(exc)
        lowered = text.casefold()
        contains = {s: s.casefold() in lowered for s in q["e2e"].get("contains", [])}
        #  any_of: groups of alternatives, at least one of each group present ("one" or "1 source").
        for group in q["e2e"].get("any_of", []):
            contains[" | ".join(group)] = any(alt.casefold() in lowered for alt in group)
        absent = {s: s.casefold() not in lowered for s in q["e2e"].get("absent", [])}
        rows[q["id"]] = {
            "pass": error is None and all(contains.values()) and all(absent.values()),
            "used_atlas": any(t.startswith("atlas_") for t in tools_used),
            "tools_used": tools_used,
            "contains": contains,
            "absent": absent,
            "seconds": round(time.monotonic() - started, 1),
            "answer": text,
            "error": error,
        }
    done = list(rows.values())
    return {
        "model": provider.model,
        "passed": f"{sum(r['pass'] for r in done)}/{len(done)}",
        "used_atlas": f"{sum(r['used_atlas'] for r in done)}/{len(done)}",
        "questions": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=HERE.parent / "index" / "atlas_index.sqlite")
    parser.add_argument("--e2e", action="store_true", help="also run end to end through the assistant")
    parser.add_argument("--model", help="Ollama model for --e2e (default: OLLAMA_CHAT_MODEL)")
    parser.add_argument("--only", help="comma-separated question ids")
    args = parser.parse_args(argv)

    spec = yaml.safe_load((HERE / "questions.yaml").read_text())
    questions = spec["questions"]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    atlas = open_atlas(args.index, expected_version=spec["release"])
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    report = {"release": atlas.version, "db_sha256": atlas.sha256, "deterministic": deterministic(atlas, questions)}
    atlas.close()
    if args.e2e:
        report["end_to_end"] = end_to_end(questions, args.model)
    (out_dir / ("e2e.json" if args.e2e else "deterministic.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    summary = {k: v for k, v in report["deterministic"].items() if k != "questions"}
    failed = {qid: r for qid, r in report["deterministic"]["questions"].items() if not r["pass"]}
    print(json.dumps({"deterministic": summary, "failed": failed}, indent=2))
    if args.e2e:
        e2e = report["end_to_end"]
        print(json.dumps({"end_to_end": {k: v for k, v in e2e.items() if k != "questions"},
                          "failed": {qid: {k: r[k] for k in ("contains", "absent", "tools_used", "error")} for qid, r in e2e["questions"].items() if not r["pass"]}}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
