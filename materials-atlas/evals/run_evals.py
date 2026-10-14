#!/usr/bin/env python
"""Run the atlas evals, and gate on them (materials-atlas/CLAUDE.md section 10).

    python materials-atlas/evals/run_evals.py                      # deterministic, smoke suite
    python materials-atlas/evals/run_evals.py --suite full         # + the questions generated from the release
    python materials-atlas/evals/run_evals.py --e2e --model qwen3:14b   # also through the assistant
    python materials-atlas/evals/run_evals.py --gate               # exit 1 if below evals/baselines.json
    python materials-atlas/evals/run_evals.py --index backend/tests/fixtures/atlas/index/atlas_index.sqlite \\
        --skip-missing --gate                                      # what CI runs, on the fixture index

Deterministic mode calls the atlas tools directly: did the right materials and values come
back, was ambiguity flagged, did a search rank the right note first (hit@1) or in the top
five (hit@5). End-to-end mode asks the CNMS assistant each question (the loop /rag/chat and
Quark use, every tool offered) and checks what the reader sees: the expected ids and values
present, the traps absent. It also records whether the answer was composed (a framing over
verbatim values), whether the model's framing had to be withheld, and any unsupported
claims, which is how the model's own reliability is tracked separately from the system's.

Questions: ``questions.yaml`` (hand-written: the core cases and the adversarial ones) and
``questions_generated.yaml`` (from the release by plain SQL, ``make_questions.py``). Never
tune the data to pass an eval; tune the index, the tools or the prompt, and say which.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.dont_write_bytecode = True

import yaml  # noqa: E402

from cnms_fom.atlas import open_atlas  # noqa: E402
from cnms_fom.atlas import tools as atlas_tools  # noqa: E402

BASELINES = HERE / "baselines.json"


def load_questions(suite: str) -> tuple[str, list[dict]]:
    spec = yaml.safe_load((HERE / "questions.yaml").read_text())
    questions = [dict(q, suite="smoke") for q in spec["questions"]]
    if suite == "full":
        generated = yaml.safe_load((HERE / "questions_generated.yaml").read_text())
        if generated["release"] != spec["release"]:
            raise SystemExit("questions_generated.yaml is for another release: re-run make_questions.py")
        questions += [dict(q, suite="full") for q in generated["questions"]]
    for q in questions:
        q.setdefault("category", "search" if q["kind"] == "search" else "core")
    return spec["release"], questions


# ---------------------------------------------------------------------------
# Deterministic
# ---------------------------------------------------------------------------


def _value_matches(expected: dict, got: dict) -> bool:
    return all(got.get(key) == want for key, want in expected.items())


def _check_matches(q: dict, result: dict) -> list[str]:
    problems = []
    matches = result.get("matches", [])
    if "notes" in q and sorted(m["note_id"] for m in matches) != sorted(q["notes"]):
        problems.append(f"notes {sorted(m['note_id'] for m in matches)} != expected {sorted(q['notes'])}")
    if "material_ids" in q and sorted(m["material_id"] for m in matches) != sorted(q["material_ids"]):
        problems.append(f"material ids {sorted(m['material_id'] for m in matches)} != expected {sorted(q['material_ids'])}")
    if q.get("ambiguous") and not result.get("ambiguous"):
        problems.append("ambiguity not flagged")
    values = [v for m in matches for v in m["values"]]
    for want in q.get("values", []):
        if not any(_value_matches(want, v) for v in values):
            problems.append(f"no value matching {want}")
    if any(v.get("value") is not None and not v.get("origin", "").endswith("(reference)") for v in values):
        problems.append("a value without its reference origin")
    return problems


def run_deterministic(atlas, q: dict) -> dict:
    if q["kind"] == "search":
        result = atlas_tools.search(atlas, q["query"], limit=8)
        ranked = list(dict.fromkeys(h["note_id"] for h in result["hits"]))
        expected = set(q["notes"])
        hit1 = bool(ranked) and ranked[0] in expected
        hit5 = bool(expected & set(ranked[:5]))
        return {"pass": hit5, "hit@1": hit1, "hit@5": hit5, "ranked_notes": ranked[:5]}
    if q["kind"] == "compare":
        result = atlas_tools.compare(atlas, q["materials"], q.get("quantity", "all"))
        found = sorted(m["note_id"] for r in result["materials"] for m in r.get("matches", []))
        problems = [] if found == sorted(q["notes"]) else [f"notes {found} != expected {sorted(q['notes'])}"]
        return {"pass": not problems, "problems": problems}
    result = atlas_tools.lookup(atlas, q["material"], q.get("quantity", "all"))
    if q["kind"] == "absent":
        problems = []
        if result.get("matches"):
            problems.append(f"expected no match, got {[m['note_id'] for m in result['matches']]}")
        if q.get("reason") and not any(q["reason"] in (g.get("reason") or "") for g in result.get("absent_from_release", [])):
            problems.append("the release's reason for the absence is not quoted")
        return {"pass": not problems, "problems": problems}
    problems = _check_matches(q, result)
    return {"pass": not problems, "problems": problems}


def available(atlas, q: dict) -> bool:
    """Whether the index holds every material a question expects (the fixture holds twelve)."""
    notes = set(q.get("notes", []))
    if notes and len(atlas.query(f"SELECT note_id FROM notes WHERE note_id IN ({','.join('?' * len(notes))})", tuple(notes))) < len(notes):
        return False
    ids = set(q.get("material_ids", []))
    if ids and len(atlas.query(f"SELECT 1 FROM notes WHERE material_id IN ({','.join('?' * len(ids))})", tuple(ids))) < len(ids):
        return False
    if q["kind"] == "absent":
        #  An absent material is only meaningfully absent from the full atlas.
        return atlas.query("SELECT COUNT(*) FROM notes WHERE type = 'material'")[0][0] > 1000
    return True


def summarise_deterministic(rows: dict, questions: list[dict]) -> dict:
    lookups = [rows[q["id"]] for q in questions if q["id"] in rows and q["kind"] != "search"]
    searches = [rows[q["id"]] for q in questions if q["id"] in rows and q["kind"] == "search"]
    by_category: dict[str, list[bool]] = defaultdict(list)
    for q in questions:
        if q["id"] in rows:
            by_category[q["category"]].append(rows[q["id"]]["pass"])
    return {
        "lookups_passed": [sum(r["pass"] for r in lookups), len(lookups)],
        "search_hit@1": [sum(r["hit@1"] for r in searches), len(searches)],
        "search_hit@5": [sum(r["hit@5"] for r in searches), len(searches)],
        "by_category": {c: [sum(v), len(v)] for c, v in sorted(by_category.items())},
    }


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def end_to_end(questions: list[dict], model: str | None) -> tuple[dict, dict]:
    import tempfile

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import cnms_fom.db.models  # noqa: F401 - registers the tables create_all makes
    from cnms_fom.db.base import Base
    from cnms_fom.rag_backend.agent import ask
    from cnms_fom.rag_backend.providers import get_provider

    tmp = tempfile.mkdtemp(prefix="atlas-e2e-")
    engine = create_engine(f"sqlite+pysqlite:///{tmp}/e2e.db", future=True)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, future=True)()
    provider = get_provider("ollama", model)
    rows: dict[str, dict] = {}
    for q in questions:
        if q["kind"] == "search" or "e2e" not in q:
            continue
        started = time.monotonic()
        try:
            answer = ask(session, q["question"], provider=provider)
            text, error = answer.answer, None
        except Exception as exc:  # noqa: BLE001 - an eval records failures, it does not stop on them
            answer, text, error = None, "", str(exc)
        lowered = text.casefold()
        contains = {s: s.casefold() in lowered for s in q["e2e"].get("contains", [])}
        for group in q["e2e"].get("any_of", []):
            contains[" | ".join(group)] = any(alt.casefold() in lowered for alt in group)
        absent = {s: s.casefold() not in lowered for s in q["e2e"].get("absent", [])}
        rows[q["id"]] = {
            "category": q["category"],
            "pass": error is None and all(contains.values()) and all(absent.values()),
            "used_atlas": bool(answer) and any(t.startswith("atlas_") for t in answer.tools_used),
            "composed": bool(answer and answer.composed),
            "framing_withheld": bool(answer and answer.composed and "[withheld framing" in (answer.suppressed_answer or "")),
            "unsupported_numbers": answer.unsupported_numbers if answer else [],
            "unsupported_claims": answer.unsupported_claims if answer else [],
            "tools_used": answer.tools_used if answer else [],
            "contains": contains,
            "absent": absent,
            "seconds": round(time.monotonic() - started, 1),
            "model_text": answer.model_text if answer else "",
            "answer": text,
            "suppressed": answer.suppressed_answer if answer else None,
            "error": error,
        }
    done = list(rows.values())
    composed = [r for r in done if r["composed"]]
    summary = {
        "model": provider.model,
        "passed": [sum(r["pass"] for r in done), len(done)],
        "used_atlas": [sum(r["used_atlas"] for r in done), len(done)],
        "composed": [len(composed), len(done)],
        "framing_withheld": [sum(r["framing_withheld"] for r in composed), len(composed)],
        "flagged_answers": sum(bool(r["unsupported_numbers"] or r["unsupported_claims"]) for r in done),
        "median_seconds": sorted(r["seconds"] for r in done)[len(done) // 2] if done else None,
        "by_category": {
            c: [sum(r["pass"] for r in done if r["category"] == c), sum(r["category"] == c for r in done)]
            for c in sorted({r["category"] for r in done})
        },
    }
    return summary, rows


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _rate(pair) -> float:
    return pair[0] / pair[1] if pair[1] else 1.0


def gate(report: dict, suite: str, tolerance: float) -> list[str]:
    """Failures against baselines.json. Deterministic must be perfect; e2e within tolerance."""
    baselines = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}
    failures = []
    det = report["deterministic"]["summary"]
    for key in ("lookups_passed", "search_hit@5"):
        if det[key][0] != det[key][1]:
            failures.append(f"deterministic {key}: {det[key][0]}/{det[key][1]}")
    floor = baselines.get("deterministic", {}).get("search_hit@1_min", 1.0)
    if _rate(det["search_hit@1"]) < floor:
        failures.append(f"deterministic search_hit@1 {det['search_hit@1']} below {floor}")
    e2e = report.get("end_to_end", {}).get("summary")
    if e2e:
        base = baselines.get("end_to_end", {}).get(e2e["model"], {}).get(suite)
        if base is None:
            failures.append(f"no end-to-end baseline for {e2e['model']} on the {suite} suite (--update-baseline)")
        else:
            if _rate(e2e["passed"]) < base["pass_rate"] - tolerance:
                failures.append(f"end-to-end pass rate {e2e['passed']} below baseline {base['pass_rate']:.2f} - {tolerance}")
            if _rate(e2e["framing_withheld"]) > base["framing_withheld_rate"] + tolerance:
                failures.append(f"framing withheld {e2e['framing_withheld']} above baseline {base['framing_withheld_rate']:.2f}")
    return failures


def update_baseline(report: dict, suite: str) -> None:
    baselines = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}
    e2e = report["end_to_end"]["summary"]
    baselines.setdefault("end_to_end", {}).setdefault(e2e["model"], {})[suite] = {
        "pass_rate": round(_rate(e2e["passed"]), 3),
        "framing_withheld_rate": round(_rate(e2e["framing_withheld"]), 3),
        "questions": e2e["passed"][1],
        "release": report["release"],
    }
    BASELINES.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=HERE.parent / "index" / "atlas_index.sqlite")
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--e2e", action="store_true", help="also run end to end through the assistant")
    parser.add_argument("--model", help="Ollama model for --e2e (default: OLLAMA_CHAT_MODEL)")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--skip-missing", action="store_true", help="skip questions whose materials the index lacks")
    parser.add_argument("--gate", action="store_true", help="exit 1 below baselines.json")
    parser.add_argument("--tolerance", type=float, default=0.05, help="end-to-end slack for --gate")
    parser.add_argument("--update-baseline", action="store_true", help="record this end-to-end run as the baseline")
    args = parser.parse_args(argv)

    release, questions = load_questions(args.suite)
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    atlas = open_atlas(args.index, expected_version=release)
    skipped = []
    if args.skip_missing:
        kept = [q for q in questions if available(atlas, q)]
        skipped = sorted({q["id"] for q in questions} - {q["id"] for q in kept})
        questions = kept
    rows = {q["id"]: run_deterministic(atlas, q) for q in questions}
    report = {
        "release": atlas.version,
        "db_sha256": atlas.sha256,
        "index": str(args.index),
        "suite": args.suite,
        "skipped": skipped,
        "deterministic": {"summary": summarise_deterministic(rows, questions), "questions": rows},
    }
    atlas.close()
    if args.e2e:
        summary, e2e_rows = end_to_end(questions, args.model)
        report["end_to_end"] = {"summary": summary, "questions": e2e_rows}
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    name = f"{'e2e' if args.e2e else 'deterministic'}-{args.suite}" + (
        f"-{report['end_to_end']['summary']['model'].replace(':', '_')}" if args.e2e else ""
    )
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    printed = {"deterministic": report["deterministic"]["summary"], "skipped": len(skipped)}
    printed["failed"] = {qid: r for qid, r in rows.items() if not r["pass"]}
    if args.e2e:
        printed["end_to_end"] = report["end_to_end"]["summary"]
        printed["end_to_end_failed"] = {
            qid: {k: r[k] for k in ("contains", "absent", "unsupported_claims", "framing_withheld", "error")}
            for qid, r in report["end_to_end"]["questions"].items() if not r["pass"]
        }
    print(json.dumps(printed, indent=2, ensure_ascii=False))
    if args.update_baseline and args.e2e:
        update_baseline(report, args.suite)
        print(f"baseline updated for {report['end_to_end']['summary']['model']} ({args.suite})")
    if args.gate:
        failures = gate(report, args.suite, args.tolerance)
        if failures:
            print("GATE FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
            return 1
        print("gate passed")
        return 0
    return 0 if not printed["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
