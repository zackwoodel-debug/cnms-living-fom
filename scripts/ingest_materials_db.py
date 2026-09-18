#!/usr/bin/env python
"""Ingest a CNMS ``materials-db`` SQLite export into the Living FOM database.

Default mode is ``survey``: it reads the source, reports what is there and what
blocks promotion, and writes nothing. Pointing an importer at a research
database and letting it write on the first run is how provenance gets lost.

    # What is in it, and what would stop it being used?
    python scripts/ingest_materials_db.py survey ~/Desktop/materials-db/data/materials_oxide_test.db

    # Copy rows verbatim into external_records (still inert).
    python scripts/ingest_materials_db.py stage <db>

    # Create analysis rows. --specimen-form is required and is recorded as an
    # explicit human assertion on every material created.
    python scripts/ingest_materials_db.py promote <db> \
        --specimen-form bulk_single_crystal --operator "Z. Woodel"

    # Optional: eps_inf = n^2 from the transparent window (Eq. 11).
    python scripts/ingest_materials_db.py derive-eps-inf --window 1000 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from cnms_fom.db.base import session_scope  # noqa: E402
from cnms_fom.db.enums import SpecimenForm  # noqa: E402
from cnms_fom.ingest import materials_db  # noqa: E402


def _print(report) -> None:
    payload = report.as_dict()
    print(json.dumps(payload, indent=2, default=str))
    if payload.get("notes"):
        print("\nNotes:", file=sys.stderr)
        for note in payload["notes"]:
            print(f"  - {note}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    survey = sub.add_parser("survey", help="Read-only report. Writes nothing.")
    survey.add_argument("database", type=Path)

    stage = sub.add_parser("stage", help="Copy rows into external_records verbatim.")
    stage.add_argument("database", type=Path)
    stage.add_argument("--limit", type=int, help="Stage at most this many rows per table.")

    promote = sub.add_parser("promote", help="Create analysis rows from staged records.")
    promote.add_argument("database", type=Path)
    promote.add_argument(
        "--specimen-form",
        required=True,
        choices=[f.value for f in SpecimenForm],
        help="Asserted by you; the external source does not record it (Sec. 2.1).",
    )
    promote.add_argument(
        "--phase-map",
        type=Path,
        help='JSON {"external name or id": "polymorph"} for records whose label has no phase.',
    )
    promote.add_argument("--operator", default="", help="Who is asserting the specimen form.")
    promote.add_argument("--skip-optical", action="store_true")

    derive = sub.add_parser("derive-eps-inf", help="eps_inf = n^2 from a transparent window.")
    derive.add_argument("--window", type=float, nargs=2, default=[1000.0, 2000.0], metavar=("LO_NM", "HI_NM"))

    args = parser.parse_args(argv)

    if args.command == "survey":
        _print(materials_db.survey(args.database))
        return 0

    with session_scope() as session:
        if args.command == "stage":
            _print(materials_db.stage(session, args.database, limit=args.limit))
        elif args.command == "promote":
            _print(
                materials_db.promote(
                    session,
                    args.database,
                    specimen_form=SpecimenForm(args.specimen_form),
                    phase_map=materials_db.load_phase_map(args.phase_map),
                    include_optical=not args.skip_optical,
                    operator=args.operator,
                )
            )
        elif args.command == "derive-eps-inf":
            _print(materials_db.derive_eps_inf(session, window_nm=tuple(args.window)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
