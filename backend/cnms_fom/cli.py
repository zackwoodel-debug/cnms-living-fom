"""Command-line entry point: ``cnms-fom <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _init_db(args: argparse.Namespace) -> int:
    from sqlalchemy import text

    from cnms_fom.config import get_settings
    from cnms_fom.db import models  # noqa: F401 - registers the mappers
    from cnms_fom.db.base import Base, get_engine

    engine = get_engine()
    if get_settings().pgvector_enabled:
        try:
            with engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            print("pgvector extension present")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not enable pgvector ({exc}); falling back to JSON embeddings.")

    Base.metadata.create_all(engine)
    print(f"Created {len(Base.metadata.tables)} tables:")
    for name in sorted(Base.metadata.tables):
        print(f"  {name}")
    return 0


def _seed(args: argparse.Namespace) -> int:
    from cnms_fom.cnms_integration.instruments import list_instruments
    from cnms_fom.db.base import session_scope
    from cnms_fom.db.models import FomDefinition, Instrument
    from cnms_fom.fom_engine.definitions import all_draft_foms, to_definition_kwargs

    with session_scope() as db:
        for spec in all_draft_foms().values():
            exists = (
                db.query(FomDefinition)
                .filter(FomDefinition.name == spec.name, FomDefinition.version == spec.version)
                .one_or_none()
            )
            if exists is None:
                db.add(FomDefinition(**to_definition_kwargs(spec)))
                print(f"  + FOM definition {spec.name} v{spec.version} (DRAFT, unapproved)")

        for record in list_instruments():
            exists = (
                db.query(Instrument)
                .filter(Instrument.instrument_id == record.instrument_id)
                .one_or_none()
            )
            if exists is None:
                db.add(
                    Instrument(
                        instrument_id=record.instrument_id,
                        name=record.name,
                        technique=record.technique,
                        location=record.location,
                        capabilities=record.capabilities,
                        available=record.available,
                    )
                )
                print(f"  + instrument {record.instrument_id} (PLACEHOLDER envelope)")

    print(
        "\nSeeded. Both the FOM weights and the instrument envelopes are placeholders — "
        "see the TODOs in fom_engine/definitions.py and cnms_integration/instruments.py."
    )
    return 0


def _ingest(args: argparse.Namespace) -> int:
    from cnms_fom.db.base import session_scope
    from cnms_fom.db.enums import SynthesisTechnique
    from cnms_fom.rag_backend.ingest import ingest_directory, ingest_pdf

    path = Path(args.path)
    technique = SynthesisTechnique(args.technique) if args.technique else None
    with session_scope() as db:
        results = (
            ingest_directory(db, path, technique=technique)
            if path.is_dir()
            else [ingest_pdf(db, path, technique=technique)]
        )
    print(json.dumps(results, indent=2))
    return 0


def _dictionary(args: argparse.Namespace) -> int:
    from cnms_fom.descriptors.registry import descriptor_dictionary

    print(json.dumps(descriptor_dictionary(), indent=2))
    return 0


def _hypotheses(args: argparse.Namespace) -> int:
    from cnms_fom.fom_engine.hypotheses import as_records, registry_fingerprint

    print(json.dumps({"fingerprint": registry_fingerprint(), "hypotheses": as_records()}, indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from cnms_fom.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "cnms_fom.main:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cnms-fom", description="CNMS Living FOM platform.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create tables (and the pgvector extension).").set_defaults(
        func=_init_db
    )
    subparsers.add_parser(
        "seed", help="Insert draft FOM definitions and placeholder instruments."
    ).set_defaults(func=_seed)
    subparsers.add_parser(
        "dictionary", help="Print the Sec. 13.1 descriptor dictionary as JSON."
    ).set_defaults(func=_dictionary)
    subparsers.add_parser(
        "hypotheses", help="Print the pre-registered sign table and its fingerprint."
    ).set_defaults(func=_hypotheses)

    ingest = subparsers.add_parser("ingest", help="Ingest a PDF or a directory of PDFs.")
    ingest.add_argument("path")
    ingest.add_argument(
        "--technique",
        choices=["mbe", "pld", "ald", "sputtering", "cvd", "solution", "cnms_user_doc", "other"],
    )
    ingest.set_defaults(func=_ingest)

    serve = subparsers.add_parser("serve", help="Run the API with uvicorn.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
