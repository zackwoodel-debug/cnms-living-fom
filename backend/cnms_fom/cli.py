"""Command-line entry point: ``cnms-fom <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _alembic_config():
    """Alembic config rooted at the repository, wired to the live DATABASE_URL."""
    from alembic.config import Config

    from cnms_fom.config import get_settings

    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    config.set_main_option("sqlalchemy.url", get_settings().database_url)
    return config


def _enable_pgvector() -> None:
    from sqlalchemy import text

    from cnms_fom.config import get_settings
    from cnms_fom.db.base import get_engine

    if not get_settings().pgvector_enabled:
        return
    try:
        with get_engine().begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        print("pgvector extension present")
    except Exception as exc:  # noqa: BLE001 - SQLite and unprivileged roles both land here
        print(f"Could not enable pgvector ({exc}); embeddings fall back to a JSON column.")


def _init_db(args: argparse.Namespace) -> int:
    """Bring the database to the current schema via Alembic.

    Alembic rather than ``metadata.create_all``: the schema now carries CHECK
    constraints, a backfilled context digest, and an enum-storage change, none
    of which ``create_all`` can apply to a database that already has rows. Going
    through migrations means the same command works on an empty database and on
    a populated one.
    """
    from alembic import command

    _enable_pgvector()
    config = _alembic_config()
    if args.stamp_baseline:
        #  For a database created by the old create_all path, before Alembic
        #  existed: tell Alembic it is already at the baseline, then upgrade.
        command.stamp(config, "0001")
        print("stamped at baseline revision 0001")
    command.upgrade(config, "head")

    from cnms_fom.db import models  # noqa: F401 - registers the mappers
    from cnms_fom.db.base import Base

    print(f"Database at head. {len(Base.metadata.tables)} tables:")
    for name in sorted(Base.metadata.tables):
        print(f"  {name}")
    return 0


def _migrate(args: argparse.Namespace) -> int:
    """Thin wrapper over the Alembic commands people actually need."""
    from alembic import command

    config = _alembic_config()
    if args.action == "up":
        command.upgrade(config, args.revision or "head")
    elif args.action == "down":
        command.downgrade(config, args.revision or "-1")
    elif args.action == "current":
        command.current(config, verbose=True)
    elif args.action == "history":
        command.history(config, verbose=False)
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

    init_db = subparsers.add_parser(
        "init-db", help="Bring the database to the current schema (runs migrations)."
    )
    init_db.add_argument(
        "--stamp-baseline",
        action="store_true",
        help="For a database created before Alembic existed: stamp it at revision 0001 first.",
    )
    init_db.set_defaults(func=_init_db)

    migrate = subparsers.add_parser("migrate", help="Run Alembic migrations.")
    migrate.add_argument("action", choices=["up", "down", "current", "history"])
    migrate.add_argument("revision", nargs="?", help="Target revision (default: head / -1).")
    migrate.set_defaults(func=_migrate)
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
