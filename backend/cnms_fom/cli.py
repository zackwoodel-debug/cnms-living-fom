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


def _import_fits(args: argparse.Namespace) -> int:
    """Import ModalFit exports as measurement records.

    Prints the warnings last, because they are the part worth reading: a fit
    claiming a technique with no matching slab-model block, or whose parameters
    finished clamped on their bounds, is stored but is not something to quote.
    """
    from cnms_fom.db.base import session_scope
    from cnms_fom.modalfit.records import import_directory, import_fit

    path = Path(args.path)
    with session_scope() as db:
        results = (
            import_directory(
                db,
                path,
                techniques=args.techniques,
                algorithm=args.algorithm,
                length_units=args.length_units,
            )
            if path.is_dir()
            else [
                import_fit(
                    db,
                    path,
                    techniques=args.techniques,
                    algorithm=args.algorithm,
                    length_units=args.length_units,
                    sample_id=args.sample_id,
                )
            ]
        )

    print(json.dumps(results, indent=2, default=str))
    warnings = [w for result in results for w in result.get("warnings", [])]
    if warnings:
        print("\nWarnings — read these before quoting any number from these fits:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


def _compare_fits(args: argparse.Namespace) -> int:
    """Cross-technique agreement for one sample."""
    from cnms_fom.db.base import session_scope
    from cnms_fom.modalfit.compare import compare_parameter, cross_technique_report

    with session_scope() as db:
        result = (
            cross_technique_report(db, args.sample_id, layer_label=args.layer)
            if args.parameter == "all"
            else compare_parameter(
                db, args.sample_id, parameter=args.parameter, layer_label=args.layer
            )
        )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _ask(args: argparse.Namespace) -> int:
    """Put one question to the research assistant and print its evidence trail."""
    from cnms_fom.config import get_settings
    from cnms_fom.db.base import session_scope
    from cnms_fom.db.enums import SynthesisTechnique
    from cnms_fom.rag_backend import memory
    from cnms_fom.rag_backend.agent import ask
    from cnms_fom.rag_backend.providers import get_provider

    settings = get_settings()
    provider = get_provider(args.provider, args.model)
    techniques = [SynthesisTechnique(t) for t in (args.technique or [])] or None

    try:
        with session_scope() as db:
            session = memory.get_or_create_session(
                db,
                args.session,
                sample_id=args.sample_id,
                techniques=techniques,
                provider=provider.name,
                chat_model=provider.model,
            )
            history = memory.load_history(db, session, turns=settings.assistant_history_turns)
            answer = ask(
                db,
                args.question,
                history=history,
                sample_id=session.sample_id,
                techniques=techniques,
                provider=provider,
                max_steps=args.max_steps or settings.assistant_max_steps,
            )
            memory.record_turn(db, session, args.question, answer)
            session_key = session.session_key
    except ImportError as exc:
        #  A traceback here tells the user nothing they can act on. The two ways
        #  this fails are a missing extra and an unreachable model server, and
        #  both have a one-line fix.
        print(f"The assistant needs an optional dependency: {exc}", file=sys.stderr)
        print("  pip install -e '.[rag]'          # local Ollama provider", file=sys.stderr)
        print("  pip install -e '.[anthropic]'    # Anthropic provider", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - a CLI reports, it does not traceback
        print(f"The assistant could not answer: {exc}", file=sys.stderr)
        if provider.name == "ollama":
            print(
                f"  Is Ollama running at {settings.ollama_base_url}, and are "
                f"{settings.ollama_chat_model} and {settings.ollama_embed_model} pulled?",
                file=sys.stderr,
            )
        return 1

    for step in answer.steps:
        print(f"[step {step.step}] {step.tool}({json.dumps(step.arguments)}) -> {step.duration_ms} ms")
    print(f"\n{answer.answer}\n")
    if answer.citations:
        print("Evidence:")
        for citation in answer.citations:
            print(f"  - {citation['kind']}: {citation['citation']}")
    print(f"\nconversation: {session_key}  ({answer.provider}/{answer.model}, {answer.latency_ms} ms)")
    #  Non-zero on a data gap, so a script can tell "answered" from "could not".
    return 0 if not answer.insufficient_context else 2


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

    import_fits = subparsers.add_parser(
        "import-fits",
        help="Import a ModalFit exported model JSON, or a directory of them.",
    )
    import_fits.add_argument("path")
    import_fits.add_argument(
        "--technique",
        dest="techniques",
        action="append",
        choices=["SE", "SPR", "QCM", "XRR", "NR"],
        help="Technique actually co-refined; repeat for a co-refinement. Required unless the "
        "export carries its own fit metadata — it is never inferred from the slab-model blocks.",
    )
    import_fits.add_argument(
        "--algorithm",
        choices=["L-BFGS-B", "Nelder-Mead", "Differential Evolution", "Basin-Hopping", "DREAM (emcee)"],
    )
    import_fits.add_argument(
        "--length-units",
        default="angstrom",
        choices=["angstrom", "nm"],
        help="Unit the export's thicknesses are in. ModalFit's physics backends use angstroms; "
        "its bundled substrate library is written in nanometres. Not guessed.",
    )
    import_fits.add_argument("--sample-id", help="Override the sample id in the export.")
    import_fits.set_defaults(func=_import_fits)

    compare = subparsers.add_parser(
        "compare-fits", help="Cross-technique agreement for one sample's fitted parameters."
    )
    compare.add_argument("sample_id")
    compare.add_argument(
        "--parameter",
        default="thickness",
        help="thickness, roughness, density, sld_real, sld_imag, n, k, or 'all'.",
    )
    compare.add_argument("--layer", help="Layer label, when the stack has more than one film.")
    compare.set_defaults(func=_compare_fits)

    ask_parser = subparsers.add_parser(
        "ask", help="Ask the research assistant one question. Exit code 2 means a data gap."
    )
    ask_parser.add_argument("question")
    ask_parser.add_argument("--session", help="Continue an existing conversation by key.")
    ask_parser.add_argument("--sample-id", help="Scope the conversation to one ModalFit sample.")
    ask_parser.add_argument(
        "--technique",
        action="append",
        choices=["mbe", "pld", "ald", "sputtering", "cvd", "solution", "cnms_user_doc", "other"],
        help="Restrict corpus retrieval to these partitions; repeat to allow several.",
    )
    ask_parser.add_argument("--provider", choices=["ollama", "anthropic"])
    ask_parser.add_argument("--model")
    ask_parser.add_argument("--max-steps", type=int)
    ask_parser.set_defaults(func=_ask)

    serve = subparsers.add_parser("serve", help="Run the API with uvicorn.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
