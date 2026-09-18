"""Ingestion of external materials databases.

    labels.py       recover phase / axis / source from packed label strings
    materials_db.py stage and promote rows from a CNMS `materials-db` SQLite file

Ingestion is two-phase throughout: everything lands in ``external_records`` with
its raw payload, and only rows that satisfy the protocol's identity and context
requirements are promoted into the analysis tables.  See
``db.models.ExternalRecord`` for why.
"""

from .labels import ParsedLabel, axis_to_tensor_component, parse_dataset_label  # noqa: F401
