"""The embedding column type, tolerant of both ways the column can be stored.

``PGVECTOR_ENABLED`` selects this type at import, before anything has looked at the
database, and nothing makes the flag agree with the schema it is pointed at. Migration
0001 created ``document_chunks.embedding`` as ``json`` regardless of it, so both
directions of disagreement occur in the wild, and before this module both were fatal:

* flag on, ``json`` column — pgvector's result parser got a list where it expected its
  own text format and raised ``'list' object has no attribute 'split'`` on *every*
  chunk read, including the lexical leg's. Retrieval was down, not degraded.
* flag off, ``vector`` column — a JSON-mapped column read back the *string*
  ``'[0.1,0.2,...]'``, ``_rank_in_python`` raised on ``np.asarray(..., dtype=float)``,
  the dense leg was dropped as an optional retriever, and lexical-only results were
  served indefinitely with nothing said.

Reporting the mismatch legibly was the first fix. This is the second and better one:
make the mismatch stop mattering. Postgres accepts the same text literal
``'[0.1,0.2,0.3]'`` for a ``json`` column *and* for a ``vector`` column, and returns a
``list`` from the first and a ``str`` from the second. So one bind format serves both,
and a result processor that accepts either shape normalises them back to
``list[float]``. Verified against Postgres in both directions rather than reasoned
about — see ``test_embedding_column_type.py``.

What the flag still decides is real but narrow: which type ``create_all`` emits for a
fresh database, and whether the ``<=>`` operator is worth attempting. The second is
already gated on the column's *actual* type by ``vectorstore._pgvector_available``, so
a mismatch now costs the ANN index and nothing else. That is a performance note, which
``/health/ready`` reports as such, rather than an outage.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator


def _as_floats(value: Any) -> Any:
    """Normalise either storage shape to ``list[float]``.

    A ``list`` came from a json column, a ``str`` from a vector one. Anything else is
    handed back untouched: this is a compatibility shim, and inventing a value for
    something it does not recognise would be worse than letting the caller see it.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(component) for component in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except ValueError:
            return value
        if isinstance(parsed, list):
            return [float(component) for component in parsed]
        return parsed
    return value


class TolerantJSON(TypeDecorator):
    """JSON that can also *read* a pgvector column.

    Used when ``PGVECTOR_ENABLED`` is false. On Postgres a ``vector`` column arrives as
    the string ``'[0.1,0.2]'``, which this turns back into a list so the portable
    ranker and everything downstream see numbers.

    Reads only, and the asymmetry is not for want of trying. Writing into a vector
    column through this type is not possible: Postgres renders the bind as
    ``%(embedding)s::JSON`` — ``render_bind_cast`` on ``dialects/postgresql/json.py``
    — and refuses ``json`` into ``vector``. The cast cannot be suppressed from here,
    because ``dialect.type_descriptor`` re-creates any impl this returns as the
    canonical Postgres JSON type and drops the override. Writing correctly would mean
    the type knowing the column's real shape, which is exactly the thing that is not
    knowable at import.

    So this direction degrades rather than works: existing rows stay readable and
    dense search keeps running, while an ingest fails with Postgres's own datatype
    error. ``vectorstore.embedding_storage_mismatch`` reports it before then.
    """

    impl = JSON
    cache_ok = True

    def __init__(self, *args, **kwargs):
        #  none_as_null so an absent embedding is SQL NULL rather than the JSON
        #  scalar 'null'. Without it SQLAlchemy's JSON stores Python None as a JSON
        #  null, which is a *value*: `embedding IS NOT NULL` is then true for a chunk
        #  that was never embedded, and `ingest` passes `embedding=vector` with vector
        #  None whenever it is called with embed=False. The consequences were quiet
        #  and wrong — `corpus_stats` over-reported `embedded_chunks`, and
        #  `benchmark._resolve_retrievers` concluded the dense leg was available on a
        #  corpus with no vectors in it, so its "lexical-only, not comparable" note
        #  never fired.
        #
        #  Absent staying absent is also the protocol's rule. A JSON null here is the
        #  storage-level version of defaulting a missing quantity to a placeholder.
        kwargs.setdefault("none_as_null", True)
        super().__init__(*args, **kwargs)

    def process_result_value(self, value, dialect):  # noqa: ARG002 - SQLAlchemy API
        return _as_floats(value)


def embedding_vector_type(dim: int):
    """A pgvector ``Vector`` that also reads a json column, or None if unavailable."""
    try:
        from pgvector.sqlalchemy import Vector
    except ImportError:  # pragma: no cover - depends on the optional extra
        return None

    class TolerantVector(Vector):
        """``Vector`` whose result processor accepts a json column's list.

        Subclasses ``Vector`` rather than wrapping it in a ``TypeDecorator`` so that
        ``cosine_distance`` and the rest of pgvector's comparator stay available —
        ``TypeDecorator`` does not forward a custom ``comparator_factory``, and losing
        ``<=>`` would defeat the point of enabling pgvector at all.

        ``Vector``'s own bind processor already emits the ``'[...]'`` text form, which
        a json column accepts and parses as an array, so writes are left alone.
        """

        cache_ok = True

        def result_processor(self, dialect, coltype):
            inherited = super().result_processor(dialect, coltype)

            def process(value):
                if isinstance(value, (list, tuple)):
                    #  A json column. pgvector's parser would call .split on it.
                    return [float(component) for component in value]
                if inherited is not None:
                    return inherited(value)
                return _as_floats(value)

            return process

    return TolerantVector(dim)
