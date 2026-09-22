"""A content-addressed cache for per-passage model calls.

Grading and extraction are the two stages that call a model once per passage, and
together they are essentially the entire cost of a brief: measured on this scaffold,
19 sequential calls of which 18 are per-passage, taking about 21 minutes against
1.2 seconds for the retrieval itself.

Both are cacheable, and for a reason worth stating precisely:

``extraction`` is a pure function of the passage.
    The extraction prompt contains the passage and nothing else — not the question.
    So its output depends only on ``(passage text, model, prompt version)``, and a
    passage needs extracting *once ever*. This is what makes a real-provider
    benchmark run affordable: twelve cases over one corpus share most of their
    passages, and a policy sweep re-runs the same cases.

``grading`` is a pure function of the question and the passage.
    One more input, same argument.

**The key is a hash of the content, not an id.** A row identifies what was asked,
not where it was stored, so a re-ingest that renumbers chunks does not serve a stale
answer and an edited passage misses the cache automatically. Correctness is a
property of the key rather than of an invalidation rule somebody has to remember.

This is infrastructure, not science. It stores no claim, no measurement, and nothing
a person would cite; the rows are a transcript of model calls and can be deleted at
any time without losing anything. That is why it is one generic table rather than a
typed one per stage — the rest of this schema is typed because the *protocol*
depends on it, and a cache is the one place here where that reasoning does not apply.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from cnms_fom.config import get_settings

logger = logging.getLogger(__name__)

#  What kind of call a row records. A short closed set, because the kind is part of
#  the key and a typo would silently create a second cache that never hits.
KIND_EXTRACTION = "extraction"
KIND_GRADE = "grade"
KINDS = (KIND_EXTRACTION, KIND_GRADE)


def extraction_key(passage_text: str) -> str:
    """Cache key for extracting from one passage."""
    return hashlib.sha256((passage_text or "").encode("utf-8")).hexdigest()


def grade_key(question: str, passage_text: str) -> str:
    """Cache key for grading one passage against one question.

    A NUL separator so that ``("ab", "c")`` and ``("a", "bc")`` cannot collide —
    unlikely with free-text questions, and free to prevent.
    """
    blob = f"{question or ''}\x00{passage_text or ''}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def enabled() -> bool:
    return bool(get_settings().llm_cache_enabled)


@contextlib.contextmanager
def _isolated(db):
    """Contain a cache operation in a SAVEPOINT so its failure stays its own.

    Both functions below promise never to raise, and they kept that promise by
    calling ``db.rollback()`` — which rolls back the *caller's* transaction. The
    cache is called from the extraction and grading loops, where the session is
    holding a half-built brief and its claims, so a cache key that was too long for
    its column silently discarded real work and the subsequent commit wrote nothing.
    No error surfaced, because the failure had already been logged as a cache miss.

    This is bug 19's shape a second time: an optional component taking down the
    request it was meant to accelerate. There it was the dense retriever poisoning a
    Postgres transaction; here it is a best-effort write destroying committed-bound
    data on every backend.

    A SAVEPOINT alone is not enough, and the reason is specific. A failed
    ``Session.flush()`` cannot be contained by one: SQLAlchemy rolls the savepoint
    back itself and then requires a full ``Session.rollback()`` before the session
    can be used again, because a half-applied flush may have left the identity map
    inconsistent. Measured, not assumed — a contained flush failure leaves the next
    query raising ``PendingRollbackError``. So the cache writes through Core
    ``insert``/``update`` statements instead, which carry no ORM state, and a
    SAVEPOINT does contain those. It costs nothing here: this is one table with no
    relationships.
    """
    nested = db.begin_nested()
    try:
        yield
    except Exception:
        if nested.is_active:
            nested.rollback()
        raise
    else:
        if nested.is_active:
            nested.commit()


def _match(entry, kind: str, key: str, model: str, prompt_version: str):
    return (
        entry.kind == kind,
        entry.cache_key == key,
        entry.model == model,
        entry.prompt_version == (prompt_version or ""),
    )


def get(db, kind: str, key: str, *, model: str, prompt_version: str = "") -> dict | None:
    """Look up a cached payload, or None.

    Never raises, and never at the caller's expense. A cache that can fail a
    request is worse than no cache; one that can silently delete the request's work
    is worse than either.
    """
    if not enabled() or db is None:
        return None
    try:
        from sqlalchemy import select, update

        from cnms_fom.db.models import LlmCacheEntry

        with _isolated(db):
            row = db.execute(
                select(
                    LlmCacheEntry.id, LlmCacheEntry.payload, LlmCacheEntry.hit_count
                ).where(*_match(LlmCacheEntry, kind, key, model, prompt_version))
            ).first()
            if row is None:
                return None
            #  Usage counters are best-effort. Counted in SQL rather than read,
            #  incremented and written back, so two workers sharing a passage
            #  cannot each overwrite the other's increment.
            db.execute(
                update(LlmCacheEntry)
                .where(LlmCacheEntry.id == row.id)
                .values(
                    hit_count=LlmCacheEntry.hit_count + 1,
                    last_used_at=datetime.now(timezone.utc),
                )
            )
            payload = row.payload
        return json.loads(payload) if isinstance(payload, str) else payload
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cache lookup failed (%s); treating as a miss.", exc)
        return None


def put(
    db, kind: str, key: str, payload: dict, *, model: str, prompt_version: str = ""
) -> None:
    """Store a payload. Never raises, for the same reason as :func:`get`."""
    if not enabled() or db is None:
        return
    if kind not in KINDS:
        logger.warning("Refusing to cache unknown kind %r.", kind)
        return
    try:
        from sqlalchemy import insert, select, update

        from cnms_fom.db.models import LlmCacheEntry

        with _isolated(db):
            existing = db.execute(
                select(LlmCacheEntry.id).where(
                    *_match(LlmCacheEntry, kind, key, model, prompt_version)
                )
            ).scalar_one_or_none()
            now = datetime.now(timezone.utc)
            if existing is not None:
                db.execute(
                    update(LlmCacheEntry)
                    .where(LlmCacheEntry.id == existing)
                    .values(payload=payload, last_used_at=now)
                )
            else:
                db.execute(
                    insert(LlmCacheEntry).values(
                        kind=kind,
                        cache_key=key,
                        model=model,
                        prompt_version=prompt_version or "",
                        payload=payload,
                        hit_count=0,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        #  No db.rollback() here. The SAVEPOINT has undone the cache's own writes,
        #  and rolling back the session would discard the caller's. The likeliest
        #  arrival here is two workers racing to insert the same key, which is a
        #  cache hit that came a moment too late — nothing to report.
        logger.debug("Cache write failed (%s); continuing uncached.", exc)


def stats(db) -> dict:
    """Cache size and hit counts, for reporting what the cache is actually saving."""
    from sqlalchemy import func

    from cnms_fom.db.models import LlmCacheEntry

    rows = (
        db.query(
            LlmCacheEntry.kind,
            LlmCacheEntry.model,
            func.count(LlmCacheEntry.id),
            func.sum(LlmCacheEntry.hit_count),
        )
        .group_by(LlmCacheEntry.kind, LlmCacheEntry.model)
        .all()
    )
    entries = [
        {"kind": kind, "model": model, "entries": int(count), "hits": int(hits or 0)}
        for kind, model, count, hits in rows
    ]
    total_entries = sum(e["entries"] for e in entries)
    total_hits = sum(e["hits"] for e in entries)
    return {
        "enabled": enabled(),
        "by_kind_and_model": entries,
        "total_entries": total_entries,
        "total_hits": total_hits,
        #  Calls avoided over calls that would have been made. The number worth
        #  quoting, because entries alone says nothing about whether it helped.
        "calls_avoided": total_hits,
        "hit_rate": (total_hits / (total_hits + total_entries)) if (total_hits + total_entries) else 0.0,
        "note": (
            "This cache holds a transcript of model calls, keyed by a hash of the content that "
            "was sent. It stores no measurement and nothing citable, and clearing it costs only "
            "time."
        ),
    }


def clear(db, *, kind: str | None = None, model: str | None = None) -> int:
    """Delete cache entries. Returns how many were removed."""
    from cnms_fom.db.models import LlmCacheEntry

    query = db.query(LlmCacheEntry)
    if kind:
        query = query.filter(LlmCacheEntry.kind == kind)
    if model:
        query = query.filter(LlmCacheEntry.model == model)
    removed = query.delete(synchronize_session=False)
    db.flush()
    logger.info("Cleared %d cache entr(ies).", removed)
    return int(removed)


def max_parallel() -> int:
    return max(1, int(get_settings().llm_max_parallel))


def map_cached(
    db,
    items: list,
    *,
    kind: str,
    key_of,
    model: str,
    call,
    prompt_version: str = "",
    max_workers: int | None = None,
) -> tuple[list, int, int, dict[int, str]]:
    """Cache-or-call across a list of items, running the calls concurrently.

    Returns ``(payloads in item order, n_cached, n_called, errors by index)``.

    The errors are returned rather than logged and dropped: a caller reporting
    "produced nothing usable" where it could have said "connection refused" has
    turned a fixable problem into a mysterious one.

    The three phases are separated deliberately, and the reason is the database
    rather than the model: a SQLAlchemy ``Session`` is not thread-safe, so every
    cache read and write happens on the calling thread and only the model calls —
    which touch nothing shared — run in the pool.

    ``call(item)`` returns the payload to cache, or ``None`` for a failure that must
    not be remembered. Caching a failure would make one unreachable server poison
    every later run, which is the kind of bug that looks like a corpus problem.
    """
    from concurrent.futures import ThreadPoolExecutor

    payloads: list = [None] * len(items)
    misses: list[tuple[int, object, str]] = []

    #  1. Serial: what do we already have?
    for index, item in enumerate(items):
        key = key_of(item)
        hit = get(db, kind, key, model=model, prompt_version=prompt_version)
        if hit is not None:
            payloads[index] = hit
        else:
            misses.append((index, item, key))

    errors: dict[int, str] = {}
    if not misses:
        return payloads, len(items), 0, errors

    #  2. Parallel: the model calls, which are independent of each other.
    workers = min(max_workers or max_parallel(), len(misses))
    results: dict[int, object] = {}
    if workers <= 1:
        for index, item, _ in misses:
            results[index] = _safe_call(call, item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_safe_call, call, item): index for index, item, _ in misses
            }
            for future, index in futures.items():
                results[index] = future.result()

    #  3. Serial: record what came back, except the failures.
    called = 0
    for index, _item, key in misses:
        outcome = results.get(index)
        called += 1
        if isinstance(outcome, _CallFailure):
            errors[index] = outcome.message
            payloads[index] = None
            continue
        payloads[index] = outcome
        if outcome is not None:
            put(db, kind, key, outcome, model=model, prompt_version=prompt_version)

    return payloads, len(items) - called, called, errors


@dataclass
class _CallFailure:
    """A transport failure. Carries the reason and is never cached."""

    message: str


def _safe_call(call, item):
    """Run one model call, turning an exception into a non-cacheable failure."""
    try:
        return call(item)
    except Exception as exc:  # noqa: BLE001 - one failed passage must not lose the batch
        logger.warning("A cached model call failed: %s", exc)
        return _CallFailure(f"{type(exc).__name__}: {exc}")
