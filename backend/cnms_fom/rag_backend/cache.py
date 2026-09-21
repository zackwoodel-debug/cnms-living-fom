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


def get(db, kind: str, key: str, *, model: str, prompt_version: str = "") -> dict | None:
    """Look up a cached payload, or None.

    Never raises. A cache that can fail a request is worse than no cache, so a
    broken lookup is logged and treated as a miss.
    """
    if not enabled() or db is None:
        return None
    try:
        from cnms_fom.db.models import LlmCacheEntry

        row = (
            db.query(LlmCacheEntry)
            .filter(
                LlmCacheEntry.kind == kind,
                LlmCacheEntry.cache_key == key,
                LlmCacheEntry.model == model,
                LlmCacheEntry.prompt_version == (prompt_version or ""),
            )
            .one_or_none()
        )
        if row is None:
            return None
        #  Usage counters are best-effort: a cache hit must not fail because a
        #  counter could not be written.
        try:
            row.hit_count = (row.hit_count or 0) + 1
            row.last_used_at = datetime.now(timezone.utc)
            db.flush()
        except Exception:  # noqa: BLE001
            db.rollback()
        return json.loads(row.payload) if isinstance(row.payload, str) else row.payload
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
        from cnms_fom.db.models import LlmCacheEntry

        existing = (
            db.query(LlmCacheEntry)
            .filter(
                LlmCacheEntry.kind == kind,
                LlmCacheEntry.cache_key == key,
                LlmCacheEntry.model == model,
                LlmCacheEntry.prompt_version == (prompt_version or ""),
            )
            .one_or_none()
        )
        if existing is not None:
            existing.payload = payload
            existing.last_used_at = datetime.now(timezone.utc)
        else:
            db.add(LlmCacheEntry(
                kind=kind,
                cache_key=key,
                model=model,
                prompt_version=prompt_version or "",
                payload=payload,
            ))
        db.flush()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cache write failed (%s); continuing uncached.", exc)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass


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
