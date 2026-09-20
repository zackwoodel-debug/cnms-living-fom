"""Conversation persistence for the research assistant.

Two things are stored and they serve different purposes.

**History** is replayed to the model so follow-ups work: "and how thick was the
interlayer?" needs the previous turn to mean anything.  Only the *text* of each
turn is replayed — not the tool calls, not the tool results.  Replaying raw
results would spend the whole context window re-reading fit records the
assistant already summarised, and by turn four there would be no room left for
the retrieval instructions, which are the first thing to fall out and the last
thing you want to lose.

**Evidence** is stored and never replayed.  ``ChatMessage.evidence`` and
``tool_calls`` hold the exact chunk ids, similarities, tool arguments, and
results behind each answer.  That is the audit trail: six months on, "where did
that 103 Å come from?" resolves to a fit record id and a technique, not to a
conversation someone half-remembers.

The split matters because these two needs pull in opposite directions — the model
needs less, the auditor needs everything — and a single "conversation log" serving
both ends up too big to replay and too lossy to audit.
"""

from __future__ import annotations

import logging
import uuid

from cnms_fom.db.enums import ChatRole, SynthesisTechnique

logger = logging.getLogger(__name__)

#  How many prior turns to replay. Six user+assistant pairs is enough for the
#  referential follow-ups this assistant actually gets ("and the roughness?",
#  "compare that to the SE fit") without the prompt growing without bound.
DEFAULT_HISTORY_TURNS = 6

#  Truncation cap for a replayed assistant answer. Long answers are mostly
#  citations and caveats, which the model does not need re-fed to interpret the
#  next question; the full text stays in the database either way.
MAX_REPLAYED_CHARS = 4_000


def new_session_key() -> str:
    return uuid.uuid4().hex


def get_or_create_session(
    db,
    session_key: str | None = None,
    *,
    title: str | None = None,
    user: str | None = None,
    sample_id: str | None = None,
    techniques: list[SynthesisTechnique] | None = None,
    provider: str | None = None,
    chat_model: str | None = None,
):
    """Fetch a conversation by key, or start one.

    A key that does not exist creates a session with that key rather than
    erroring: a client generating its own ids is the normal case, and making it
    call a separate create endpoint first buys nothing.
    """
    from cnms_fom.db.models import ChatSession

    key = session_key or new_session_key()
    session = db.query(ChatSession).filter(ChatSession.session_key == key).one_or_none()
    if session is not None:
        #  Scope may be supplied on a later turn ("actually, this is about
        #  PILOT-07"). Fill what is unset; never overwrite what is set, since
        #  silently re-pointing a conversation at another sample would
        #  reattribute every answer in it.
        if sample_id and not session.sample_id:
            session.sample_id = sample_id
        if techniques and not session.techniques:
            session.techniques = [t.value for t in techniques]
        return session

    session = ChatSession(
        session_key=key,
        title=title,
        user=user,
        sample_id=sample_id,
        techniques=[t.value for t in techniques] if techniques else None,
        provider=provider,
        chat_model=chat_model,
    )
    db.add(session)
    db.flush()
    return session


def load_history(db, session, *, turns: int = DEFAULT_HISTORY_TURNS) -> list[dict]:
    """The last ``turns`` exchanges, as provider-shaped messages.

    Text only — see the module docstring for why the tool results stay in the
    database rather than going back to the model.
    """
    from cnms_fom.db.models import ChatMessage

    rows = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session.id,
            ChatMessage.role.in_([ChatRole.USER, ChatRole.ASSISTANT]),
        )
        .order_by(ChatMessage.turn_index.desc())
        .limit(turns * 2)
        .all()
    )
    rows.reverse()

    messages: list[dict] = []
    for row in rows:
        content = row.content
        if row.role is ChatRole.ASSISTANT and len(content) > MAX_REPLAYED_CHARS:
            content = (
                content[:MAX_REPLAYED_CHARS]
                + "\n[... earlier answer truncated for context; full text and its evidence are "
                "stored on this conversation ...]"
            )
        messages.append({"role": row.role.value, "content": content})

    #  A history that opens on an assistant turn (the window cut mid-exchange)
    #  is malformed for every provider. Drop the orphan rather than repair it.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    return messages


def _next_turn_index(db, session) -> int:
    from sqlalchemy import func

    from cnms_fom.db.models import ChatMessage

    highest = (
        db.query(func.max(ChatMessage.turn_index))
        .filter(ChatMessage.session_id == session.id)
        .scalar()
    )
    return 0 if highest is None else int(highest) + 1


def record_turn(db, session, question: str, answer) -> tuple:
    """Persist one exchange: the question, the answer, and its evidence.

    ``answer`` is an ``agent.AgentAnswer``.  Tool results are stored whole — they
    are the evidence, and a summarised audit trail is not one.
    """
    from cnms_fom.db.models import ChatMessage

    index = _next_turn_index(db, session)

    user_message = ChatMessage(
        session_id=session.id,
        turn_index=index,
        role=ChatRole.USER,
        content=question,
    )
    assistant_message = ChatMessage(
        session_id=session.id,
        turn_index=index + 1,
        role=ChatRole.ASSISTANT,
        content=answer.answer,
        tool_calls=[step.as_dict() for step in answer.steps] or None,
        evidence=answer.citations or None,
        insufficient_context=answer.insufficient_context,
        chat_model=answer.model or None,
        latency_ms=answer.latency_ms,
    )
    db.add_all([user_message, assistant_message])

    if not session.title:
        #  First question becomes the conversation's title, so a list of
        #  sessions is readable without opening each one.
        session.title = question.strip()[:200]
    if answer.model and not session.chat_model:
        session.chat_model = answer.model
    if answer.provider and not session.provider:
        session.provider = answer.provider

    db.flush()
    return user_message, assistant_message


def session_summary(db, session) -> dict:
    """One conversation's metadata and turn counts.

    ``n_data_gaps`` is reported because a refusal rate is a corpus-coverage
    metric: a conversation that is mostly gaps is telling you what to ingest next,
    not that the assistant is broken.
    """
    from sqlalchemy import func

    from cnms_fom.db.models import ChatMessage

    counts = (
        db.query(ChatMessage.role, func.count(ChatMessage.id))
        .filter(ChatMessage.session_id == session.id)
        .group_by(ChatMessage.role)
        .all()
    )
    gaps = (
        db.query(func.count(ChatMessage.id))
        .filter(
            ChatMessage.session_id == session.id,
            ChatMessage.insufficient_context.is_(True),
        )
        .scalar()
        or 0
    )
    return {
        "session_key": session.session_key,
        "title": session.title,
        "user": session.user,
        "sample_id": session.sample_id,
        "techniques": session.techniques,
        "provider": session.provider,
        "chat_model": session.chat_model,
        "created_at": str(session.created_at) if session.created_at else None,
        "messages_by_role": {
            (role.value if hasattr(role, "value") else str(role)): int(n) for role, n in counts
        },
        "n_data_gaps": int(gaps),
    }


def list_sessions(db, *, limit: int = 25, user: str | None = None) -> list[dict]:
    """Recent conversations, newest first."""
    from cnms_fom.db.models import ChatSession

    query = db.query(ChatSession)
    if user:
        query = query.filter(ChatSession.user == user)
    sessions = query.order_by(ChatSession.created_at.desc()).limit(limit).all()
    return [session_summary(db, session) for session in sessions]


def transcript(db, session) -> list[dict]:
    """The full conversation with its evidence, for audit and for the UI."""
    from cnms_fom.db.models import ChatMessage

    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.turn_index)
        .all()
    )
    return [
        {
            "turn_index": row.turn_index,
            "role": row.role.value,
            "content": row.content,
            "tool_calls": row.tool_calls,
            "evidence": row.evidence,
            "insufficient_context": row.insufficient_context,
            "chat_model": row.chat_model,
            "latency_ms": row.latency_ms,
            "created_at": str(row.created_at) if row.created_at else None,
        }
        for row in rows
    ]
