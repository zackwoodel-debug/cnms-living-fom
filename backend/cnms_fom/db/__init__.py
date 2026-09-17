"""Database layer: engine/session wiring plus the ORM schema.

Deliberately does *not* re-export from ``base``: ``enums`` is a pure-Python
vocabulary that the science modules depend on, and importing it should not drag
in SQLAlchemy or build a database engine.  Import the session helpers from
``cnms_fom.db.base`` explicitly.
"""
