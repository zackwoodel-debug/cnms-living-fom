"""Retrieval-augmented generation over the synthesis corpus.

Runs entirely locally through Ollama — MBE/PLD/ALD process notes and CNMS user
documents do not leave the machine.

The one rule that shapes every module here: retrieval supports *reading*, never
*data entry*.  FOM_PROOF Sec. 2.3 forbids filling a missing property from a
plausible number, and a language model is an extremely efficient source of
plausible numbers.  See ``chains.SYNTHESIS_SYSTEM_PROMPT`` and
``chains.assert_not_property_ingestion``.
"""

from .chains import RagAnswer, answer_question  # noqa: F401
from .ingest import ingest_directory, ingest_pdf  # noqa: F401
from .vectorstore import ChunkHit, search_chunks  # noqa: F401
