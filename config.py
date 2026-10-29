"""
Central configuration and shared factories for the NIT-D regulations chatbot.

WHY THIS FILE EXISTS
--------------------
The embedding model used to BUILD the vector store (ingest.py) and the one used
to QUERY it (app.py / rag.py) must be byte-for-byte identical. If they ever
drift, retrieval silently returns garbage with no error. Defining them in ONE
place - here - makes that drift impossible. Every other module imports its
settings from this file; nothing is duplicated.

This is the single file a maintainer edits to change a model, path, or knob.
"""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# --- Programs (UG / PG) ------------------------------------------------------
# The bot serves two separate rule books: UG and PG regulations. They overlap in
# TOPIC (branch change, grading, attendance) but DIFFER in the actual numbers, so
# they must NEVER share a vector store - a PG query retrieving a UG clause would
# answer with the wrong figure. Each program therefore has its own PDF, its own
# parse cache, and its own Chroma store on disk. The user picks the program in
# the UI; retrieval only ever opens that program's store, so the two can't mix.
#
# Everything BELOW this block (embedder, LLM, chunking, retrieval knobs) is shared
# across both programs - only the source document and where it lands differ here.
PROGRAMS = {
    "ug": {
        "label": "UG Regulations",
        "pdf": "ugregulations.pdf",
        "chroma_dir": "./vector_store_ug",
        "parse_cache": "parsed_cache_ug.md",
        "forms_dir": "forms",            # downloadable appendix forms (UG only for now)
        "has_forms": True,               # run appendix-form classification during ingest
        "use_faq": True,                 # build with the curated FAQ.md layer
        "faq_path": "FAQ.md",
        # Used only to validate section detection (a printed warning, not a hard stop).
        "expected_clauses": set(range(1, 16)),               # Clauses 1..15
        "expected_appendices": ["I", "II", "III", "IV", "V", "VI",
                                "VII", "VIII", "IX", "X", "XI", "XII"],
    },
    "pg": {
        "label": "PG Regulations",
        "pdf": "pgregulations.pdf",
        "chroma_dir": "./vector_store_pg",
        "parse_cache": "parsed_cache_pg.md",
        "forms_dir": "forms_pg",
        "has_forms": True,               # PG forms X, XI, XII, XIII catalogued in forms.py
        "use_faq": False,                # no PG FAQ yet
        "faq_path": None,
        "expected_clauses": set(range(1, 16)),               # PG TOC confirms Clauses 1..15
        "expected_appendices": ["I", "II", "III", "IV", "V", "VI", "VII",
                                "VIII", "IX", "X", "XI", "XII", "XIII"],  # PG has 13
    },
}

DEFAULT_PROGRAM = "ug"


def get_program(program):
    """Return one program's settings dict, or raise a clear error for a typo."""
    try:
        return PROGRAMS[program]
    except KeyError:
        raise ValueError(f"Unknown program {program!r}. Choose one of: {list(PROGRAMS)}")


# --- Chunking ----------------------------------------------------------------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# --- Retrieval ---------------------------------------------------------------
TOP_K = 4            # Final hits kept after re-ranking. Each expands into its
                     # whole section via sibling-fetch (rag.retrieve_with_siblings).

# Two-stage retrieval: pull RERANK_TOP_N candidates by embedding similarity, then
# a cross-encoder re-scores them and we keep the best TOP_K. The cross-encoder
# reads query + chunk together, so it is far more precise than embedding distance
# - it lifts the right clause to the top without the noise that simply raising
# TOP_K introduced. A small model is used so it runs on CPU; if it can't load,
# retrieval falls back to plain embedding order (see rag._rerank).
RERANK_TOP_N = 15
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # ~80 MB, CPU-friendly

# Max characters of retrieved context sent to the LLM per request. Sibling-fetch
# can pull large sections; this bounds the request so it stays under Groq's
# free-tier input-tokens-per-minute limit (7000 ITPM). ~8000 chars ~= 2000
# tokens, leaving ample room for the prompt, history and answer. Raise it if you
# move to a paid tier.
MAX_CONTEXT_CHARS = 8000

# --- Models ------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DEVICE = "cpu"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

LLM_MODEL = "qwen/qwen3.8-27b"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 800


# --- Factories ---------------------------------------------------------------
# Both ingestion and serving call these, so the configuration can never diverge.

def get_embeddings():
    """The one embedding model for the whole system."""
    return HuggingFaceBgeEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        query_instruction=QUERY_INSTRUCTION,
    )


def get_llm():
    """The one chat LLM for the whole system."""
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        reasoning_effort="none",
        max_tokens=LLM_MAX_TOKENS,
    )


_reranker = None


def get_reranker():
    """The cross-encoder re-ranker, loaded once and reused. Imported lazily so a
    machine without it (or short on memory) only fails at call time, where
    rag._rerank catches it and falls back to embedding order."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL, device=EMBED_DEVICE)
    return _reranker
