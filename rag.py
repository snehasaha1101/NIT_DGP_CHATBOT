"""
Retrieval and RAG-chain logic for the NIT-D regulations chatbot.

This module is deliberately FREE of Streamlit so it can be imported and tested
on its own (e.g. by the eval harness). app.py is only a thin UI on top of this.

The two ideas that live here:
  1. normalize_appendix_refs - rewrite "appendix 10" -> "Appendix-X" before
     embedding, so the query speaks the document's symbol language.
  2. retrieve_with_siblings - after a normal similarity search, pull every chunk
     sharing a retrieved chunk's section_id, so a matched half of a clause drags
     in the rest of the clause.
"""

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from config import (
    DEFAULT_PROGRAM, TOP_K, RERANK_TOP_N, MAX_CONTEXT_CHARS,
    get_program, get_embeddings, get_llm, get_reranker,
)


# --- Query normalization -----------------------------------------------------
def _int_to_roman(n):
    """Integer -> Roman numeral. None if outside a sane appendix range."""
    if not (0 < n < 40):
        return None
    numerals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in numerals:
        while n >= value:
            out += symbol
            n -= value
    return out


def normalize_appendix_refs(text):
    """Rewrite 'appendix 10' / 'appendix-10' -> 'Appendix-X'. Algorithmic, so no
    hardcoded appendix list to maintain."""
    if not text:
        return text

    def _repl(m):
        roman = _int_to_roman(int(m.group(1)))
        return f"Appendix-{roman}" if roman else m.group(0)

    return re.sub(r"\bappendix[\s\-]*(\d+)\b", _repl, text, flags=re.IGNORECASE)


# --- Vector store ------------------------------------------------------------
def load_vectorstore(program=DEFAULT_PROGRAM):
    """Open the persisted Chroma store for one program (ug/pg) with the shared
    embedding model. Each program has its own store, so UG and PG never mix."""
    cfg = get_program(program)
    return Chroma(persist_directory=cfg["chroma_dir"], embedding_function=get_embeddings())


# --- Section-aware retrieval -------------------------------------------------
def _rerank(query, docs, k):
    """Re-score (query, chunk) pairs with the cross-encoder and keep the best k.
    Falls back to embedding order if the re-ranker can't load (e.g. low memory)."""
    if len(docs) <= k:
        return docs
    try:
        scores = get_reranker().predict([(query, d.page_content) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in order[:k]]
    except Exception:
        return docs[:k]


def retrieve_with_siblings(vectorstore, query, k=TOP_K):
    """Return the best-k chunks (embedding candidates re-ranked by a cross-encoder)
    PLUS every sibling sharing their section_id, de-duplicated and ordered by `seq`
    so each clause reads top-to-bottom. Pure function of (vectorstore, query).
    """
    # Stage 1: wide candidate net by embedding. Stage 2: precise re-rank to k.
    candidates = vectorstore.as_retriever(search_kwargs={"k": RERANK_TOP_N}).invoke(query)
    hits = _rerank(query, candidates, k)

    section_ids = {d.metadata.get("section_id") for d in hits if d.metadata.get("section_id")}
    if not section_ids:
        return hits

    try:
        raw = vectorstore.get(where={"section_id": {"$in": list(section_ids)}})
        siblings = [
            Document(page_content=c, metadata=m)
            for c, m in zip(raw["documents"], raw["metadatas"])
        ]
    except Exception:
        # Never let a metadata-fetch failure break the answer.
        return hits

    merged, seen = [], set()
    for d in siblings + hits:
        key = d.metadata.get("seq", d.page_content[:60])
        if key not in seen:
            seen.add(key)
            merged.append(d)
    merged.sort(key=lambda d: d.metadata.get("seq", 0))

    return _apply_budget(merged, hits)


def _apply_budget(merged, hits):
    """Keep the context under MAX_CONTEXT_CHARS so a request can't exceed the
    LLM provider's input limit. The hits are always kept, AND so is every table a
    hit dragged in: a table a clause points to ("as elucidated in the table") is
    small and is often the actual answer (e.g. the supplementary grade map A->B),
    so it must survive even when bulky prose siblings compete for space. The
    remaining prose siblings are then added by proximity to a hit (closest first)
    until the budget runs out, and everything is re-ordered by seq so each clause
    still reads top-to-bottom."""
    if sum(len(d.page_content) for d in merged) <= MAX_CONTEXT_CHARS:
        return merged

    hit_seqs = {d.metadata.get("seq") for d in hits}

    def is_protected(d):
        # A hit, or any table (tables are small and dense - never trim one out).
        return d.metadata.get("seq") in hit_seqs or d.metadata.get("type") == "table"

    selected = [d for d in merged if is_protected(d)]
    budget = MAX_CONTEXT_CHARS - sum(len(d.page_content) for d in selected)

    def distance_to_nearest_hit(d):
        s = d.metadata.get("seq", 0)
        return min((abs(s - h) for h in hit_seqs if h is not None), default=0)

    others = sorted(
        (d for d in merged if not is_protected(d)),
        key=distance_to_nearest_hit,
    )
    for d in others:
        if len(d.page_content) <= budget:
            selected.append(d)
            budget -= len(d.page_content)

    selected.sort(key=lambda d: d.metadata.get("seq", 0))
    return selected


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- Chain -------------------------------------------------------------------
_CONTEXTUALIZE_SYSTEM = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

_QA_SYSTEM = """You are an academic assistant for NIT Durgapur students. \
Answer ONLY from the retrieved context below.

- Lead with the direct answer (the grade, number, or yes/no), then give the brief reason. Do NOT deliberate between clauses or think out loud in the response.
- Keep prose answers concise and formal.
- When the answer is a list, a table, or a rule with several lettered or numbered conditions (a, b, c / i, ii, iii), include EVERY relevant row, item, or condition in full - do not summarise, abbreviate, give only examples, or stop early. A requirement that has several parts is only answered correctly when all of its parts are stated.
- When a rule lists banded thresholds (e.g. attendance ranges, mark ranges), match the given value to the exact band it falls in and quote THAT band's range - do not quote a neighbouring band's range.
- If the answer is genuinely not in the provided regulations, say so clearly and suggest the student contact the Academic Section (or check the official Academic Calendar / notification) for that information. Never invent a figure, date, name, or procedure.

Context:
{context}"""


def build_rag_chain(program=DEFAULT_PROGRAM, vectorstore=None):
    """Assemble the full RAG chain for one program (ug/pg). Pass a vectorstore for
    testing, or leave it None to load that program's persisted store."""
    vectorstore = vectorstore or load_vectorstore(program)
    llm = get_llm()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", _CONTEXTUALIZE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", _QA_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def standalone_question(x):
        if x.get("chat_history"):
            return (contextualize_prompt | llm | StrOutputParser()).invoke(x)
        return x["input"]

    def build_context(x):
        query = normalize_appendix_refs(standalone_question(x))
        return _format_docs(retrieve_with_siblings(vectorstore, query))

    return (
        RunnablePassthrough.assign(context=build_context)
        | qa_prompt
        | llm
        | StrOutputParser()
    )