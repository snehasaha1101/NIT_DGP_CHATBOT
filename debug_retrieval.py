"""
Debug: show exactly which chunks the retriever returns for a query, so we can see
whether the right clause or table is being retrieved (and kept within the context
budget) or missed entirely.

Run:
    python debug_retrieval.py                              # default query, UG store
    python debug_retrieval.py "your question here"         # custom query, UG store
    python debug_retrieval.py "your question here" pg      # custom query, PG store
    python debug_retrieval.py section:appendix-V ug        # dump EVERY chunk in a section

It prints each returned chunk's section, sequence, and a preview, plus the full
text of every chunk so you can see exactly what the model was given. The
section: mode ignores retrieval and lists every chunk stored under that
section_id, so you can check whether a specific table actually made it into the
store at all.
"""

import sys

from rag import load_vectorstore, retrieve_with_siblings, normalize_appendix_refs

args = [a for a in sys.argv[1:]]
QUERY = args[0] if args else "what minor subjects options are there and what are their seat vacancies?"
PROGRAM = args[1] if len(args) > 1 else "ug"

vs = load_vectorstore(PROGRAM)

# --- Section-dump mode: list every chunk stored under a section_id -----------
if QUERY.startswith("section:"):
    section_id = QUERY.split(":", 1)[1]
    raw = vs.get(where={"section_id": section_id})
    docs = list(zip(raw["documents"], raw["metadatas"]))
    docs.sort(key=lambda d: d[1].get("seq", 0))
    print(f"\nProgram: {PROGRAM}")
    print(f"Section: {section_id}  ({len(docs)} chunk(s))\n")
    for content, meta in docs:
        print(f"----- [seq={meta.get('seq')} type={meta.get('type', 'body')}] -----")
        print(content)
        print()
    sys.exit(0)

q = normalize_appendix_refs(QUERY)
docs = retrieve_with_siblings(vs, q)

print(f"\nProgram: {PROGRAM}")
print(f"Query: {QUERY}")
print(f"Normalized: {q}")
print(f"Returned {len(docs)} chunks "
      f"({sum(len(d.page_content) for d in docs)} chars total):\n")

for d in docs:
    sid = d.metadata.get("section_id")
    seq = d.metadata.get("seq")
    preview = d.page_content[:90].replace("\n", " ")
    print(f"[{sid:14s} seq={seq}] {preview!r}")

print("\n" + "=" * 70)
print("FULL TEXT OF EACH RETRIEVED CHUNK")
print("=" * 70)
for d in docs:
    sid = d.metadata.get("section_id")
    seq = d.metadata.get("seq")
    print(f"\n----- [{sid} seq={seq}] -----")
    print(d.page_content)
