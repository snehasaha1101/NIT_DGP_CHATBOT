"""
Ingestion pipeline: PDF -> parsed markdown -> namespaced sections -> chunks ->
embedded vector store.

Section detection is based on the REAL structure of the NIT-D regulations
(verified against the PDF):
  - Body clauses are headed "N.0 Title" for N = 1..15 (15 = Exceptions).
  - Appendices live under "16.0 Appendices"; each appendix page is titled
    "Appendix-<Roman>" and the index lists them as "16. <Roman>".
  - Repealed text is struck through in the PDF.

Detection NORMALIZES each line (strips bold/underline/heading markup, normalizes
dashes) before matching, so it is robust to how LlamaParse renders emphasis -
the brittleness that made the first version miss appendices I-IX.

All tunable settings live in config.py; appendix-form logic lives in forms.py.
"""

import os
import re
import html
import shutil

from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from forms import FORM_SUMMARIES, classify_appendix_form


# --- Parse caching -----------------------------------------------------------
# LlamaParse is non-deterministic: the same PDF can parse to different markdown
# (different page count, different table rendering) run to run. Freezing one
# parse to disk makes every later rebuild identical - and saves credits. Each
# program caches to its own file (cfg["parse_cache"]); delete that file to force
# a fresh parse (e.g. after that program's regulations are updated).
_PAGE_SEP = "\n\n@@@PAGE_BREAK@@@\n\n"


class _Page:
    """Minimal stand-in for a LlamaParse document (only .text is used)."""
    def __init__(self, text):
        self.text = text


def _load_parsed_docs(cfg):
    cache = cfg["parse_cache"]
    if os.path.exists(cache):
        print(f"Using cached parse from {cache} (delete it to re-parse).")
        with open(cache, encoding="utf-8") as f:
            return [_Page(t) for t in f.read().split(_PAGE_SEP)]

    if not os.environ.get("LLAMA_CLOUD_API_KEY"):
        print("ERROR: LLAMA_CLOUD_API_KEY not set. Please set it in a .env file.")
        raise SystemExit(1)
    print(f"Starting LlamaParse extraction of {cfg['pdf']} (This takes a moment)...")
    docs = LlamaParse(result_type="markdown", verbose=True).load_data(cfg["pdf"])
    with open(cache, "w", encoding="utf-8") as f:
        f.write(_PAGE_SEP.join(d.text for d in docs))
    print(f"Parsed and cached to {cache} (delete it to re-parse next time).")
    return docs


# --- Text cleaning -----------------------------------------------------------
# Repealed amendments are struck through in the source. IF LlamaParse preserves
# strikethrough as ~~...~~, this removes the repealed text so the bot can never
# quote an old rule (e.g. the superseded 8.75 branch-change cutoff). If the
# parser drops strikethrough entirely, run inspect_parse.py to check whether the
# old numbers survived as plain text - that case needs manual handling.
STRIKE_RE = re.compile(r"~~.*?~~", re.DOTALL)


def _clean(line):
    """Normalize a line for heading detection: strip HTML tags (LlamaParse wraps
    headings like '# <u>**Appendix-I**</u>'), leading markdown (#, *, _, >,
    whitespace), and normalize dashes - so detection doesn't depend on how the
    heading was rendered."""
    s = line.strip()
    s = re.sub(r"<[^>]+>", "", s)                         # strip HTML tags, e.g. <u>, </u>
    s = s.lstrip("#>*_ \t")
    s = s.replace("–", "-").replace("—", "-")   # en/em dash -> hyphen
    s = s.replace("*", "").replace("_", "")               # drop inline emphasis
    return s.strip()


# --- Section detection patterns ---------------------------------------------
# Roman numerals are matched case-SENSITIVE (uppercase) so a lowercase word like
# "via" after "appendix" can't be misread as "Appendix-VI". The real headings
# look like "# <u>**Appendix-I**</u>" or "# <u>Appendix- X</u>" in the parsed
# markdown, which _clean() reduces to "Appendix-I" / "Appendix- X" before match.
APPENDIX_RE = re.compile(r"^(?i:appendix)\s*-?\s*([IVX]+)\b")  # word any-case, Roman upper-case
CLAUSE_RE = re.compile(r"^(\d{1,2})\.(\d+)?\b")
# The closing page. UG titles it "...USEFUL GUIDANCE FOR THE STUDENTS"; PG titles
# it "...USEFUL INFORMATION FOR THE STUDENTS" - match either so both documents'
# trailer page gets its own section instead of merging into the last appendix.
GUIDANCE_RE = re.compile(r"^some important and useful (guidance|information)", re.IGNORECASE)


def build_sections(parsed_docs, program, has_forms=True):
    """Group parsed pages into ordered, namespaced sections, merging a clause's
    text across page breaks. Returns (sections, replaced_form_names).

    program: picks which program's form summaries/detector to use (forms.py).
    has_forms: when False, skip the appendix-form replacement so appendix pages
    ingest as their own plain text."""
    sections = []
    replaced = set()
    summaries = FORM_SUMMARIES.get(program, {})
    state = {"region": "front", "sid": "front-matter"}

    def append(sid, text, kind="body"):
        if sections and sections[-1]["section_id"] == sid and sections[-1]["type"] == kind:
            sections[-1]["text"] += "\n" + text
        else:
            sections.append({"section_id": sid, "text": text, "type": kind})

    for i, doc in enumerate(parsed_docs):
        page_text = STRIKE_RE.sub("", doc.text or "")   # drop repealed (struck) text

        # Blank appendix FORM pages -> replace with a describable summary.
        name = classify_appendix_form(page_text, program) if has_forms else None
        if name and name not in replaced:
            sid = f"appendix-{name.split('-')[1]}"       # "Appendix-X" -> "appendix-X"
            sections.append({"section_id": sid, "text": summaries[name], "type": "form_summary"})
            replaced.add(name)
            state["region"], state["sid"] = "appendix", sid
            continue

        # Each front-matter page gets its own id so cover/index pages don't merge
        # into one giant section. Once we enter clauses/appendices we never
        # return to 'front', so this only affects the genuine opening pages.
        if state["region"] == "front":
            state["sid"] = f"front-{i}"

        for line in page_text.splitlines():
            c = _clean(line)

            app = APPENDIX_RE.match(c)
            if app:
                # First real "Appendix-<Roman>" heading flips us into the
                # appendix region for good (appendices are the last thing in the
                # document), so clause detection stops and appendix numbered
                # lists ("1.", "2.") can't leak back into clause-* buckets.
                state["region"] = "appendix"
                state["sid"] = f"appendix-{app.group(1)}"
            elif state["region"] == "appendix" and GUIDANCE_RE.match(c):
                # Closing guidance page: its own section, so its numbered tips
                # ("1.".."12.") don't merge into the last appendix or get
                # mistaken for clauses.
                # Require region=="appendix" because guidance is the FINAL
                # section (after all appendices). Without this guard, the
                # page-1 table of contents - which also lists this title -
                # matches and swallows the whole document into "guidance".
                state["region"] = "tail"
                state["sid"] = "guidance"
            elif state["region"] not in ("appendix", "tail"):
                cl = CLAUSE_RE.match(c)
                if cl and 1 <= int(cl.group(1)) <= 15:
                    state["region"] = "clause"
                    state["sid"] = f"clause-{int(cl.group(1))}"

            append(state["sid"], line)   # keep the ORIGINAL line text

    return sections, replaced


def _is_table_line(line):
    """True for a line that is part of a table, in EITHER format LlamaParse
    emits: markdown pipe rows ('| a | b |') or HTML tables. HTML tables may put
    every tag on its own line, INCLUDING closing tags like </tr> and </thead>;
    we must match those too, or the table fragments at every row boundary."""
    s = line.strip().lower()
    if s.startswith("|"):
        return True
    return any(tag in s for tag in (
        "<table", "</table", "<thead", "</thead", "<tbody", "</tbody",
        "<tr", "</tr", "<td", "</td", "<th", "</th",
    ))


def _table_caption(table_text):
    """Build a short prose caption from a table's header row, so the table is
    findable by column name even though its body embeds poorly. Handles markdown
    (one-line header) and HTML (header cells may each be on their own line)."""
    cells = []
    for line in table_text.splitlines():
        s = line.strip()
        low = s.lower()
        if not s:
            continue
        if s.startswith("|"):
            got = [c.strip() for c in s.strip("|").split("|")]
            got = [c for c in got if c and (set(c) - set("-: "))]
            if got:
                return "Table listing " + ", ".join(got) + "."
        else:
            for c in re.findall(r">([^<]+)<", s):                      # HTML cell text
                c = c.strip()
                if c and (set(c) - set("-: ")):
                    cells.append(c)
            # Header ends where the body (<td>/<tbody>) or </thead> begins.
            if cells and ("<td" in low or "<tbody" in low or "</thead" in low):
                break
    if cells:
        return "Table listing " + ", ".join(cells) + "."
    return "Table."


def _html_table_to_text(text):
    """Convert an HTML table to compact pipe rows ('a | b | c' per row). HTML
    tables are tag-heavy - roughly 4x the size of the data - which makes them
    get evicted by the context budget; the compact form fits, embeds cleaner,
    and reads better for the LLM. Cells of one row may be on separate source
    lines, so row structure is rebuilt from </tr>, not from newlines."""
    t = text.replace("\n", " ")
    t = re.sub(r"(?i)</tr>", "\n", t)        # row end  -> newline
    t = re.sub(r"(?i)</t[dh]>", " | ", t)    # cell end -> separator
    t = re.sub(r"<[^>]+>", "", t)            # drop remaining tags
    t = html.unescape(t)                     # &#x26; -> &, etc.
    rows = []
    for line in t.split("\n"):
        line = re.sub(r"\s+", " ", line).strip().strip("|").strip()
        line = re.sub(r"\s*\|\s*", " | ", line)
        if line:
            rows.append(line)
    return "\n".join(rows)


def _section_label(lines):
    """A short topic label from a section's heading line(s), e.g.
    'Appendix-V - RULES REGARDING SUPPLEMENTARY / BACKLOG EXAMINATION'. Prepended
    to each table's caption so the table carries its topic and can be retrieved by
    it (a generic 'Table listing ...' caption does not match a query like
    'supplementary grade', so the table ranks low and gets trimmed)."""
    headings = []
    for ln in lines:
        if ln.strip().startswith("#"):
            h = _clean(ln)
            if h:
                headings.append(h)
        if len(headings) >= 2:
            break
    return " - ".join(headings)


def chunk_section(doc, splitter):
    """Chunk one section, keeping each table (markdown OR HTML) WHOLE as one
    atomic chunk with a caption, and splitting the surrounding prose normally.
    This stops a table from being cut mid-rows, which left the LLM with a
    fragment it couldn't use."""
    lines = doc.page_content.splitlines()
    section_label = _section_label(lines)
    chunks, prose = [], []

    def flush_prose():
        text = "\n".join(prose).strip()
        prose.clear()
        if text:
            for piece in splitter.split_text(text):
                chunks.append(Document(page_content=piece, metadata=dict(doc.metadata)))

    i, n = 0, len(lines)
    while i < n:
        if _is_table_line(lines[i]):
            table = []
            while i < n and _is_table_line(lines[i]):
                table.append(lines[i])
                i += 1
            flush_prose()                     # keep order: prose before this table
            table_text = "\n".join(table)
            caption = _table_caption(table_text)              # from the header (handles HTML)
            if section_label:
                caption = f"{section_label}. {caption}"        # carry the section's topic
            if "<" in table_text and ">" in table_text:
                table_text = _html_table_to_text(table_text)  # compact HTML -> pipe rows
            meta = dict(doc.metadata)
            meta["type"] = "table"
            chunks.append(Document(
                page_content=caption + "\n" + table_text,
                metadata=meta,
            ))
        else:
            prose.append(lines[i])
            i += 1
    flush_prose()
    return chunks


def _print_section_map(chunks, cfg):
    """Show how the document was carved up and warn on anything missing, against
    this program's expected clauses/appendices (cfg)."""
    counts = {}
    for ch in chunks:
        sid = ch.metadata["section_id"]
        counts[sid] = counts.get(sid, 0) + 1

    print("\n--- SECTION MAP (section_id : #chunks) ---")
    for sid in sorted(counts):
        print(f"  {sid:16s} : {counts[sid]}")

    found_clauses = {int(s.split("-")[1]) for s in counts if s.startswith("clause-")}
    found_apps = {s.split("-")[1] for s in counts if s.startswith("appendix-")}
    missing_clauses = sorted(cfg["expected_clauses"] - found_clauses)
    # An empty expected_appendices (PG, before we know the real list) means "don't
    # check" - just report what was found so you can fill the list in afterwards.
    missing_apps = [a for a in cfg["expected_appendices"] if a not in found_apps]
    if missing_clauses:
        print(f"  WARNING: expected clauses not detected: {missing_clauses}")
    if missing_apps:
        print(f"  WARNING: expected appendices not detected: {missing_apps}")
    if not cfg["expected_appendices"]:
        print(f"  Appendices detected (no expected list set): {sorted(found_apps)}")
    if not missing_clauses and not missing_apps and cfg["expected_appendices"]:
        print("  All expected clauses and appendices detected.")
    print("-----------------------------------------\n")


def process_and_ingest(program=config.DEFAULT_PROGRAM):
    cfg = config.get_program(program)
    print(f"=== Ingesting {cfg['label']} ({program}) ===")

    parsed_docs = _load_parsed_docs(cfg)
    print(f"Found {len(parsed_docs)} pages/sections.")

    sections, replaced = build_sections(parsed_docs, program, has_forms=cfg["has_forms"])

    if cfg["has_forms"]:
        missing_forms = [n for n in FORM_SUMMARIES.get(program, {}) if n not in replaced]
        if missing_forms:
            print(f"WARNING: appendix form pages not found by content: {missing_forms}")
            print("  -> The PDF layout may have changed; check the appendix pages.")

    faq_path = cfg["faq_path"]
    if cfg["use_faq"] and faq_path and os.path.exists(faq_path):
        with open(faq_path, "r", encoding="utf-8") as f:
            sections.append({"section_id": "faq", "text": f.read(), "type": "curated_table_layer"})
        print(f"Ingesting curated {faq_path} layer.")
    elif cfg["use_faq"]:
        print(f"Warning: use_faq is True but {faq_path} not found. Proceeding without it.")
    else:
        print("NO-FAQ build: curated FAQ layer skipped on purpose.")

    section_docs = [
        Document(page_content=s["text"],
                 metadata={"section_id": s["section_id"], "type": s["type"]})
        for s in sections
    ]

    print("Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for d in section_docs:
        chunks.extend(chunk_section(d, splitter))   # tables kept whole, prose split
    for seq, ch in enumerate(chunks):
        ch.metadata["seq"] = seq
    n_tables = sum(1 for ch in chunks if ch.metadata.get("type") == "table")
    print(f"Created {len(chunks)} chunks across {len(section_docs)} sections "
          f"({n_tables} kept as whole tables).")

    _print_section_map(chunks, cfg)

    print("Creating local embeddings and saving to ChromaDB...")
    if os.path.exists(cfg["chroma_dir"]):
        print("Cleaning up old vector database...")
        shutil.rmtree(cfg["chroma_dir"], ignore_errors=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=config.get_embeddings(),
        persist_directory=cfg["chroma_dir"],
    )
    print(f"Success! Vector database saved to {cfg['chroma_dir']}")


if __name__ == "__main__":
    import sys
    # Usage: python ingest.py [ug|pg]   (defaults to ug)
    prog = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_PROGRAM
    process_and_ingest(prog)
