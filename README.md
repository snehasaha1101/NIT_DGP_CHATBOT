# 🎓 NIT Durgapur Academic Chatbot

A question-answering chatbot over NIT Durgapur's academic regulations. Ask it about
grading, attendance, branch change, graduation requirements, exam rules, and more —
it answers only from the official rule books and links the relevant appendix forms.

**Live demo:** <!-- paste your Streamlit link here once deployed -->

It covers **two separate rule books** — Undergraduate (UG) and Postgraduate (PG).
You pick one at the start of a conversation and can switch any time. The two are kept
in completely separate stores, because they reuse the same clause and appendix numbers
for different rules, and mixing them would produce wrong answers.

---

## What it does

- Answers grounded in the regulations only — it does not make up figures, dates, or
  procedures, and says so when something isn't in the rule book.
- Serves UG and PG separately, with a button to switch mid-conversation.
- Offers a download button for the relevant appendix form (internship declaration,
  admission cancellation, fee refund, etc.) when an answer mentions one.

## How it works

The pipeline, in order:

1. **Parse** — the PDF regulations are converted to clean markdown (LlamaParse).
2. **Chunk** — split section-aware, so a clause stays with its heading and tables stay intact.
3. **Embed + store** — chunks are embedded with `BAAI/bge-small-en-v1.5` into a ChromaDB
   vector store (one per program).
4. **Retrieve** — a query pulls candidate chunks by similarity, a cross-encoder
   (`ms-marco-MiniLM-L-6-v2`) re-ranks them for precision, and each hit pulls in the rest
   of its section so a clause reads top-to-bottom.
5. **Answer** — the retrieved context goes to a Qwen model via Groq, with a prompt that
   keeps answers grounded and refuses out-of-scope questions.

## Run it yourself

### Option A — Google Colab (no setup)
1. Open `NIT_DGP_Chatbot_Colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. In the 🔑 **Secrets** panel, add `GROQ_API_KEY` (free from [console.groq.com](https://console.groq.com)).
3. Run the cells top to bottom. The last cell prints a public link to the app.

### Option B — Local
```bash
git clone https://github.com/reetu58/NIT_DGP_CHATBOT.git
cd NIT_DGP_CHATBOT
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
streamlit run app.py
```

Both options use the prebuilt vector stores in the repo, so you only need a Groq key.
A LlamaParse key is required **only** if you rebuild the stores from the PDFs
(`python ingest.py ug` / `python ingest.py pg`).

## Evaluation

The chatbot is tested against hand-built question sets with known correct answers pulled
straight from the regulations (`eval.json` for UG, `eval_pg.json` for PG). Each question
runs through the live pipeline and its answer is checked against the expected one.

```bash
python evaluate.py eval.json ug        # UG
python evaluate.py eval_pg.json pg      # PG
```

The question sets deliberately include direct table lookups, paraphrased questions,
out-of-scope questions (to confirm it refuses rather than inventing), and multi-turn
follow-ups.

## Repo layout

| File | What it is |
| --- | --- |
| `app.py` | Streamlit UI — the program picker, chat, and form-download buttons |
| `rag.py` | Retrieval + the RAG chain (no Streamlit, so it can be tested on its own) |
| `config.py` | Single source of truth for models, paths, and per-program settings |
| `ingest.py` | Builds a program's vector store from its PDF |
| `forms.py` | Appendix-form catalogue and detection |
| `evaluate.py` | Runs an eval set and scores it |
| `eval.json` / `eval_pg.json` | UG / PG evaluation question sets |
| `vector_store_ug/` `vector_store_pg/` | Prebuilt vector stores |

## Tech

Python · Streamlit · LangChain · ChromaDB · Sentence-Transformers · Groq (Qwen) · LlamaParse

## Limitations

- The knowledge is only as current as the PDFs it was built from. For fees, dates, and
  anything revised by notification, it points you to the Academic Section.
- The evaluation set is small and hand-built — good for catching regressions, not a
  statistical accuracy claim.
