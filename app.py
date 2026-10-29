"""
Streamlit UI for the NIT Durgapur Academic Chatbot.

This file is intentionally THIN: all retrieval and chain logic lives in rag.py
(so it can be tested without Streamlit). app.py only handles the interface -
page layout, the program-picker, chat history, and the appendix download buttons.

CONVERSATION FLOW
-----------------
Every conversation opens with a welcome message and two buttons - UG and PG -
that stay on screen for the whole chat. The student must pick one before the
chat input appears, and can switch at any time by clicking the other button; the
active rule book is highlighted. Switching does NOT wipe the visible chat.

Cross-contamination guard: each turn is tagged with the program it was asked
under, and when the chain answers it is only given the history from the CURRENT
program. So a PG follow-up is never rewritten using earlier UG turns (the two
rule books reuse clause/appendix numbers for different rules), even though both
sets of turns remain visible in the transcript.
"""

import os
import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# On Streamlit Community Cloud, API keys are set in the app's Secrets. Mirror them
# into the environment so config.py (which reads os.environ via load_dotenv) finds
# them. Harmless locally, where there is no secrets file and this just no-ops.
for _k in ("GROQ_API_KEY", "LLAMA_CLOUD_API_KEY"):
    try:
        if _k not in os.environ and _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
    except Exception:
        pass

import config
from rag import build_rag_chain

# --- Page Config & Styling ---
st.set_page_config(page_title="NIT DGP Academic Chatbot", page_icon="🎓", layout="centered")
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stChatFloatingInputContainer { bottom: 20px; }
    h1 { color: #1e3a8a; text-align: center; }
</style>
""", unsafe_allow_html=True)
st.title("🎓 NIT Durgapur Academic Chatbot")

WELCOME = (
    "Hello! 👋 I'm the **NIT Durgapur Academic Chatbot**.\n\n"
    "I can answer questions about the institute's academic regulations — "
    "grading, attendance, branch change, graduation requirements, exam rules, "
    "and the downloadable appendix forms.\n\n"
    "Pick which rule book you'd like to ask about — you can switch any time:"
)


# --- Session State ---
# chat_history is a list of records: {"role", "content", "program"}. Tagging each
# turn with its program lets the transcript stay visible across a switch while the
# chain only ever sees the current program's turns.
if "program" not in st.session_state:
    st.session_state.program = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- Sidebar: full reset ---
if st.sidebar.button("🔄 Clear conversation"):
    st.session_state.program = None
    st.session_state.chat_history = []
    st.rerun()


@st.cache_resource
def load_chain(program):
    # Cached per program, so UG and PG chains are each built once and reused.
    return build_rag_chain(program)


# Appendix download buttons, keyed off the appendices named in an answer. Each
# program has its own form set (UG: X-XII; PG: X-XIII), stored in that program's
# forms_dir. (roman, filename, label, key_prefix) per form.
_APPENDIX_BUTTONS = {
    "ug": [
        ("x",   "Appendix-X.pdf",   "📥 Download Internship Declaration Form (Appendix-X)",  "btn_ug_x"),
        ("xi",  "Appendix-XI.pdf",  "📥 Download Cancellation of Admission (Appendix-XI)",   "btn_ug_xi"),
        ("xii", "Appendix-XII.pdf", "📥 Download Refund of Admission Fees (Appendix-XII)",    "btn_ug_xii"),
    ],
    "pg": [
        ("x",    "Appendix-X.pdf",    "📥 Download Conversion to Off-Campus Form (Appendix-X)",            "btn_pg_x"),
        ("xi",   "Appendix-XI.pdf",   "📥 Download Internship Declaration Form (Appendix-XI)",             "btn_pg_xi"),
        ("xii",  "Appendix-XII.pdf",  "📥 Download Cancellation of Admission (Appendix-XII)",              "btn_pg_xii"),
        ("xiii", "Appendix-XIII.pdf", "📥 Download Refund of Admission Fees / Caution Money (Appendix-XIII)", "btn_pg_xiii"),
    ],
}


def render_appendix_buttons(text, msg_index, program):
    """Show a download button for each appendix FORM the answer names. Matching is
    boundary-aware so 'Appendix-X' doesn't also fire on 'Appendix-XIII' - the roman
    must not be followed by another roman letter."""
    forms_dir = config.PROGRAMS[program]["forms_dir"]
    low = text.lower()
    for roman, filename, label, key_prefix in _APPENDIX_BUTTONS.get(program, []):
        if re.search(rf"appendix[\s\-]*{roman}(?![ivxl])", low):
            path = os.path.join(forms_dir, filename)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    st.download_button(label, f, file_name=filename, key=f"{key_prefix}_{msg_index}")


def current_program_history(records, program):
    """LangChain messages for ONLY the current program's turns, so a follow-up is
    never contextualized against the other rule book's answers."""
    msgs = []
    for r in records:
        if r["program"] != program:
            continue
        msgs.append(HumanMessage(content=r["content"]) if r["role"] == "user"
                    else AIMessage(content=r["content"]))
    return msgs


# --- Welcome message + persistent program picker ---
# Shown on every run. Both buttons stay visible for the whole conversation; the
# active rule book is a 'primary' button, the other 'secondary'. Clicking the
# inactive one switches program WITHOUT clearing the visible transcript.
with st.chat_message("assistant"):
    st.write(WELCOME)
    col1, col2 = st.columns(2)
    ug_type = "primary" if st.session_state.program == "ug" else "secondary"
    pg_type = "primary" if st.session_state.program == "pg" else "secondary"
    if col1.button("📘 UG Regulations", key="pick_ug", use_container_width=True, type=ug_type):
        if st.session_state.program != "ug":
            st.session_state.program = "ug"
            st.rerun()
    if col2.button("📗 PG Regulations", key="pick_pg", use_container_width=True, type=pg_type):
        if st.session_state.program != "pg":
            st.session_state.program = "pg"
            st.rerun()
    if st.session_state.program:
        st.caption(
            f"Answering from the **{config.PROGRAMS[st.session_state.program]['label']}**. "
            "Click the other button any time to switch."
        )

# Nothing more until the student has picked a rule book.
if st.session_state.program is None:
    st.stop()

program = st.session_state.program

# --- Load that program's chain (cached per program) ---
try:
    with st.spinner("Loading models and database (this can take a minute on the first run)..."):
        rag_chain = load_chain(program)
except Exception as e:
    st.error(f"⚠️ Error loading pipeline: {str(e)}")
    st.stop()


# --- Render the full transcript (both programs' turns stay visible) ---
for i, r in enumerate(st.session_state.chat_history):
    if r["role"] == "user":
        st.chat_message("user").write(r["content"])
    else:
        with st.chat_message("assistant"):
            st.write(r["content"])
            render_appendix_buttons(r["content"], i, r["program"])

# --- Handle new input ---
if user_input := st.chat_input(f"Ask about the {config.PROGRAMS[program]['label']}..."):
    st.chat_message("user").write(user_input)
    with st.spinner("Searching regulations..."):
        answer = rag_chain.invoke({
            "input": user_input,
            "chat_history": current_program_history(st.session_state.chat_history, program),
        })
        with st.chat_message("assistant"):
            current_index = len(st.session_state.chat_history)
            st.write(answer)
            render_appendix_buttons(answer, current_index, program)
        st.session_state.chat_history.append({"role": "user", "content": user_input, "program": program})
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "program": program})
