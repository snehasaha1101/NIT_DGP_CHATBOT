"""
Appendix-form data and detection, shared by extract_forms.py and ingest.py.

Some appendices are blank PDF forms. For each program we:
  - extract them as standalone PDFs for download (extract_forms.py), and
  - replace their blank pages with describable summaries in the vector store
    (ingest.py),
both of which need to IDENTIFY a form page. That identification lives here once,
so the two scripts can never disagree about what counts as which form.

UG and PG REUSE the same appendix numbers for DIFFERENT forms (e.g. UG Appendix-X
is the internship form, PG Appendix-X is the off-campus conversion form), so the
summaries and the detector are keyed by program.
"""

import re

# Summaries that replace the blank form pages in the vector store, so the bot can
# talk about each form (and trigger its download button) without trying to read
# empty form fields. Keyed by program, then by appendix.
FORM_SUMMARIES = {
    "ug": {
        "Appendix-X": "Appendix-X is the Declaration form for students going for the internship in final year UG courses. It is a blank form required for internships. If a student asks for this form or about the final year internship declaration, inform them that you can provide the PDF form of Appendix-X, and do not try to quote the form fields.",
        "Appendix-XI": "Appendix-XI is the Cancellation of Admission (UG) form. It is a blank form required for students who wish to cancel their admission. If a student asks for this form or about cancelling admission, inform them that you can provide the PDF form of Appendix-XI.",
        "Appendix-XII": "Appendix-XII is the Form for Refund of Admission Fees / Caution Money (UG). It is a blank form required for fee refund. If a student asks for this form or about getting a refund of fees/caution money, inform them that you can provide the PDF form of Appendix-XII.",
    },
    "pg": {
        "Appendix-X": "Appendix-X is the form for Conversion from M.Tech. (Regular) to M.Tech. (Off-Campus) mode, submitted after successful completion of the first year of the programme to complete the final-year project in off-campus mode. It is a blank form. If a student asks for this form or about converting to off-campus mode, inform them that you can provide the PDF form of Appendix-X, and do not try to quote the form fields.",
        "Appendix-XI": "Appendix-XI is the Declaration form for students going for the internship in final year PG courses. It is a blank form. If a student asks for this form or about the final-year internship declaration, inform them that you can provide the PDF form of Appendix-XI.",
        "Appendix-XII": "Appendix-XII is the Cancellation of Admission (PG) form. It is a blank form required for students who wish to cancel their admission. If a student asks for this form or about cancelling admission, inform them that you can provide the PDF form of Appendix-XII.",
        "Appendix-XIII": "Appendix-XIII is the Form for Refund of Admission Fees / Caution Money (PG). It is a blank form required for a refund of fees or caution money. If a student asks for this form or about getting a refund, inform them that you can provide the PDF form of Appendix-XIII.",
    },
}

_APPENDIX_HEADING = re.compile(r"appendix\s*-?\s*([ivx]+)")   # matched on lowercased text


def classify_appendix_form(text, program="ug"):
    """Identify an appendix FORM page by its CONTENT, not its page position, so the
    pipeline survives the regulations being re-paginated. Returns the appendix name
    (e.g. 'Appendix-XII') if the page is a known form for `program`, else None.

    Guard: a real form page always has a 'Signature of the ...' line, which clause
    prose and plain text appendices (committee lists, duties) never do - this stops
    a mere textual mention of an appendix from being mistaken for the form itself.
    """
    t = (text or "").lower()
    if "signature of the" not in t:
        return None

    if program == "ug":
        if "declaration form" in t and "internship" in t:
            return "Appendix-X"
        # Refund checked BEFORE cancellation: the UG Appendix-XII page also
        # contains "cancellation of admission", so the reverse order misclassifies.
        if "refund of admission" in t:
            return "Appendix-XII"
        if "cancellation of admission" in t:
            return "Appendix-XI"
        return None

    # PG (and any future program): identify by the Appendix-<Roman> heading printed
    # on the form page, accepting it only if it is a known PG form. This keeps PG
    # robust even though it has four forms reusing UG's numbers for other purposes.
    m = _APPENDIX_HEADING.search(t)
    if not m:
        return None
    name = "Appendix-" + m.group(1).upper()
    return name if name in FORM_SUMMARIES.get(program, {}) else None
