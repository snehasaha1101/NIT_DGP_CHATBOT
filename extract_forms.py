"""
Extract the blank appendix FORM pages as standalone PDFs for download.

Form detection is shared with ingest.py via forms.classify_appendix_form, so the
two scripts can never disagree about which page is which. The detector is
program-aware (UG forms X-XII, PG forms X-XIII).

Usage:
    python extract_forms.py          # UG (default)
    python extract_forms.py pg       # PG
"""

import os
import sys
from pypdf import PdfReader, PdfWriter

import config
from forms import FORM_SUMMARIES, classify_appendix_form


def extract_forms(program=config.DEFAULT_PROGRAM):
    cfg = config.get_program(program)
    os.makedirs(cfg["forms_dir"], exist_ok=True)
    reader = PdfReader(cfg["pdf"])
    print(f"Extracting {cfg['label']} forms from {cfg['pdf']} into {cfg['forms_dir']}/ ...")

    found = {}
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        name = classify_appendix_form(text, program)
        if name and name not in found:
            found[name] = i
            writer = PdfWriter()
            writer.add_page(page)
            with open(os.path.join(cfg["forms_dir"], f"{name}.pdf"), "wb") as f:
                writer.write(f)
            print(f"Saved {name} from page index {i}")

    missing = [n for n in FORM_SUMMARIES.get(program, {}) if n not in found]
    if missing:
        print(f"WARNING: could not locate these form pages by content: {missing}")
        print("  -> The PDF layout may have changed; check the appendix pages manually.")
    else:
        print("All appendix forms extracted successfully!")


if __name__ == "__main__":
    prog = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_PROGRAM
    extract_forms(prog)
