"""
One-off diagnostic: parse the PDF, cache the markdown to parsed_output.md, and
print every line that mentions "appendix" EXACTLY as parsed (repr shows the real
dashes/spaces/case). Use this to fix the appendix heading regex in ingest.py.
"""

import sys
from dotenv import load_dotenv
import config

load_dotenv()

# Works whether or not you've migrated off the deprecated package.
try:
    from llama_cloud_services import LlamaParse
except ImportError:
    from llama_parse import LlamaParse

# Usage: python inspect_parse.py [ug|pg]   (defaults to ug)
_prog = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_PROGRAM
docs = LlamaParse(result_type="markdown", verbose=False).load_data(config.get_program(_prog)["pdf"])
full = "\n".join(d.text for d in docs)

with open("parsed_output.md", "w", encoding="utf-8") as f:
    f.write(full)
print(f"Saved parsed_output.md ({len(full)} chars, {len(docs)} pages)\n")

print("--- every line containing 'appendix' (repr = exact characters) ---")
for line in full.splitlines():
    if "appendix" in line.lower():
        print(repr(line))

# Also surface the specific repealed numbers we worried about earlier.
print("\n--- lines containing the repealed '8.75' (should be struck through) ---")
for line in full.splitlines():
    if "8.75" in line:
        print(repr(line))
