"""
Turn an eval results file into a readable HTML report for manual grading.

Usage:
    python view_results.py            # UG: eval.json + eval_results_ug.json -> results_ug.html
    python view_results.py pg         # PG: eval_pg.json + eval_results_pg.json -> results_pg.html
Then open the results_<program>.html file in a browser.

It joins each result with its case (question, expected, forbidden, grading note)
so you can eyeball whether a FAIL is a real miss or just a string-match collision
(e.g. the answer says "Fail grade" and tripped a must_not of "fail").
"""

import json
import html
import sys

# Which program's files to read. UG uses eval.json; PG uses eval_pg.json.
_PROGRAM = sys.argv[1] if len(sys.argv) > 1 else "ug"
_CASES_FILE = "eval.json" if _PROGRAM == "ug" else f"eval_{_PROGRAM}.json"
_RESULTS_FILE = f"eval_results_{_PROGRAM}.json"
_OUT_FILE = f"results_{_PROGRAM}.html"


def _get(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _as_list(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


# Load the cases (questions + criteria) and the run results (answers).
with open(_CASES_FILE, encoding="utf-8") as f:
    data = json.load(f)
cases = data if isinstance(data, list) else data.get("cases", [])
by_id = {_get(c, "id", default=str(i)): c for i, c in enumerate(cases)}

with open(_RESULTS_FILE, encoding="utf-8") as f:
    results = json.load(f)

n_pass = sum(1 for r in results if r.get("pass"))
rows = []
for r in results:
    cid = r.get("id", "?")
    case = by_id.get(cid, {})
    turns = _as_list(_get(case, "turns", "questions", "user"))
    expected = _as_list(_get(case, "expected_facts", "expected"))
    forbidden = _as_list(_get(case, "must_not", "forbidden", "must_not_contain"))
    note = _get(case, "grading_note", default="")
    ok = r.get("pass")
    color = "#1b5e20" if ok else "#b71c1c"
    bg = "#e8f5e9" if ok else "#ffebee"

    q = "<br>".join(f"<b>Q{i+1}:</b> {html.escape(t)}" for i, t in enumerate(turns))
    rows.append(f"""
    <div style="border-left:6px solid {color};background:{bg};margin:14px 0;padding:12px 16px;border-radius:6px">
      <div style="font-size:16px"><b>{html.escape(cid)}</b>
          <span style="color:{color};font-weight:700">&nbsp;{'PASS' if ok else 'FAIL'}</span></div>
      <div style="margin:6px 0">{q}</div>
      <div style="font-size:13px;color:#333">
        <b>Expected:</b> {html.escape(", ".join(expected)) or '—'} &nbsp;|&nbsp;
        <b>Must not:</b> {html.escape(", ".join(forbidden)) or '—'}
      </div>
      <div style="font-size:13px;color:#b71c1c">
        {('<b>Missing expected:</b> ' + html.escape(", ".join(r.get("missing_expected", [])))) if r.get("missing_expected") else ''}
        {('&nbsp; <b>Hit forbidden:</b> ' + html.escape(", ".join(r.get("forbidden_present", [])))) if r.get("forbidden_present") else ''}
      </div>
      <details style="margin-top:8px"><summary style="cursor:pointer;color:#1a237e">Model answer</summary>
        <pre style="white-space:pre-wrap;font-family:inherit;background:#fff;padding:10px;border-radius:4px">{html.escape(r.get("answer", ""))}</pre>
      </details>
      <div style="font-size:12px;color:#666;margin-top:6px"><i>{html.escape(note)}</i></div>
    </div>""")

report = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Eval results</title></head>
<body style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 16px">
<h1>Evaluation results — {_PROGRAM.upper()}</h1>
<p style="font-size:18px">Mechanical score: <b>{n_pass}/{len(results)}</b>
 &nbsp;<span style="color:#666;font-size:14px">(string-match; read the FAILs — some are correct answers that tripped a must_not term)</span></p>
{''.join(rows)}
</body></html>"""

with open(_OUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Wrote {_OUT_FILE}  ({n_pass}/{len(results)} mechanical pass). Open it in a browser.")
