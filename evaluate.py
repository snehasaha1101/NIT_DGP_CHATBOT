"""
Run the held-out evaluation set against the RAG pipeline and print a score.

Usage:
    python evaluate.py              # reads eval.json
    python evaluate.py myset.json   # reads a named file

Scoring is answer-level, matching your rubric: a case PASSES if the final answer
contains every "expected" string and none of the "forbidden" strings. Matching
is case-insensitive and word-boundary aware, so expecting "8" does NOT match the
"8" inside "80" (the exact false-positive your eval notes warned about).

Expected JSON: a list of cases. Field aliases are accepted so you can reuse your
existing wording.
    [
      {
        "id": "APP-01",
        "turns": ["I scored 72 in a theory subject. What grade and grade points?"],
        "expected": ["B", "8"],
        "forbidden": ["A", "9", "C"]
      },
      {
        "id": "MT-01",
        "turns": [
          "What SGPA do I need to change branch as a General student?",
          "And for SC/ST students?"
        ],
        "expected": ["8.50", "both"],
        "forbidden": ["9.00 for SC"]
      }
    ]
  - "turns" (or "questions"/"user"): one string, or a list for multi-turn. For a
    multi-turn case the FINAL answer is scored.
  - "expected" (or "expected_facts"): list of strings that must appear.
  - "forbidden" (or "must_not_contain"): list of strings that must not appear.
"""

import json
import re
import sys
import time

from langchain_core.messages import HumanMessage, AIMessage
from config import DEFAULT_PROGRAM
from rag import build_rag_chain

# --- Rate limiting (Groq free tier is 7000 input tokens/minute) --------------
SLEEP_BETWEEN = 25     # seconds to wait between cases; lower it on a paid tier
MAX_RETRIES = 4        # retries when a call is rate-limited


def results_path_for(program):
    """Per-program results file, so a UG run and a PG run never overwrite each other."""
    return f"eval_results_{program}.json"


def _get(case, *keys, default=None):
    for k in keys:
        if k in case:
            return case[k]
    return default


def _as_list(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def contains(answer, term):
    """Case-insensitive, word-boundary match so '8' doesn't match '80'."""
    term = str(term).strip()
    if not term:
        return True
    return re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", answer, re.IGNORECASE) is not None


class DailyLimitReached(Exception):
    """Groq's per-day token cap (TPD) is exhausted. Retrying cannot help - the
    budget only resets at the daily rollover - so we stop and let --resume finish
    the remaining cases on the next day's budget."""


def run_case(chain, turns):
    """Run all turns, carrying history, and return the FINAL answer."""
    history, answer = [], ""
    for q in turns:
        for attempt in range(MAX_RETRIES):
            try:
                answer = chain.invoke({"input": q, "chat_history": history})
                break
            except Exception as e:
                msg = str(e).lower()
                # A daily-cap 429 names "per day" / "tpd". No amount of waiting
                # within a run refills it, so bail straight away.
                if "per day" in msg or "tpd" in msg:
                    raise DailyLimitReached(str(e))
                rate_limited = "rate" in msg or "413" in msg or "429" in msg or "too large" in msg
                if rate_limited and attempt < MAX_RETRIES - 1:
                    wait = SLEEP_BETWEEN * (attempt + 2)
                    print(f"      rate limited (per-minute) - waiting {wait}s and retrying...")
                    time.sleep(wait)
                else:
                    raise
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=answer))
    return answer


def _load_prior_results(path):
    """Read any results already on disk, keyed by case id. Used so a run that
    halted part-way can be resumed instead of starting over."""
    try:
        with open(path, encoding="utf-8") as f:
            return {r["id"]: r for r in json.load(f) if "id" in r}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    resume = "--resume" in flags          # skip cases already saved from a prior run

    path = args[0] if args else "eval.json"
    program = args[1] if len(args) > 1 else DEFAULT_PROGRAM   # which rule book to test
    results_path = results_path_for(program)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # The set may be a bare list, or an object with the cases under "cases"/"tests".
    cases = data if isinstance(data, list) else data.get("cases") or data.get("tests") or []
    if not cases:
        print("No cases found. Expected a JSON list, or an object with a 'cases' array.")
        return

    # Keyed by id so results survive a mid-run halt and keep a stable order.
    done = _load_prior_results(results_path) if resume else {}
    if done:
        print(f"Resuming: {len(done)} case(s) already have saved answers; those are skipped.")

    print(f"Loading {program} pipeline and running {len(cases)} cases from {path}...\n")
    chain = build_rag_chain(program)

    def save():
        # Write after every case so a halt (Ctrl+C, OOM, dropped shell) never
        # loses more than the single case in flight.
        ordered = [done[_get(c, "id", default=f"case-{i}")]
                   for i, c in enumerate(cases, 1)
                   if _get(c, "id", default=f"case-{i}") in done]
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)

    for i, case in enumerate(cases, 1):
        cid = _get(case, "id", default=f"case-{i}")
        if resume and cid in done:
            print(f"[{i}/{len(cases)}] {cid}  (skipped - already done)")
            continue

        turns = _as_list(_get(case, "turns", "questions", "user"))
        expected = _as_list(_get(case, "expected", "expected_facts"))
        forbidden = _as_list(_get(case, "forbidden", "must_not_contain", "must_not"))

        print(f"[{i}/{len(cases)}] {cid}")
        try:
            answer = run_case(chain, turns)
        except DailyLimitReached:
            print("\n  Groq daily token limit (200k TPD) reached - stopping here.")
            print(f"  {len(done)} case(s) are saved in {results_path}.")
            print(f"  Once the daily budget resets, run:  python evaluate.py {path} {program} --resume")
            break

        missing = [e for e in expected if not contains(answer, e)]
        hit_forbidden = [fb for fb in forbidden if contains(answer, fb)]
        ok = not missing and not hit_forbidden

        print(f"    {'PASS' if ok else 'FAIL'}")
        if missing:
            print(f"      missing expected : {missing}")
        if hit_forbidden:
            print(f"      contains forbidden: {hit_forbidden}")

        done[cid] = {
            "id": cid, "pass": ok,
            "missing_expected": missing, "forbidden_present": hit_forbidden,
            "answer": answer,
        }
        save()                 # persist progress immediately

        if i < len(cases):
            time.sleep(SLEEP_BETWEEN)

    scored = [done[_get(c, "id", default=f"case-{i}")]
              for i, c in enumerate(cases, 1)
              if _get(c, "id", default=f"case-{i}") in done]
    passed = sum(1 for r in scored if r["pass"])
    pct = 100 * passed / len(scored) if scored else 0
    print(f"\n==== {program.upper()} SCORE: {passed}/{len(scored)}  ({pct:.0f}%) ====")
    print(f"Per-case details (including full answers) written to {results_path}")


if __name__ == "__main__":
    main()
