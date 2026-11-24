"""Run every question in eval/questions.json through both pipelines and score them.

    python -m eval.run_eval            # uses OPENAI_* from the environment
    python -m eval.run_eval --env path/to/.env

Scores per pipeline:
  retrieval  the evidence text is inside one of the retrieved excerpts
  answer     the answer contains every required fact (regex), or refuses when it should
  refusal    out-of-scope questions refused, in-scope questions not refused
Writes eval/results.json, which the demo's Eval tab shows.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SOFT_REFUSAL = re.compile(
    r"couldn't find|could not find|can't find|cannot find|unable to find|no information|don't have (any )?information|"
    r"(isn't|is not|aren't|are not|wasn't|was not|not) (mentioned|covered|provided|included|specified|available in the)|"
    r"(doesn't|does not|do not|don't) (mention|say|include|provide|contain|cover|specify)|not (in|part of) the (context|help)", re.I)


def load_env(path: str) -> None:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def refused(answer: str) -> bool:
    return bool(SOFT_REFUSAL.search(answer))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env")
    ap.add_argument("--only", help="comma-separated question ids")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between questions (free-tier rate limits)")
    args = ap.parse_args()
    if args.env:
        load_env(args.env)
    from rag.pipeline import RagService

    svc = RagService()
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    if args.only:
        questions = [q for q in questions if q["id"] in args.only.split(",")]
    rows = []
    for q in questions:
        row = {"id": q["id"], "type": q["type"], "q": q["q"], "history": q.get("history", [])}
        for p in ("naive", "fixed"):
            r = svc.ask(q["q"], q.get("history", []), p)
            texts = [s["text"] for s in r["sources"]]
            hit = not q["evidence"] or any(all(e.lower() in t.lower() for e in q["evidence"]) for t in texts)
            ref = refused(r["answer"])
            if q.get("refuse"):
                ok = ref
            else:
                ok = not ref and all(re.search(m, r["answer"], re.I) for m in q["must"])
            row[p] = {"answer": r["answer"], "ok": ok, "retrieval_hit": hit, "refused": ref,
                      "grounded": all(c["ok"] for c in r["checks"]), "attempts": r["attempts"],
                      "standalone": r["standalone_question"], "ms": r["timings"]["total_ms"],
                      "top": [f"{s['title']} > {s['section']}" for s in r["sources"][:3]]}
            print(f"{p:5} {'PASS' if ok else 'FAIL'} {'R' if hit else '-'} {q['id']:24} {r['answer'][:110]!r}", flush=True)
            time.sleep(args.pause)
        rows.append(row)

    summary = {}
    for p in ("naive", "fixed"):
        by_type = defaultdict(lambda: [0, 0])
        for row in rows:
            by_type[row["type"]][0] += row[p]["ok"]
            by_type[row["type"]][1] += 1
        in_scope = [r for r in rows if r["type"] != "out_of_scope"]
        oos = [r for r in rows if r["type"] == "out_of_scope"]
        summary[p] = {
            "answers_correct": [sum(r[p]["ok"] for r in rows), len(rows)],
            "retrieval_hit": [sum(r[p]["retrieval_hit"] for r in in_scope), len(in_scope)],
            "out_of_scope_refused": [sum(r[p]["refused"] for r in oos), len(oos)],
            "in_scope_wrongly_refused": [sum(r[p]["refused"] for r in in_scope), len(in_scope)],
            "median_ms": sorted(r[p]["ms"] for r in rows)[len(rows) // 2],
            "by_type": {k: v for k, v in by_type.items()},
        }
    out = {"model": svc.model, "embeddings": svc.meta()["embeddings"], "ran_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "summary": summary, "rows": rows}
    if not args.only:
        (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for p in ("naive", "fixed"):
        print(p, json.dumps({k: v for k, v in summary[p].items() if k != "by_type"}))


if __name__ == "__main__":
    main()
