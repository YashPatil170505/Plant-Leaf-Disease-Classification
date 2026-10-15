"""
run_eval.py - Evaluate the RAG assistant end to end

Gemini is only used to generate the answers under test. All scoring is local:
  1. Retrieval quality   - Recall@k / MRR, global search vs. class-filtered search (no LLM)
  2. Tool use            - did the agent call the right tool before answering?
  3. Citation correctness- cited ids exist (validity) + cited passage entails the sentence (NLI)
  4. Hallucination rate  - answer sentences not entailed by any retrieved passage (NLI)
  5. Answer relevance    - cross-encoder relevance between question and answer
  6. Abstention          - refuses out-of-scope questions, answers in-scope ones
  7. Confidence gating   - image cases get the expected confident / uncertain / healthy status
  8. Latency             - end-to-end, LLM, local tools (p50 / p95)

Answers are cached in outputs/eval/answers.jsonl, so re-running only calls Gemini
for questions that have no cached answer (use --fresh to regenerate).

Usage:
    python -m evaluation.run_eval --retrieval-only
    python -m evaluation.run_eval --limit 20 --sleep 5
"""

import os
import json
import time
import argparse
import numpy as np

from rag.knowledge import CITE_TAG
from rag.retriever import Retriever
from rag.agent import PlantAssistant, gate
from model import split_label

QUESTIONS = "evaluation/questions.json"
OUT_DIR   = "outputs/eval"
ANSWERS   = os.path.join(OUT_DIR, "answers.jsonl")
KS        = [1, 3, 4]


# ── 1. retrieval (local only) ─────────────────────────────────────────
def eval_retrieval(retriever, questions):
    items = [q for q in questions if q.get("gold_class") and q.get("gold_section")]
    out = {}
    for mode in ("global", "filtered"):
        ranks = []
        for q in items:
            gold = f"{q['gold_class']}#{q['gold_section']}"
            hits = retriever.search(q["question"], k=10,
                                    class_filter=[q["gold_class"]] if mode == "filtered" else None)
            ids = [h["id"] for h in hits]
            ranks.append(ids.index(gold) + 1 if gold in ids else None)
        out[mode] = {f"recall@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in KS}
        out[mode]["mrr"] = float(np.mean([1 / r if r else 0 for r in ranks]))
        out[mode]["n"] = len(ranks)
    return out


# ── 2. generate answers with the real agent ───────────────────────────
def synthetic_prediction(class_name, confidence=0.96):
    plant, disease = split_label(class_name)
    return {"image": "(eval)", "top_k": [{"class_name": class_name, "plant": plant, "disease": disease,
                                         "is_healthy": disease.lower() == "healthy", "confidence": confidence}]}


def run_question(q, retriever, classifier):
    bot = PlantAssistant(retriever=retriever, classifier=classifier, verbose=False)
    if q.get("image"):
        msg = f"I uploaded a leaf image: {q['image']}. {q['question']}"
    elif q.get("context_class"):
        # Simulates the state after a confident image prediction, without needing an image
        bot.prediction = synthetic_prediction(q["context_class"])
        bot.gating = gate(bot.prediction)
        plant, disease = split_label(q["context_class"])
        msg = f"The leaf classifier diagnosed my plant as {plant} - {disease} (96% confidence). {q['question']}"
    else:
        msg = q["question"]

    res = bot.ask(msg)
    retrieved = {}
    for t in res["tool_calls"]:
        if t["tool"] == "retrieve_disease_info" and "passages" in t["result"]:
            for p in t["result"]["passages"]:
                retrieved[p["id"]] = bot.passages[p["id"]]
    return {
        "id": q["id"], "message": msg, "answer": res["answer"], "model": res["model"],
        "tools_called": [t["tool"] for t in res["tool_calls"]],
        "retrieved": {pid: {"id": c["id"], "text": CITE_TAG.sub("", c["text"])} for pid, c in retrieved.items()},
        "cited": list(res["citations"]), "invalid_citations": res["invalid_citations"],
        "gating_status": res["gating"]["status"] if res["gating"] else None,
        "latency_ms": res["latency_ms"],
    }


def load_cache():
    if not os.path.exists(ANSWERS):
        return {}
    with open(ANSWERS, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


# ── 3. aggregate ──────────────────────────────────────────────────────
def pct(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def aggregate(questions, rows):
    by_id = {q["id"]: q for q in questions}
    ans   = [r for r in rows if by_id[r["id"]]["answerable"]]
    unans = [r for r in rows if not by_id[r["id"]]["answerable"]]
    imgs  = [r for r in rows if by_id[r["id"]].get("expected_status")]

    def expected_tool(r):
        return "predict_disease" if by_id[r["id"]].get("image") else "retrieve_disease_info"

    claims   = sum(r["score"]["n_claims"] for r in ans)
    unsup    = sum(r["score"]["n_unsupported"] for r in ans)
    cited_c  = sum(r["score"]["n_cited_claims"] for r in ans)
    cited_ok = sum(r["score"]["n_cited_supported"] for r in ans)
    n_cites  = sum(len(r["cited"]) + len(r["invalid_citations"]) for r in rows)
    gold_hit = [f"{by_id[r['id']]['gold_class']}#{by_id[r['id']]['gold_section']}" in
                {p["id"] for p in r["retrieved"].values()}
                for r in ans if by_id[r["id"]].get("gold_section")]
    lat = lambda key: [r["latency_ms"][key] for r in rows]

    return {
        "n_questions": len(rows), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "tool_use": {
            "correct_tool_called": pct([expected_tool(r) in r["tools_called"] for r in rows]),
            "answered_without_any_tool": pct([not r["tools_called"] for r in rows]),
        },
        "retrieval_in_loop": {"gold_passage_retrieved": pct(gold_hit)},
        "citations": {
            "answers_with_citations": pct([bool(r["cited"]) for r in ans if not r["score"]["abstained"]]),
            "citation_validity": (1 - sum(len(r["invalid_citations"]) for r in rows) / n_cites) if n_cites else None,
            "cited_claims_supported_by_cited_passage": cited_ok / cited_c if cited_c else None,
        },
        "hallucination": {
            "claim_level_unsupported_rate": unsup / claims if claims else None,
            "answers_with_any_unsupported_claim": pct([r["score"]["n_unsupported"] > 0 for r in ans]),
        },
        "answer_relevance_mean": pct([r["score"]["relevance"] for r in ans]),
        "abstention": {
            "correct_refusal_on_unanswerable": pct([r["score"]["abstained"] for r in unans]),
            "false_refusal_on_answerable": pct([r["score"]["abstained"] for r in ans]),
        },
        "confidence_gating": {
            "status_matches_expected": pct([r["gating_status"] == by_id[r["id"]]["expected_status"] for r in imgs]),
            "n": len(imgs),
        },
        "latency_ms": {k: {"p50": float(np.percentile(lat(k), 50)), "p95": float(np.percentile(lat(k), 95))}
                       for k in ("total", "llm", "tools")},
    }


def write_report(retrieval, summary, rows, path):
    f = lambda x: "-" if x is None else f"{x*100:.1f}%"
    L = ["# RAG Assistant Evaluation", "",
         "All scores are computed locally (NLI + cross-encoder models); Gemini only generated the answers.", "",
         "## Retrieval (local, no LLM)", "", "| Search | Recall@1 | Recall@3 | Recall@4 | MRR | n |", "|---|---|---|---|---|---|"]
    for mode, m in retrieval.items():
        L.append(f"| {mode} | {f(m['recall@1'])} | {f(m['recall@3'])} | {f(m['recall@4'])} | {m['mrr']:.3f} | {m['n']} |")
    if summary:
        s = summary
        L += ["", f"## End-to-end ({s['n_questions']} questions: {s['n_answerable']} answerable, {s['n_unanswerable']} out-of-scope)", "",
              "| Metric | Value |", "|---|---|",
              f"| Correct tool called | {f(s['tool_use']['correct_tool_called'])} |",
              f"| Answered without any tool | {f(s['tool_use']['answered_without_any_tool'])} |",
              f"| Gold passage retrieved (in loop) | {f(s['retrieval_in_loop']['gold_passage_retrieved'])} |",
              f"| Answers with citations | {f(s['citations']['answers_with_citations'])} |",
              f"| Citation validity (id exists) | {f(s['citations']['citation_validity'])} |",
              f"| Cited claims supported by cited passage (NLI) | {f(s['citations']['cited_claims_supported_by_cited_passage'])} |",
              f"| Unsupported claims (hallucination, claim-level) | {f(s['hallucination']['claim_level_unsupported_rate'])} |",
              f"| Answers with >=1 unsupported claim | {f(s['hallucination']['answers_with_any_unsupported_claim'])} |",
              f"| Answer relevance (mean, 0-1) | {s['answer_relevance_mean']:.3f} |" if s["answer_relevance_mean"] is not None else "| Answer relevance | - |",
              f"| Correct refusal on out-of-scope | {f(s['abstention']['correct_refusal_on_unanswerable'])} |",
              f"| False refusal on in-scope | {f(s['abstention']['false_refusal_on_answerable'])} |",
              f"| Confidence-gating status correct (image cases) | {f(s['confidence_gating']['status_matches_expected'])} (n={s['confidence_gating']['n']}) |",
              f"| Latency total p50 / p95 | {s['latency_ms']['total']['p50']:.0f} / {s['latency_ms']['total']['p95']:.0f} ms |",
              f"| Latency LLM p50 / p95 | {s['latency_ms']['llm']['p50']:.0f} / {s['latency_ms']['llm']['p95']:.0f} ms |",
              f"| Latency local tools p50 / p95 | {s['latency_ms']['tools']['p50']:.0f} / {s['latency_ms']['tools']['p95']:.0f} ms |",
              "", "NLI scoring is conservative: correct paraphrases are sometimes judged 'neutral', so the hallucination "
              "rate is an upper bound. Flagged sentences are listed below for manual review.", "",
              "## Flagged sentences (not entailed by any retrieved passage)", ""]
        for r in rows:
            for sen in r["score"]["sentences"]:
                if not sen["supported_by_any"]:
                    L.append(f"- **{r['id']}** (p={sen['p_any']}): {sen['sentence']}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true", help="Skip Gemini; evaluate retrieval only")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    ap.add_argument("--sleep", type=float, default=4.0, help="Seconds between Gemini questions (free-tier rate limits)")
    ap.add_argument("--fresh", action="store_true", help="Ignore cached answers")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(QUESTIONS, encoding="utf-8") as fh:
        questions = json.load(fh)
    if args.limit:
        questions = questions[:args.limit]

    retriever = Retriever()
    print("Retrieval evaluation (local) ...")
    retrieval = eval_retrieval(retriever, questions)
    for mode, m in retrieval.items():
        print(f"  {mode:8s}  R@1 {m['recall@1']:.3f}  R@3 {m['recall@3']:.3f}  R@4 {m['recall@4']:.3f}  MRR {m['mrr']:.3f}")

    summary, rows = None, []
    if not args.retrieval_only:
        cache = {} if args.fresh else load_cache()
        classifier = None
        if any(q.get("image") for q in questions):
            from model import LeafClassifier
            classifier = LeafClassifier()

        mode = "w" if args.fresh else "a"
        with open(ANSWERS, mode, encoding="utf-8") as fh:
            for i, q in enumerate(questions, 1):
                if q["id"] in cache:
                    rows.append(cache[q["id"]]); continue
                print(f"  [{i}/{len(questions)}] {q['id']}: {q['question'][:70]}")
                try:
                    r = run_question(q, retriever, classifier)
                except Exception as e:
                    print(f"    FAILED: {e}"); continue
                fh.write(json.dumps(r) + "\n"); fh.flush()
                rows.append(r)
                time.sleep(args.sleep)

        print("\nScoring locally (NLI + relevance cross-encoders) ...")
        from evaluation.metrics import Scorer
        scorer = Scorer()
        by_id = {q["id"]: q for q in questions}
        for r in rows:
            r["score"] = scorer.score_answer(by_id[r["id"]]["question"], r["answer"],
                                             {k: v["text"] for k, v in r["retrieved"].items()},
                                             {k: r["retrieved"][k]["text"] for k in r["cited"] if k in r["retrieved"]})
        summary = aggregate(questions, rows)
        with open(os.path.join(OUT_DIR, "scored.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump({"retrieval": retrieval, "end_to_end": summary}, fh, indent=2)
    write_report(retrieval, summary, rows, os.path.join(OUT_DIR, "report.md"))
    if summary:
        print(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_DIR}/results.json and {OUT_DIR}/report.md")
