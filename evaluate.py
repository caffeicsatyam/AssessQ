"""
evaluate.py — AssessQ RAG Pipeline Evaluation with LangSmith + LLM-as-Judge
============================================================================

What this does
──────────────
1. Uploads your evaluation dataset to LangSmith as a named Dataset.
2. Runs your RAG pipeline as the "target" function — every retrieval is traced.
3. Applies four evaluators per query:
     • hit_rate@K      — deterministic: any expected name in top-K?
     • mrr@K           — deterministic: reciprocal rank of first hit
     • ndcg@K          — deterministic: normalised DCG @ K
     • judge_relevance — LLM-as-Judge: semantic relevance via Groq
4. Pushes all scores to LangSmith UI — viewable at https://smith.langchain.com

Why LLM-as-Judge matters here
──────────────────────────────
Exact-match metrics fail when catalog name differs slightly from expected.
The judge scores semantic relevance, so "Core Java (Advanced Level) (New)"
scores 1.0 even if expected said "Java Advanced".

Setup
──────
pip install langsmith langchain-groq

.env must contain:
    LANGSMITH_API_KEY=ls__...
    LANGSMITH_PROJECT=assessq-eval
    GROQ_API_KEY=gsk_...

Usage
──────
  python evaluate.py                          # BM25 only, K=5, local only
  python evaluate.py --upload                 # same + push to LangSmith
  python evaluate.py --k 10 --upload          # K=10, push to LangSmith
  python evaluate.py --dense --rerank         # full pipeline
  python evaluate.py --compare                # all 4 configs
  python evaluate.py --verbose                # per-query breakdown
  python evaluate.py --experiment my-run-v2   # custom experiment name
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Set

from dotenv import load_dotenv

load_dotenv()

# ── env ───────────────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "assessq-eval")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")

os.environ.setdefault("LANGSMITH_API_KEY",     LANGSMITH_API_KEY)
os.environ.setdefault("LANGSMITH_PROJECT",     LANGSMITH_PROJECT)
os.environ.setdefault("LANGCHAIN_TRACING_V2",  "true")

from agent import AssessmentRecommender, load_catalog, DEFAULT_CATALOG_PATH, format_table


# ── deterministic metrics ─────────────────────────────────────────────────────

def _dcg(relevances: List[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def _hit_rate(results: List[dict], expected: Set[str], k: int) -> float:
    for r in results[:k]:
        if r["metadata"]["assessment_name"] in expected:
            return 1.0
    return 0.0


def _mrr(results: List[dict], expected: Set[str], k: int) -> float:
    for rank, r in enumerate(results[:k]):
        if r["metadata"]["assessment_name"] in expected:
            return 1.0 / (rank + 1)
    return 0.0


def _ndcg(results: List[dict], expected: Set[str], k: int) -> float:
    gains  = [1.0 if r["metadata"]["assessment_name"] in expected else 0.0
              for r in results[:k]]
    actual = _dcg(gains)
    ideal  = _dcg(sorted(gains, reverse=True))
    return actual / ideal if ideal > 0.0 else 0.0


# ── dataset helpers ───────────────────────────────────────────────────────────

def load_local_dataset(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def upload_dataset_to_langsmith(client, dataset: List[dict], dataset_name: str):
    existing = [d for d in client.list_datasets() if d.name == dataset_name]
    if existing:
        print(f"  Reusing existing LangSmith dataset: '{dataset_name}'")
        return existing[0]

    ls_dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="AssessQ RAG evaluation dataset",
    )
    examples = [
        {
            "inputs":  {"query": item["query"]},
            "outputs": {
                "expected_assessments": item["expected_assessments"],
                "difficulty":           item.get("difficulty", "unknown"),
                "notes":                item.get("notes", ""),
            },
        }
        for item in dataset
    ]
    client.create_examples(dataset_id=ls_dataset.id, examples=examples)
    print(f"  Uploaded {len(examples)} examples → LangSmith dataset '{dataset_name}'")
    return ls_dataset


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator for an SHL assessment recommendation system.

Job role / query:
{query}

Retrieved assessments (top {k}):
{table}

Expected assessment types / skills needed:
{expected}

Task:
Score how well the retrieved assessments match what this role actually needs.
Consider semantic relevance — a "Core Java Advanced" test is relevant even if
the exact catalog name differs slightly from the expected list.

Score 0.0 to 1.0:
  1.0 — All retrieved assessments are highly relevant; nothing important is missing
  0.7 — Most are relevant; minor gaps or 1-2 irrelevant items
  0.4 — Mixed; some relevant but significant gaps or noise
  0.1 — Mostly irrelevant to the role described
  0.0 — Completely wrong results

Return ONLY valid JSON, nothing else:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


def build_judge(k: int):
    """LLM-as-Judge evaluator factory. Returns a LangSmith-compatible evaluator."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        groq_api_key=GROQ_API_KEY,
    )

    def judge_relevance(
        inputs: dict,
        outputs: dict,
        reference_outputs: dict = None,
    ) -> dict:
        query    = inputs.get("query", "")
        results  = outputs.get("results", [])
        expected = (reference_outputs or {}).get("expected_assessments", [])

        if not results:
            return {"key": "judge_relevance", "score": 0.0,
                    "comment": "No results returned"}

        table  = format_table(results[:k])
        prompt = JUDGE_PROMPT.format(
            query=query, k=k, table=table,
            expected=", ".join(expected) if expected else "not specified",
        )

        try:
            response = llm.invoke(prompt)
            content  = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed    = json.loads(content.strip())
            score     = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
            reasoning = parsed.get("reasoning", "")
            return {"key": "judge_relevance", "score": score, "comment": reasoning}
        except Exception as exc:
            return {"key": "judge_relevance", "score": 0.0,
                    "comment": f"Judge error: {exc}"}

    return judge_relevance


# ── deterministic evaluator factories ────────────────────────────────────────

def build_deterministic_evaluators(k: int) -> list:
    def hit_rate_ev(inputs, outputs, reference_outputs=None) -> dict:
        results  = outputs.get("results", [])
        expected = set((reference_outputs or {}).get("expected_assessments", []))
        return {"key": f"hit_rate@{k}", "score": _hit_rate(results, expected, k)}

    def mrr_ev(inputs, outputs, reference_outputs=None) -> dict:
        results  = outputs.get("results", [])
        expected = set((reference_outputs or {}).get("expected_assessments", []))
        return {"key": f"mrr@{k}", "score": _mrr(results, expected, k)}

    def ndcg_ev(inputs, outputs, reference_outputs=None) -> dict:
        results  = outputs.get("results", [])
        expected = set((reference_outputs or {}).get("expected_assessments", []))
        return {"key": f"ndcg@{k}", "score": _ndcg(results, expected, k)}

    return [hit_rate_ev, mrr_ev, ndcg_ev]


# ── target function factory ───────────────────────────────────────────────────

def build_target(recommender: AssessmentRecommender, k: int):
    """
    LangSmith target function: inputs dict → outputs dict.
    Every call is automatically traced in LangSmith when tracing is enabled.
    """
    def target(inputs: dict) -> dict:
        query   = inputs["query"]
        results = recommender.query(query, top_n=k)
        return {
            "results":         results,
            "retrieved_names": [r["metadata"]["assessment_name"] for r in results],
            "query":           query,
        }
    return target


# ── single-config evaluation ──────────────────────────────────────────────────

def run_evaluation(
    recommender: AssessmentRecommender,
    local_dataset: List[dict],
    k: int,
    experiment_prefix: str,
    upload: bool,
    dataset_name: str,
    verbose: bool = False,
) -> dict:
    target     = build_target(recommender, k)
    evaluators = build_deterministic_evaluators(k)

    if GROQ_API_KEY:
        evaluators.append(build_judge(k))
        print("  LLM-as-Judge : enabled (llama-3.3-70b)")
    else:
        print("  LLM-as-Judge : disabled (GROQ_API_KEY not set)")

    # ── LangSmith path ────────────────────────────────────────────────────────
    if upload:
        from langsmith import Client
        from langsmith.evaluation import evaluate as ls_evaluate

        client     = Client(api_key=LANGSMITH_API_KEY)
        upload_dataset_to_langsmith(client, local_dataset, dataset_name)

        print(f"  Experiment   : '{experiment_prefix}'")
        ls_results = ls_evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            metadata={"k": k},
            max_concurrency=4,
            client=client,
        )

        # Aggregate scores
        scores: dict = {}
        for result in ls_results:
            for fb in result.get("evaluation_results", {}).get("results", []):
                scores.setdefault(fb.key, []).append(fb.score or 0.0)

        summary = {k_: round(sum(v) / len(v), 4) for k_, v in scores.items()}
        print(f"\n  → https://smith.langchain.com/projects/{LANGSMITH_PROJECT}")
        return summary

    # ── local path ────────────────────────────────────────────────────────────
    all_scores: dict = {}
    rows = []

    for item in local_dataset:
        inputs  = {"query": item["query"]}
        ref     = {"expected_assessments": item["expected_assessments"]}
        outputs = target(inputs)

        row = {"query": item["query"], "difficulty": item.get("difficulty", "?")}
        for ev in evaluators:
            result = ev(inputs, outputs, ref)
            key    = result["key"]
            score  = float(result["score"])
            row[key] = round(score, 4)
            all_scores.setdefault(key, []).append(score)
            if "comment" in result and result["comment"]:
                row[f"{key}_comment"] = result["comment"]
        rows.append(row)

    summary = {k_: round(sum(v) / len(v), 4) for k_, v in all_scores.items()}

    if verbose:
        metric_keys = [k_ for k_ in all_scores]
        header = f"{'#':>3}  {'Diff':<8}" + "".join(f" {m:>14}" for m in metric_keys) + "  Query"
        print("\n" + header)
        print("─" * (len(header) + 10))
        for i, row in enumerate(rows, 1):
            line = f"{i:>3}  {row['difficulty']:<8}"
            line += "".join(f" {row.get(m, 0):>14.4f}" for m in metric_keys)
            q = row["query"]
            line += f"  {q[:45]}{'…' if len(q) > 45 else ''}"
            print(line)
            # Show judge reasoning on failures
            if row.get(f"hit_rate@{k}", 0) == 0.0:
                judge_comment = row.get("judge_relevance_comment", "")
                if judge_comment:
                    print(f"     judge: {judge_comment}")

    return summary


# ── comparison ────────────────────────────────────────────────────────────────

def run_comparison(
    assessments,
    local_dataset: List[dict],
    k: int,
    dataset_name: str,
    upload: bool,
    persist_dir: str,
):
    configs = [
        {"label": "bm25-only",           "use_dense": False, "use_reranker": False},
        {"label": "bm25-rerank",          "use_dense": False, "use_reranker": True},
        {"label": "dense-bm25-rrf",       "use_dense": True,  "use_reranker": False},
        {"label": "dense-bm25-rrf-rerank","use_dense": True,  "use_reranker": True},
    ]

    all_summaries = []
    for cfg in configs:
        print(f"\n── {cfg['label']} ──")
        rec = AssessmentRecommender(
            assessments,
            use_dense=cfg["use_dense"],
            use_reranker=cfg["use_reranker"],
            persist_dir=persist_dir,
        )
        summary = run_evaluation(
            recommender=rec,
            local_dataset=local_dataset,
            k=k,
            experiment_prefix=f"assessq-{cfg['label']}",
            upload=upload,
            dataset_name=dataset_name,
        )
        all_summaries.append({"config": cfg["label"], **summary})

    metric_keys = [k_ for k_ in all_summaries[0] if k_ != "config"]
    col_w  = max(len(r["config"]) for r in all_summaries) + 2
    header = f"{'Config':<{col_w}}" + "".join(f" {m:>18}" for m in metric_keys)
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for row in all_summaries:
        line = f"{row['config']:<{col_w}}"
        line += "".join(f" {row.get(m, 0):>18.4f}" for m in metric_keys)
        print(line)
    print(sep)

    if upload:
        print(f"\nAll experiments → https://smith.langchain.com/projects/{LANGSMITH_PROJECT}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AssessQ RAG pipeline with LangSmith + LLM-as-Judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",    default="data/evaluation_dataset.json")
    parser.add_argument("--catalog",    default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--k",          type=int, default=5)
    parser.add_argument("--dense",      action="store_true")
    parser.add_argument("--rerank",     action="store_true")
    parser.add_argument("--compare",    action="store_true",
                        help="Benchmark all 4 pipeline configs")
    parser.add_argument("--verbose",    action="store_true",
                        help="Per-query breakdown with judge reasoning")
    parser.add_argument("--upload",     action="store_true",
                        help="Push results to LangSmith (requires LANGSMITH_API_KEY)")
    parser.add_argument("--experiment", default=None,
                        help="Custom LangSmith experiment name prefix")
    parser.add_argument("--ls-dataset", default="assessq-eval-dataset",
                        help="LangSmith dataset name")
    parser.add_argument("--output",     default=None,
                        help="Save summary to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    upload = args.upload and bool(LANGSMITH_API_KEY)
    if args.upload and not LANGSMITH_API_KEY:
        print("⚠  LANGSMITH_API_KEY not set — running locally.")

    print("Loading catalog...")
    assessments = load_catalog(Path(args.catalog))
    persist_dir = str(Path(args.catalog).parent.parent / "chroma_db")

    print("Loading dataset...")
    local_dataset = load_local_dataset(args.dataset)
    print(f"  {len(local_dataset)} queries loaded.")

    if args.compare:
        print(f"\nComparing all 4 configs at K={args.k}...")
        run_comparison(
            assessments=assessments,
            local_dataset=local_dataset,
            k=args.k,
            dataset_name=args.ls_dataset,
            upload=upload,
            persist_dir=persist_dir,
        )
        return

    use_dense    = args.dense
    use_reranker = args.rerank
    print(f"\nInitializing (Dense={use_dense}, Reranker={use_reranker})...")
    recommender = AssessmentRecommender(
        assessments, use_dense=use_dense,
        use_reranker=use_reranker, persist_dir=persist_dir,
    )

    prefix = args.experiment or (
        f"assessq-{'dense' if use_dense else 'bm25'}"
        f"{'-rerank' if use_reranker else ''}"
    )

    print(f"Evaluating {len(local_dataset)} queries at K={args.k}...")
    summary = run_evaluation(
        recommender=recommender,
        local_dataset=local_dataset,
        k=args.k,
        experiment_prefix=prefix,
        upload=upload,
        dataset_name=args.ls_dataset,
        verbose=args.verbose,
    )

    print("\n" + "─" * 42)
    print(f"  Dense={use_dense}  Reranker={use_reranker}  K={args.k}")
    print("─" * 42)
    for metric, score in summary.items():
        print(f"  {metric:<22}: {score:.4f}")
    print("─" * 42)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {"config": {"dense": use_dense, "reranker": use_reranker, "k": args.k},
                 "metrics": summary},
                f, indent=2,
            )
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()