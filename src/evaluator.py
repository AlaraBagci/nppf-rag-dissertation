from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from rich.console import Console
from rich.table import Table

from src.config import resolve_endpoint_config


console = Console()

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _build_ragas_dataset(results: list[dict], golden_dataset: list[dict]) -> Dataset:
    golden_by_question = {item["question"]: item["ground_truth"] for item in golden_dataset}

    questions, answers, contexts, ground_truths = [], [], [], []
    for result in results:
        question = result["query"]
        ground_truth = golden_by_question.get(question)
        if ground_truth is None:
            continue
        answer = result.get("answer", "")
        ctxs = [m["text"] for m in result.get("matches", [])]
        if not answer or not ctxs:
            continue
        questions.append(question)
        answers.append(answer)
        contexts.append(ctxs)
        ground_truths.append(ground_truth)

    if not questions:
        raise ValueError("No rows could be matched between results and golden dataset.")

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


def _azure_endpoint(base_url: str) -> str:
    """Extract the root Azure endpoint from a full deployment URL."""
    match = re.match(r"(https://[^/]+\.azure\.com)", base_url)
    return (match.group(1) + "/") if match else base_url


def run_ragas_evaluation(
    results: list[dict],
    golden_dataset: list[dict],
    eval_model: str = "gpt-4.1",
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_api_version: str | None = None,
) -> dict[str, Any]:
    resolved_key     = openai_api_key  or os.getenv("OPENAI_API_KEY")  or ""
    resolved_url, url_version = resolve_endpoint_config(openai_base_url or os.getenv("OPENAI_BASE_URL") or "")
    resolved_version = openai_api_version or os.getenv("OPENAI_API_VERSION") or url_version or ""

    ragas_dataset = _build_ragas_dataset(results, golden_dataset)
    console.print(f"[bold]Running RAGAS over {len(ragas_dataset)} samples (judge: {eval_model})[/bold]")

    from langchain_openai import AzureChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    judge_llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=_azure_endpoint(resolved_url),
            api_key=resolved_key,
            api_version=resolved_version,
            azure_deployment=eval_model,
            temperature=0,
        )
    )

    import warnings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import RunConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        judge_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
        )

    run_config = RunConfig(max_workers=2, timeout=120, max_retries=10, max_wait=60)

    score_result = evaluate(
        ragas_dataset,
        metrics=METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    df = score_result.to_pandas()
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores: dict[str, float | None] = _to_python(df[metric_cols].mean().to_dict())

    table = Table(title="RAGAS Evaluation Scores")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    for metric, score in scores.items():
        table.add_row(metric, f"{score:.4f}" if score is not None else "nan")
    console.print(table)

    return {
        "aggregate": scores,
        "per_question": _to_python(df.to_dict(orient="records")),
    }


def load_golden_dataset(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


CITATION_JUDGE_PROMPT = """You are evaluating whether every factual claim in a RAG answer is properly cited.

Score the answer from 0.0 to 1.0 based on these criteria:
- 1.0: Every factual sentence has an inline citation with both the document name AND a specific identifier (policy code, paragraph number, or section). A References section is present.
- 0.7: Most sentences are cited with document name + identifier, but one or two sentences are missing citations.
- 0.4: Some sentences are cited but several factual claims have no citation, OR citations name the document but lack a specific policy/paragraph/section identifier.
- 0.0: One or more factual claims have no citation at all, OR the answer contains general background knowledge not attributed to any source.

Question: {question}
Answer: {answer}

Reply with ONLY a single decimal number between 0.0 and 1.0."""


def evaluate_citation_quality(
    results: list[dict],
    eval_model: str,
    openai_api_key: str,
    openai_base_url: str,
    openai_api_version: str,
) -> dict[str, Any]:
    from langchain_openai import AzureChatOpenAI
    from src.config import resolve_endpoint_config

    resolved_url, url_version = resolve_endpoint_config(openai_base_url)
    resolved_version = openai_api_version or url_version or ""

    llm = AzureChatOpenAI(
        azure_endpoint=_azure_endpoint(resolved_url),
        api_key=openai_api_key,
        api_version=resolved_version,
        azure_deployment=eval_model,
        temperature=0,
    )

    scores: list[float] = []
    per_question: list[dict] = []
    for result in results:
        question = result.get("query", "")
        answer = result.get("answer", "")
        if not answer:
            per_question.append({"question": question, "citation_score": None})
            continue
        prompt = CITATION_JUDGE_PROMPT.format(question=question, answer=answer)
        try:
            raw = llm.invoke(prompt).content.strip()
            score = float(raw.split()[0])
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.0
        scores.append(score)
        per_question.append({"question": question, "citation_score": score})

    avg = float(np.mean(scores)) if scores else None

    table = Table(title="Citation Quality Scores")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_row("citation_quality (avg)", f"{avg:.4f}" if avg is not None else "nan")
    console.print(table)

    return {"aggregate": {"citation_quality": avg}, "per_question": per_question}


def save_ragas_scores(scores: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]RAGAS scores saved to {output_path}[/green]")
