"""
Experiment 04 — Domain-Specific Embeddings
===========================================
Plan step 4: Swap the general BGE embedding for two domain-specific alternatives
and measure retrieval quality improvement.

Model A — LEGAL-BERT  (nlpaueb/legal-bert-base-uncased):
  Pre-trained on 12GB of legal text (EU legislation, UK legal cases, contracts).
  Understands legal terminology but not fine-tuned for retrieval.

Model B — Fine-tuned BGE  (outputs/bge-planning-finetuned/):
  BAAI/bge-large-en-v1.5 fine-tuned on (question, source_chunk) pairs from
  the golden dataset. Adapts the embedding space specifically to UK planning
  policy vocabulary (Green Belt, Affordable Housing, Policy DS1, etc.).

Both models use the same hybrid retrieval pipeline as exp02
(BM25 + Dense → RRF → BGE cross-encoder reranker) for a fair comparison.

Hypothesis (plan.md Step 4):
  Fine-tuned BGE achieves highest vector retrieval scores because it has
  learned the specific jargon of Coventry planning policy.

Prerequisites:
  Run scripts/finetune_embeddings.py before this experiment.

Outputs:
  outputs/exp04_legal_bert_ragas_scores.json
  outputs/exp04_finetuned_bge_ragas_scores.json
  outputs/exp04_domain_embeddings_results.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table

from src.config import get_settings, resolve_endpoint_config
from src.evaluator import (
    evaluate_citation_quality,
    load_golden_dataset,
    run_ragas_evaluation,
    save_ragas_scores,
)
from src.generator import answer_question
from src.indexer import build_index, get_embedding_model, load_chunks
from src.ingest import run_phase0
from src.retriever import BM25Index, hybrid_query

console = Console()

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "20"))

MODELS = {
    "legal_bert": {
        "model_name":      "nlpaueb/legal-bert-base-uncased",
        "collection_name": "legal_bert_hybrid",
        "label":           "LEGAL-BERT",
    },
    "finetuned_bge": {
        "model_name":      str(Path("outputs") / "bge-planning-finetuned"),
        "collection_name": "finetuned_bge_hybrid",
        "label":           "Fine-tuned BGE",
    },
}

# Prior experiment scores for comparison table
PRIOR = {
    "exp01": {"faithfulness": 0.8788, "answer_relevancy": 0.7904,
              "context_precision": 0.6475, "context_recall": 0.6883, "citation_quality": 0.9320},
    "exp02": {"faithfulness": 0.8981, "answer_relevancy": 0.9115,
              "context_precision": 0.9481, "context_recall": 0.8508, "citation_quality": 0.9520},
    "exp03": {"faithfulness": 0.8553, "answer_relevancy": 0.8837,
              "context_precision": 0.9136, "context_recall": 0.7850, "citation_quality": 0.9010},
}


def run_model_experiment(
    model_key: str,
    model_cfg: dict,
    settings,
    chunks,
    golden_dataset: list[dict],
    openai_api_key: str,
    openai_base_url: str,
    openai_api_version: str,
) -> dict:
    model_name      = model_cfg["model_name"]
    collection_name = model_cfg["collection_name"]
    label           = model_cfg["label"]

    console.rule(f"[bold cyan]{label}[/bold cyan]")
    console.print(f"Model : {model_name}")

    # Resolve fine-tuned model path relative to project root
    if not model_name.startswith("outputs/") and not Path(model_name).exists():
        resolved_model = model_name
    else:
        resolved_model = str(Path(__file__).resolve().parent.parent / model_name) \
            if not Path(model_name).is_absolute() else model_name

    if model_key == "finetuned_bge" and not Path(resolved_model).exists():
        console.print(f"[red]Fine-tuned model not found at {resolved_model}[/red]")
        console.print("[yellow]Run: python scripts/finetune_embeddings.py first[/yellow]")
        return {}

    # ── Build ChromaDB index with this embedding model ─────────────────────
    index_summary = build_index(
        chunks_path=settings.chunks_path,
        chroma_path=settings.chroma_path,
        collection_name=collection_name,
        model_name=resolved_model,
    )

    # ── Build BM25 index (same for all models — text doesn't change) ───────
    bm25_index = BM25Index(chunks)

    # ── Load this model for query embedding ───────────────────────────────
    embed_model = get_embedding_model(resolved_model, device="cpu")

    # ── Retrieval + Generation ─────────────────────────────────────────────
    queries = [item["question"] for item in golden_dataset]
    console.print(f"Running {len(queries)} queries")

    results: list[dict] = []
    for query in queries:
        matches = hybrid_query(
            query=query,
            bm25_index=bm25_index,
            embed_model=embed_model,
            chroma_path=settings.chroma_path,
            collection_name=collection_name,
            reranker_model=RERANKER_MODEL,
            top_k=settings.top_k,
            candidate_pool=CANDIDATE_POOL,
        )
        answer = ""
        if settings.llm_base_url and settings.llm_api_key:
            try:
                answer = answer_question(
                    question=query,
                    matches=matches,
                    model=settings.llm_model,
                    base_url=settings.llm_base_url,
                    api_key=settings.llm_api_key,
                    api_version=settings.llm_api_version,
                )
            except Exception as exc:
                console.print(f"[red]LLM call failed: {exc}[/red]")
        results.append({"query": query, "matches": matches, "answer": answer})

    # ── RAGAS + Citation ───────────────────────────────────────────────────
    ragas_scores: dict = {}
    if openai_base_url and openai_api_key:
        try:
            ragas_scores = run_ragas_evaluation(
                results=results,
                golden_dataset=golden_dataset,
                eval_model=settings.eval_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                openai_api_version=openai_api_version,
            )
            citation_scores = evaluate_citation_quality(
                results=results,
                eval_model=settings.eval_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                openai_api_version=openai_api_version,
            )
            ragas_scores["aggregate"]["citation_quality"] = citation_scores["aggregate"]["citation_quality"]
            ragas_scores["citation_per_question"] = citation_scores["per_question"]

            score_path = Path(settings.output_dir) / f"exp04_{model_key}_ragas_scores.json"
            save_ragas_scores(ragas_scores, score_path)
        except Exception as exc:
            console.print(f"[red]RAGAS failed: {exc}[/red]")

    return {
        "label": label,
        "model_name": model_name,
        "collection_name": collection_name,
        "index_summary": index_summary,
        "results": results,
        "ragas_scores": ragas_scores,
    }


def print_comparison(all_results: dict) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "citation_quality"]

    table = Table(title="Experiment Comparison — All Steps")
    table.add_column("Metric", style="bold")
    table.add_column("Exp01\nBaseline", justify="right")
    table.add_column("Exp02\nHybrid", justify="right")
    table.add_column("Exp03\nHierarchical", justify="right")
    table.add_column("Exp04a\nLEGAL-BERT", justify="right")
    table.add_column("Exp04b\nFine-tuned BGE", justify="right")

    def fmt(d: dict, m: str) -> str:
        v = d.get(m)
        return f"{v:.4f}" if isinstance(v, float) else "—"

    legal_agg   = all_results.get("legal_bert",    {}).get("ragas_scores", {}).get("aggregate", {})
    ft_agg      = all_results.get("finetuned_bge", {}).get("ragas_scores", {}).get("aggregate", {})

    for m in metrics:
        table.add_row(
            m,
            fmt(PRIOR["exp01"], m),
            fmt(PRIOR["exp02"], m),
            fmt(PRIOR["exp03"], m),
            fmt(legal_agg, m),
            fmt(ft_agg, m),
        )
    console.print(table)


def main() -> None:
    settings = get_settings()
    console.print("[bold blue]Experiment 04: Domain-Specific Embeddings[/bold blue]")
    console.print("  Model A : nlpaueb/legal-bert-base-uncased")
    console.print("  Model B : Fine-tuned BAAI/bge-large-en-v1.5 (planning domain)")
    console.print(f"  Retrieval: BM25 + Domain Embeddings → RRF → {RERANKER_MODEL}")

    # ── Shared setup ──────────────────────────────────────────────────────
    if not settings.chunks_path.exists():
        console.print("[yellow]No chunks found — running ingestion.[/yellow]")
        run_phase0(
            dataset_path=settings.dataset_path,
            output_path=settings.chunks_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            max_files=settings.max_files,
        )

    chunks = load_chunks(settings.chunks_path)

    golden_path = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    if not golden_dataset:
        console.print("[red]No golden dataset found.[/red]")
        return

    openai_base_url, openai_url_version = resolve_endpoint_config(os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key     = os.getenv("OPENAI_API_KEY", "")
    openai_api_version = os.getenv("OPENAI_API_VERSION", "") or openai_url_version

    # ── Run each domain model ─────────────────────────────────────────────
    all_results: dict = {}
    for model_key, model_cfg in MODELS.items():
        result = run_model_experiment(
            model_key=model_key,
            model_cfg=model_cfg,
            settings=settings,
            chunks=chunks,
            golden_dataset=golden_dataset,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_api_version=openai_api_version,
        )
        all_results[model_key] = result

    # ── Save combined output ───────────────────────────────────────────────
    output_path = Path(settings.output_dir) / "exp04_domain_embeddings_results.json"
    payload = {
        "experiment": "exp04_domain_embeddings",
        "description": "Domain embedding comparison: LEGAL-BERT vs fine-tuned BGE",
        "retrieval_pipeline": f"BM25 + Domain Embeddings → RRF → {RERANKER_MODEL}",
        "models": {k: {
            "label": v.get("label"),
            "model_name": v.get("model_name"),
            "ragas_aggregate": v.get("ragas_scores", {}).get("aggregate", {}),
        } for k, v in all_results.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]Saved combined results to {output_path}[/green]")

    print_comparison(all_results)


if __name__ == "__main__":
    main()
