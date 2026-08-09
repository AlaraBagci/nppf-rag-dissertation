"""
Experiment 02 — Hybrid Search + Cross-Encoder Reranking
========================================================
Plan step 2: Upgrade retrieval from pure dense (exp01) to:
  1. BM25 sparse retrieval  — catches exact policy code matches (e.g. "Policy DS3")
  2. Dense vector retrieval — same BGE embeddings as exp01
  3. Reciprocal Rank Fusion — merges both ranked lists into one
  4. Jina AI cross-encoder  — reranks the fused pool for final precision

Hypothesis (plan.md Step 2):
  Sharp increase in Context Precision because BM25 finds exact policy
  keyword matches that pure cosine similarity misses.

Compare outputs/exp02_hybrid_ragas_scores.json against
             outputs/exp01_ragas_scores.json to measure the uplift.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console

from src.config import get_settings, resolve_endpoint_config
from src.evaluator import (
    load_golden_dataset,
    run_ragas_evaluation,
    save_ragas_scores,
    evaluate_citation_quality,
)
from src.generator import answer_question
from src.indexer import build_index, get_embedding_model, load_chunks
from src.ingest import run_phase0
from src.retriever import BM25Index, hybrid_query

console = Console()

COLLECTION_NAME  = "hybrid_naive"
RERANKER_MODEL   = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
CANDIDATE_POOL   = int(os.getenv("CANDIDATE_POOL", "20"))  # BM25 + dense each return this many before fusion


def main() -> None:
    settings = get_settings()
    console.print("[bold blue]Experiment 02: Hybrid Search + Cross-Encoder Reranking[/bold blue]")
    console.print(f"  Dense model : {settings.embedding_model}")
    console.print(f"  Reranker    : {RERANKER_MODEL}")
    console.print(f"  Candidate pool (per retriever): {CANDIDATE_POOL}")
    console.print(f"  Final top-k : {settings.top_k}")

    # ── Ingestion (reuse clean chunks from exp01 if available) ────────────
    if not settings.chunks_path.exists():
        console.print("[yellow]No chunks file found — running ingestion first.[/yellow]")
        run_phase0(
            dataset_path=settings.dataset_path,
            output_path=settings.chunks_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            max_files=settings.max_files,
        )

    # ── Build / reuse ChromaDB dense index ────────────────────────────────
    index_summary = build_index(
        chunks_path=settings.chunks_path,
        chroma_path=settings.chroma_path,
        collection_name=COLLECTION_NAME,
        model_name=settings.embedding_model,
    )

    # ── Build BM25 index in memory ────────────────────────────────────────
    console.print("[bold]Building BM25 index...[/bold]")
    chunks = load_chunks(settings.chunks_path)
    bm25_index = BM25Index(chunks)

    # ── Load embedding model once (reused across all queries) ─────────────
    embed_model = get_embedding_model(settings.embedding_model, device="cpu")

    # ── Load golden dataset ───────────────────────────────────────────────
    golden_path = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    if not golden_dataset:
        console.print("[red]No golden dataset found. Run scripts/generate_golden_dataset.py first.[/red]")
        return
    queries = [item["question"] for item in golden_dataset]
    console.print(f"Running {len(queries)} queries from golden dataset")

    # ── Retrieval + Generation ────────────────────────────────────────────
    results: list[dict] = []
    for query in queries:
        matches = hybrid_query(
            query=query,
            bm25_index=bm25_index,
            embed_model=embed_model,
            chroma_path=settings.chroma_path,
            collection_name=COLLECTION_NAME,
            reranker_model=RERANKER_MODEL,
            top_k=settings.top_k,
            candidate_pool=CANDIDATE_POOL,
        )

        answer = ""
        if settings.llm_base_url and settings.llm_api_key:
            console.print(f"[dim]Calling LLM...[/dim]")
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

    # ── RAGAS + Citation Evaluation ───────────────────────────────────────
    ragas_scores: dict = {}
    openai_base_url, openai_url_version = resolve_endpoint_config(os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key     = os.getenv("OPENAI_API_KEY", "")
    openai_api_version = os.getenv("OPENAI_API_VERSION", "") or openai_url_version

    if golden_dataset and openai_base_url and openai_api_key:
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

            ragas_output = Path(settings.output_dir) / "exp02_hybrid_ragas_scores.json"
            save_ragas_scores(ragas_scores, ragas_output)
        except Exception as exc:
            console.print(f"[red]RAGAS evaluation failed: {exc}[/red]")

    # ── Save full results ──────────────────────────────────────────────────
    output_path = Path(settings.output_dir) / "exp02_hybrid_results.json"
    payload = {
        "experiment": "exp02_hybrid",
        "description": "Hybrid BM25 + dense retrieval with RRF fusion and BGE cross-encoder reranking",
        "retrieval_config": {
            "dense_model": settings.embedding_model,
            "reranker_model": RERANKER_MODEL,
            "candidate_pool_per_retriever": CANDIDATE_POOL,
            "final_top_k": settings.top_k,
            "fusion": "Reciprocal Rank Fusion (k=60)",
        },
        "config": {
            "dataset_path": str(settings.dataset_path),
            "chunks_path": str(settings.chunks_path),
            "chroma_path": str(settings.chroma_path),
            "collection_name": COLLECTION_NAME,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "top_k": settings.top_k,
        },
        "index_summary": index_summary,
        "results": results,
        "ragas_scores": ragas_scores,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved hybrid results to {output_path}[/green]")

    # ── Print comparison summary ───────────────────────────────────────────
    if ragas_scores.get("aggregate"):
        console.print("\n[bold]Experiment 02 Results:[/bold]")
        console.print("[bold]Metric                  Exp02 (Hybrid)[/bold]")
        for metric, score in ragas_scores["aggregate"].items():
            label = f"{score:.4f}" if isinstance(score, float) else str(score)
            console.print(f"  {metric:<25} {label}")
        console.print("\n[dim]Compare against outputs/exp01_ragas_scores.json[/dim]")


if __name__ == "__main__":
    main()
