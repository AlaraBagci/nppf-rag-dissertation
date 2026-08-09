"""
Experiment 00 — No Preprocessing Ablation Study
================================================
Runs the same Naive RAG pipeline as exp01_baseline but with ALL data quality
gates disabled during ingestion:

  - No duplicate document removal (same file in multiple folders is indexed twice)
  - No empty / whitespace-only page filtering (blank pages become noise chunks)
  - No junk file exclusion (~$temp files, __MACOSX artefacts included)
  - No minimum chunk length (single-word chunks from headers / page numbers included)

Purpose: demonstrate that RAGAS metrics drop without preprocessing, justifying
every cleaning step in the dissertation methodology chapter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console

from src.config import get_settings, resolve_endpoint_config
from src.evaluator import load_golden_dataset, run_ragas_evaluation, save_ragas_scores, evaluate_citation_quality
from src.generator import answer_question
from src.indexer import build_index, query_index
from src.ingest import run_phase0_raw

console = Console()

COLLECTION_NAME = "no_preprocess_naive"


def main() -> None:
    settings = get_settings()
    console.print("[bold red]Experiment 00: Naive RAG — NO Preprocessing (Ablation Study)[/bold red]")
    console.print("All data quality gates are disabled. Results will be compared against exp01_baseline.")

    # ── Raw ingestion (no quality gates) ──────────────────────────────────
    raw_chunks_path = Path(settings.output_dir) / "chunks_raw.jsonl"
    if not raw_chunks_path.exists():
        console.print("[yellow]Running raw ingestion (no preprocessing)...[/yellow]")
        ingest_summary = run_phase0_raw(
            dataset_path=settings.dataset_path,
            output_path=raw_chunks_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    else:
        console.print(f"[green]Raw chunks already exist at {raw_chunks_path} — skipping re-ingest.[/green]")
        ingest_summary = {"note": "loaded from cache"}

    # ── Build index from raw (unfiltered) chunks ───────────────────────────
    index_summary = build_index(
        chunks_path=raw_chunks_path,
        chroma_path=settings.chroma_path,
        collection_name=COLLECTION_NAME,
        model_name=settings.embedding_model,
    )

    # ── Load golden dataset (same 100 Q&A pairs as exp01) ─────────────────
    golden_path = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    if not golden_dataset:
        console.print("[red]No golden dataset found. Run scripts/generate_golden_dataset.py first.[/red]")
        return
    queries = [item["question"] for item in golden_dataset]
    console.print(f"Running {len(queries)} queries from golden dataset")

    # ── Retrieval + Generation ─────────────────────────────────────────────
    results: list[dict] = []
    for query in queries:
        matches = query_index(
            query=query,
            chroma_path=settings.chroma_path,
            collection_name=COLLECTION_NAME,
            model_name=settings.embedding_model,
            top_k=settings.top_k,
            device="cpu",
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

    # ── RAGAS + Citation Evaluation ────────────────────────────────────────
    ragas_scores: dict = {}
    openai_base_url, openai_url_version = resolve_endpoint_config(os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
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

            ragas_output = Path(settings.output_dir) / "exp00_no_preprocess_ragas_scores.json"
            save_ragas_scores(ragas_scores, ragas_output)
        except Exception as exc:
            console.print(f"[red]RAGAS evaluation failed: {exc}[/red]")

    # ── Save full results ──────────────────────────────────────────────────
    output_path = Path(settings.output_dir) / "exp00_no_preprocess_results.json"
    payload = {
        "experiment": "exp00_no_preprocess",
        "description": "Ablation study — no data preprocessing or quality filtering",
        "preprocessing_disabled": [
            "duplicate document removal",
            "empty page filtering",
            "junk file exclusion",
            "minimum chunk length filter",
        ],
        "config": {
            "dataset_path": str(settings.dataset_path),
            "raw_chunks_path": str(raw_chunks_path),
            "chroma_path": str(settings.chroma_path),
            "collection_name": COLLECTION_NAME,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "top_k": settings.top_k,
        },
        "ingest_summary": ingest_summary,
        "index_summary": index_summary,
        "results": results,
        "ragas_scores": ragas_scores,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved no-preprocessing results to {output_path}[/green]")

    # ── Print comparison reminder ──────────────────────────────────────────
    if ragas_scores.get("aggregate"):
        console.print("\n[bold]Ablation Results (no preprocessing):[/bold]")
        for metric, score in ragas_scores["aggregate"].items():
            label = f"{score:.4f}" if isinstance(score, float) else str(score)
            console.print(f"  {metric}: {label}")
        console.print("\n[dim]Compare these against exp01_ragas_scores.json to quantify the impact of preprocessing.[/dim]")


if __name__ == "__main__":
    main()
