"""
Experiment 03 — Hierarchical RAG + Hybrid Retrieval + Reranking (Option B)
===========================================================================
Plan step 3: 3-level chunk hierarchy with improved leaf size and hybrid retrieval.

  Level 0 — Grandparent : 2048 words  (full section context for LLM)
  Level 1 — Parent      :  512 words  (policy paragraph for LLM)
  Level 2 — Leaf        :  256 words  (embedded & retrieved — richer than 128w)

Why 256-word leaves (not 128):
  128-word chunks are too short for BGE to form a meaningful embedding of
  planning policy text. 256 words captures a complete policy clause while
  still being smaller than the 512-word parent — the merge step still
  adds value by restoring surrounding context.

Retrieval pipeline (Option B — Hybrid + Hierarchical):
  1. BM25 sparse retrieval on leaf index (exact policy code matching)
  2. Dense vector retrieval on leaf index (semantic matching)
  3. Reciprocal Rank Fusion of both leaf lists
  4. AutoMerge: if ≥50% of a parent's leaves retrieved → use parent (512w)
                if ≥50% of a grandparent's parents merged → use grandparent (2048w)
  5. Cross-encoder reranking (BAAI/bge-reranker-v2-m3) of merged results

Outputs: outputs/exp03_hierarchical_results.json
         outputs/exp03_hierarchical_ragas_scores.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console

from src.auto_merger import build_chunk_lookup, hybrid_hierarchical_query
from src.config import get_settings, resolve_endpoint_config
from src.evaluator import (
    evaluate_citation_quality,
    load_golden_dataset,
    run_ragas_evaluation,
    save_ragas_scores,
)
from src.generator import answer_question
from src.indexer import build_index, get_embedding_model, load_chunks
from src.ingest import run_phase0_hierarchical
from src.retriever import BM25Index

console = Console()

COLLECTION_NAME  = "hierarchical_256_leaves"
HIERARCHY_SIZES  = (2048, 512, 256)   # 256-word leaves — richer embeddings
CANDIDATE_POOL   = int(os.getenv("CANDIDATE_POOL", "30"))
MERGE_RATIO      = float(os.getenv("MERGE_RATIO", "0.5"))
RERANKER_MODEL   = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def main() -> None:
    settings = get_settings()
    console.print("[bold blue]Experiment 03: Hierarchical RAG (256w leaves) + Hybrid + Reranking[/bold blue]")
    console.print(f"  Hierarchy       : {HIERARCHY_SIZES[0]}w → {HIERARCHY_SIZES[1]}w → {HIERARCHY_SIZES[2]}w")
    console.print(f"  Retrieval       : BM25 + Dense → RRF → AutoMerge → Rerank")
    console.print(f"  Reranker        : {RERANKER_MODEL}")
    console.print(f"  Candidate pool  : {CANDIDATE_POOL}  |  Merge ratio: {MERGE_RATIO}  |  Top-k: {settings.top_k}")

    # ── Hierarchical ingestion (256-word leaves) ───────────────────────────
    hier_chunks_path = Path(settings.output_dir) / "chunks_hierarchical_256.jsonl"
    if not hier_chunks_path.exists():
        console.print("[yellow]Building hierarchical chunks (2048/512/256) — runs once.[/yellow]")
        run_phase0_hierarchical(
            dataset_path=settings.dataset_path,
            output_path=hier_chunks_path,
            sizes=HIERARCHY_SIZES,
            max_files=settings.max_files,
        )
    else:
        console.print(f"[green]Hierarchical chunks already exist — skipping.[/green]")

    # ── Filter & write leaf-only JSONL for ChromaDB indexing ──────────────
    leaf_chunks_path = Path(settings.output_dir) / "chunks_hierarchical_256_leaves.jsonl"
    if not leaf_chunks_path.exists():
        console.print("[yellow]Filtering leaf chunks (level 2) for indexing...[/yellow]")
        with hier_chunks_path.open("r", encoding="utf-8") as src, \
             leaf_chunks_path.open("w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    chunk = json.loads(line)
                    if chunk.get("metadata", {}).get("level") == 2:
                        dst.write(line)
        console.print(f"[green]Leaf chunk file ready: {leaf_chunks_path}[/green]")

    # ── Build ChromaDB dense index over leaves only ────────────────────────
    index_summary = build_index(
        chunks_path=leaf_chunks_path,
        chroma_path=settings.chroma_path,
        collection_name=COLLECTION_NAME,
        model_name=settings.embedding_model,
    )

    # ── Build BM25 index over leaves ──────────────────────────────────────
    console.print("[bold]Building BM25 index over leaf chunks...[/bold]")
    leaf_chunks = load_chunks(leaf_chunks_path)
    bm25_index = BM25Index(leaf_chunks)

    # ── Load full hierarchy into memory for auto-merging ──────────────────
    console.print("[bold]Loading full chunk hierarchy...[/bold]")
    chunk_lookup = build_chunk_lookup(hier_chunks_path)
    console.print(f"[green]Loaded {len(chunk_lookup)} chunks (all levels)[/green]")

    # ── Load embedding model ───────────────────────────────────────────────
    embed_model = get_embedding_model(settings.embedding_model, device="cpu")

    # ── Load golden dataset ───────────────────────────────────────────────
    golden_path = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    if not golden_dataset:
        console.print("[red]No golden dataset found. Run scripts/generate_golden_dataset.py first.[/red]")
        return
    queries = [item["question"] for item in golden_dataset]
    console.print(f"Running {len(queries)} queries from golden dataset")

    # ── Retrieval + AutoMerge + Rerank + Generation ───────────────────────
    results: list[dict] = []
    for query in queries:
        matches = hybrid_hierarchical_query(
            query=query,
            bm25_index=bm25_index,
            embed_model=embed_model,
            chroma_path=settings.chroma_path,
            collection_name=COLLECTION_NAME,
            chunk_lookup=chunk_lookup,
            reranker_model=RERANKER_MODEL,
            top_k=settings.top_k,
            candidate_pool=CANDIDATE_POOL,
            merge_ratio=MERGE_RATIO,
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

            ragas_output = Path(settings.output_dir) / "exp03_hierarchical_ragas_scores.json"
            save_ragas_scores(ragas_scores, ragas_output)
        except Exception as exc:
            console.print(f"[red]RAGAS evaluation failed: {exc}[/red]")

    # ── Save full results ─────────────────────────────────────────────────
    output_path = Path(settings.output_dir) / "exp03_hierarchical_results.json"
    payload = {
        "experiment": "exp03_hierarchical",
        "description": "Hierarchical RAG (2048/512/256w) + BM25 + Dense RRF + AutoMerge + Rerank",
        "retrieval_config": {
            "hierarchy_sizes_words": list(HIERARCHY_SIZES),
            "indexed_level": "leaf (256 words)",
            "retrieval": "BM25 + Dense → RRF → AutoMerge → bge-reranker-v2-m3",
            "candidate_pool": CANDIDATE_POOL,
            "merge_ratio": MERGE_RATIO,
            "final_top_k": settings.top_k,
        },
        "config": {
            "dataset_path": str(settings.dataset_path),
            "hier_chunks_path": str(hier_chunks_path),
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
    console.print(f"[green]Saved results to {output_path}[/green]")

    if ragas_scores.get("aggregate"):
        console.print("\n[bold]Experiment 03 Results (Hybrid + Hierarchical):[/bold]")
        headers = f"{'Metric':<25} {'Exp01':>8} {'Exp02':>8} {'Exp03':>8}"
        console.print(headers)
        baselines = {"faithfulness": 0.8788, "answer_relevancy": 0.7904,
                     "context_precision": 0.6475, "context_recall": 0.6883,
                     "citation_quality": 0.9320}
        hybrid =    {"faithfulness": 0.8981, "answer_relevancy": 0.9115,
                     "context_precision": 0.9481, "context_recall": 0.8508,
                     "citation_quality": 0.9520}
        for metric, score in ragas_scores["aggregate"].items():
            b = baselines.get(metric, 0)
            h = hybrid.get(metric, 0)
            s = f"{score:.4f}" if isinstance(score, float) else str(score)
            console.print(f"  {metric:<25} {b:>8.4f} {h:>8.4f} {s:>8}")


if __name__ == "__main__":
    main()
