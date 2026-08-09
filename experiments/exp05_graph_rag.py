"""
Experiment 05 — Graph RAG (Multi-Hop Property Graph Retrieval)
==============================================================
Plan step 5: Introduce a property graph over planning policy entities to
enable multi-hop retrieval that captures cross-referencing between policies.

Architecture:
  Embedding   : Fine-tuned BGE (best from exp04 — planning domain adapted)
  Sparse      : BM25 (exact policy code / keyword matching)
  Seed fusion : Reciprocal Rank Fusion (BM25 + Dense)
  Graph       : Policy knowledge graph (policy codes, NPPF paras, planning topics)
  Expansion   : 2-hop entity walk (seed entities → co-occurring policies → chunks)
  Reranking   : BAAI/bge-reranker-v2-m3 cross-encoder

Multi-hop logic:
  Hop 1 — find all chunks that share a policy code / topic with any seed chunk
  Hop 2 — follow CO_OCCURS edges to find related policies, retrieve their chunks

Hypothesis (plan.md Step 5):
  Achieves highest RAGAS Context Recall for complex, cross-referencing queries
  because graph expansion surfaces policy-linked chunks that dense retrieval alone
  would miss.

Prerequisites:
  1. Run: python scripts/finetune_embeddings.py   (fine-tuned BGE model)
  2. Run: python scripts/build_graph.py           (policy knowledge graph)

Outputs:
  outputs/exp05_graph_rag_results.json
  outputs/exp05_graph_rag_ragas_scores.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table

from src.auto_merger import build_chunk_lookup
from src.config import get_settings, resolve_endpoint_config
from src.evaluator import (
    evaluate_citation_quality,
    load_golden_dataset,
    run_ragas_evaluation,
    save_ragas_scores,
)
from src.generator import answer_question
from src.graph_retriever import PolicyGraph, hybrid_graph_query
from src.indexer import build_index, get_embedding_model, load_chunks
from src.ingest import run_phase0
from src.retriever import BM25Index

console = Console()

GRAPH_PATH      = Path("outputs") / "policy_graph.json"
FINETUNED_BGE   = Path("outputs") / "bge-planning-finetuned"
COLLECTION_NAME = "graph_rag_finetuned_bge"
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
CANDIDATE_POOL  = int(os.getenv("CANDIDATE_POOL", "20"))
GRAPH_HOPS      = int(os.getenv("GRAPH_HOPS", "2"))
GRAPH_MAX_EXP   = int(os.getenv("GRAPH_MAX_EXPANDED", "40"))

# Prior experiment results for final comparison table
PRIOR = {
    "exp01": {"faithfulness": 0.8788, "answer_relevancy": 0.7904,
              "context_precision": 0.6475, "context_recall": 0.6883, "citation_quality": 0.9320},
    "exp02": {"faithfulness": 0.8981, "answer_relevancy": 0.9115,
              "context_precision": 0.9481, "context_recall": 0.8508, "citation_quality": 0.9520},
    "exp03": {"faithfulness": 0.8553, "answer_relevancy": 0.8837,
              "context_precision": 0.9136, "context_recall": 0.7850, "citation_quality": 0.9010},
    "exp04a": {"faithfulness": 0.8983, "answer_relevancy": 0.8759,
               "context_precision": 0.9113, "context_recall": 0.8308, "citation_quality": 0.9370},
    "exp04b": {"faithfulness": 0.9022, "answer_relevancy": 0.9111,
               "context_precision": 0.9459, "context_recall": 0.8608, "citation_quality": 0.9520},
}


def print_final_comparison(exp05_aggregate: dict) -> None:
    metrics = [
        "faithfulness", "answer_relevancy", "context_precision",
        "context_recall", "citation_quality",
    ]

    table = Table(title="Final Experiment Comparison — All 5 Steps")
    table.add_column("Metric",            style="bold")
    table.add_column("Exp01\nBaseline",   justify="right")
    table.add_column("Exp02\nHybrid",     justify="right")
    table.add_column("Exp03\nHierarch.",  justify="right")
    table.add_column("Exp04a\nLEGAL-BRT", justify="right")
    table.add_column("Exp04b\nFT-BGE",   justify="right")
    table.add_column("Exp05\nGraph RAG", justify="right", style="bold green")

    def fmt(d: dict, m: str) -> str:
        v = d.get(m)
        return f"{v:.4f}" if isinstance(v, float) else "—"

    for m in metrics:
        table.add_row(
            m,
            fmt(PRIOR["exp01"],  m),
            fmt(PRIOR["exp02"],  m),
            fmt(PRIOR["exp03"],  m),
            fmt(PRIOR["exp04a"], m),
            fmt(PRIOR["exp04b"], m),
            fmt(exp05_aggregate, m),
        )
    console.print(table)


def main() -> None:
    settings = get_settings()
    project_root = Path(__file__).resolve().parent.parent

    console.print("[bold blue]Experiment 05: Graph RAG (Multi-Hop Property Graph)[/bold blue]")
    console.print(f"  Embedding  : Fine-tuned BGE (planning domain)")
    console.print(f"  Retrieval  : BM25 + Dense → RRF → Graph Expansion ({GRAPH_HOPS} hops) → Rerank")
    console.print(f"  Reranker   : {RERANKER_MODEL}")
    console.print(f"  Graph hops : {GRAPH_HOPS}  |  Max expanded: {GRAPH_MAX_EXP}")

    # ── Resolve fine-tuned model path ────────────────────────────────────────
    ft_model_path = project_root / FINETUNED_BGE
    if not ft_model_path.exists():
        console.print(f"[red]Fine-tuned BGE not found at {ft_model_path}[/red]")
        console.print("[yellow]Run: python scripts/finetune_embeddings.py first[/yellow]")
        return

    # ── Build graph if needed ────────────────────────────────────────────────
    graph_path = project_root / GRAPH_PATH
    if not graph_path.exists():
        console.print("[yellow]Graph not found — building now. Run scripts/build_graph.py.[/yellow]")
        import subprocess, sys
        subprocess.run(
            [sys.executable, str(project_root / "scripts" / "build_graph.py")],
            check=True,
        )

    # ── Load the policy graph ────────────────────────────────────────────────
    console.print(f"[bold]Loading policy graph from {graph_path}[/bold]")
    graph = PolicyGraph(graph_path)
    n_chunks   = sum(1 for n in graph.nodes.values() if n["type"] == "chunk")
    n_entities = len(graph.nodes) - n_chunks
    console.print(f"  Chunk nodes   : {n_chunks}")
    console.print(f"  Entity nodes  : {n_entities}")
    console.print(f"  Total edges   : {len(graph.edges)}")

    # ── Ingestion ────────────────────────────────────────────────────────────
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
    console.print(f"Loaded {len(chunks)} chunks")

    # ── Build chunk lookup (for graph expansion text retrieval) ───────────────
    chunk_lookup = build_chunk_lookup(settings.chunks_path)

    # ── Build ChromaDB index with fine-tuned BGE ─────────────────────────────
    index_summary = build_index(
        chunks_path=settings.chunks_path,
        chroma_path=settings.chroma_path,
        collection_name=COLLECTION_NAME,
        model_name=str(ft_model_path),
    )

    # ── Build BM25 index ──────────────────────────────────────────────────────
    console.print("[bold]Building BM25 index...[/bold]")
    bm25_index = BM25Index(chunks)

    # ── Load embedding model for query encoding ───────────────────────────────
    embed_model = get_embedding_model(str(ft_model_path), device="cpu")

    # ── Load golden dataset ───────────────────────────────────────────────────
    golden_path = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    if not golden_dataset:
        console.print("[red]No golden dataset found. Run scripts/generate_golden_dataset.py first.[/red]")
        return
    queries = [item["question"] for item in golden_dataset]
    console.print(f"Running {len(queries)} queries from golden dataset")

    # ── Retrieval + Generation ────────────────────────────────────────────────
    results: list[dict] = []
    for i, query in enumerate(queries, 1):
        if i % 10 == 0:
            console.print(f"  Query {i}/{len(queries)}...")

        matches = hybrid_graph_query(
            query=query,
            bm25_index=bm25_index,
            embed_model=embed_model,
            chroma_path=settings.chroma_path,
            collection_name=COLLECTION_NAME,
            graph=graph,
            chunk_lookup=chunk_lookup,
            reranker_model=RERANKER_MODEL,
            top_k=settings.top_k,
            candidate_pool=CANDIDATE_POOL,
            graph_hops=GRAPH_HOPS,
            graph_max_expanded=GRAPH_MAX_EXP,
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

        # Track whether graph expansion was used for any retrieved chunk
        graph_used = any(m.get("graph_expanded", False) for m in matches)
        results.append({
            "query":      query,
            "matches":    matches,
            "answer":     answer,
            "graph_used": graph_used,
        })

    n_graph_used = sum(1 for r in results if r["graph_used"])
    console.print(f"[green]Graph expansion triggered for {n_graph_used}/{len(results)} queries[/green]")

    # ── RAGAS + Citation Evaluation ───────────────────────────────────────────
    ragas_scores: dict = {}
    openai_base_url, openai_url_version = resolve_endpoint_config(os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key     = os.getenv("OPENAI_API_KEY", "")
    openai_api_version = os.getenv("OPENAI_API_VERSION", "") or openai_url_version

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

            ragas_path = Path(settings.output_dir) / "exp05_graph_rag_ragas_scores.json"
            save_ragas_scores(ragas_scores, ragas_path)
            console.print(f"[green]RAGAS scores saved to {ragas_path}[/green]")
        except Exception as exc:
            console.print(f"[red]RAGAS evaluation failed: {exc}[/red]")

    # ── Save full results ─────────────────────────────────────────────────────
    output_path = Path(settings.output_dir) / "exp05_graph_rag_results.json"
    payload = {
        "experiment":  "exp05_graph_rag",
        "description": "Graph RAG: Fine-tuned BGE + BM25 + RRF + 2-hop graph expansion + Rerank",
        "retrieval_config": {
            "embedding_model":  str(ft_model_path),
            "collection_name":  COLLECTION_NAME,
            "retrieval":        f"BM25 + Dense → RRF → Graph ({GRAPH_HOPS} hops, max {GRAPH_MAX_EXP}) → {RERANKER_MODEL}",
            "candidate_pool":   CANDIDATE_POOL,
            "graph_hops":       GRAPH_HOPS,
            "graph_max_expanded": GRAPH_MAX_EXP,
            "final_top_k":      settings.top_k,
        },
        "graph_stats": {
            "chunk_nodes":  n_chunks,
            "entity_nodes": n_entities,
            "total_edges":  len(graph.edges),
        },
        "graph_expansion_rate": f"{n_graph_used}/{len(results)}",
        "index_summary":  index_summary,
        "results":        results,
        "ragas_scores":   ragas_scores,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved results to {output_path}[/green]")

    # ── Final comparison table ────────────────────────────────────────────────
    if ragas_scores.get("aggregate"):
        print_final_comparison(ragas_scores["aggregate"])
    else:
        console.print("[yellow]No RAGAS scores available for comparison table.[/yellow]")


if __name__ == "__main__":
    main()
