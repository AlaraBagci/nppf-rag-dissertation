"""
Experiment 07 — LLM Graph RAG (GPT-4.1 Extracted Triples)
==========================================================
Plan step 7: Replace regex-based entity extraction in the graph with
GPT-4.1-extracted typed triples, then compare retrieval quality against
the regex graph (exp05).

The key difference from exp05:
  exp05 graph  — edges built by regex pattern matching:
                 (chunk) -[MENTIONS]-> policy/topic/nppf_para
                 (chunk) -[CROSS_DOC]-> (chunk)  (shared topic keyword)
                 (policy) -[CO_OCCURS]-> (policy)  (same chunk)

  exp07 graph  — all of the above PLUS GPT-4.1 triples:
                 (chunk) -[LLM_IMPLEMENTS]-> nppf:NPPF_para_63   weight=3
                 (chunk) -[LLM_REQUIRES]->   concept:affordable_housing  weight=3
                 (policy:H4) -[LLM_IMPLEMENTS]-> nppf:NPPF_para_63  weight=3
                 ...

Why typed relations matter for graph expansion:
  Regex knows DS1 and NPPF_para_11 appear near each other.
  GPT-4.1 knows DS1 *implements* NPPF_para_11 — a legal dependency.
  During expansion, IMPLEMENTS edges give weight=3 (vs weight=1 for
  a shared keyword match), so the actual NPPF paragraph ranks
  higher in the candidate pool before cross-encoder reranking.

Hypothesis:
  LLM-typed edges improve context_recall for multi-hop queries because
  the graph now understands *why* Local Plan policies connect to NPPF
  paragraphs, not just *that* they co-occur.

Prerequisites:
  1. python scripts/build_graph.py             (v2 regex graph)
  2. python scripts/build_graph_llm.py         (LLM-enriched graph)
  3. python scripts/finetune_embeddings.py     (fine-tuned BGE)

Outputs:
  outputs/exp07_llm_graph_rag_results.json
  outputs/exp07_llm_graph_rag_ragas_scores.json
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

LLM_GRAPH_PATH  = Path("outputs") / "policy_graph_llm.json"
FINETUNED_BGE   = Path("outputs") / "bge-planning-finetuned"
COLLECTION_NAME = "llm_graph_rag_finetuned_bge"
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
CANDIDATE_POOL  = int(os.getenv("CANDIDATE_POOL", "20"))
GRAPH_HOPS      = 2
GRAPH_MAX_EXP   = 40

PRIOR = {
    "exp01":  {"faithfulness": 0.8788, "answer_relevancy": 0.7904,
               "context_precision": 0.6475, "context_recall": 0.6883, "citation_quality": 0.9320},
    "exp02":  {"faithfulness": 0.8981, "answer_relevancy": 0.9115,
               "context_precision": 0.9481, "context_recall": 0.8508, "citation_quality": 0.9520},
    "exp04b": {"faithfulness": 0.9022, "answer_relevancy": 0.9111,
               "context_precision": 0.9459, "context_recall": 0.8608, "citation_quality": 0.9520},
    "exp05":  {"faithfulness": 0.9074, "answer_relevancy": 0.9113,
               "context_precision": 0.9506, "context_recall": 0.8542, "citation_quality": 0.9270},
}


def print_comparison(exp07_aggregate: dict) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "citation_quality"]

    table = Table(title="Graph RAG Comparison — Regex vs LLM Triples")
    table.add_column("Metric",            style="bold")
    table.add_column("Exp01\nBaseline",   justify="right")
    table.add_column("Exp02\nHybrid",     justify="right")
    table.add_column("Exp04b\nFT-BGE",   justify="right")
    table.add_column("Exp05\nRegex\nGraph", justify="right")
    table.add_column("Exp07\nLLM\nGraph", justify="right", style="bold green")

    def fmt(d: dict, m: str) -> str:
        v = d.get(m)
        return f"{v:.4f}" if isinstance(v, float) else "—"

    for m in metrics:
        table.add_row(
            m,
            fmt(PRIOR["exp01"],  m),
            fmt(PRIOR["exp02"],  m),
            fmt(PRIOR["exp04b"], m),
            fmt(PRIOR["exp05"],  m),
            fmt(exp07_aggregate, m),
        )
    console.print(table)


def main() -> None:
    settings     = get_settings()
    project_root = Path(__file__).resolve().parent.parent

    console.print("[bold blue]Experiment 07: LLM Graph RAG (GPT-4.1 Typed Triples)[/bold blue]")
    console.print(f"  Graph      : policy_graph_llm.json  (regex v2 + GPT-4.1 triples)")
    console.print(f"  Embedding  : Fine-tuned BGE (planning domain)")
    console.print(f"  Retrieval  : BM25 + Dense → RRF → LLM Graph Expansion → Rerank")

    # ── Check prerequisites ───────────────────────────────────────────────────
    llm_graph_path = project_root / LLM_GRAPH_PATH
    if not llm_graph_path.exists():
        console.print(f"[red]LLM graph not found at {llm_graph_path}[/red]")
        console.print("[yellow]Run: python scripts/build_graph_llm.py first[/yellow]")
        return

    ft_model_path = project_root / FINETUNED_BGE
    if not ft_model_path.exists():
        console.print(f"[red]Fine-tuned BGE not found at {ft_model_path}[/red]")
        console.print("[yellow]Run: python scripts/finetune_embeddings.py first[/yellow]")
        return

    # ── Load LLM graph ────────────────────────────────────────────────────────
    console.print(f"[bold]Loading LLM-enriched graph...[/bold]")
    graph = PolicyGraph(llm_graph_path)

    n_chunks   = sum(1 for n in graph.nodes.values() if n["type"] == "chunk")
    n_entities = len(graph.nodes) - n_chunks
    llm_edges  = sum(1 for e in graph.edges if e["relation"].startswith("LLM_"))

    console.print(f"  Chunk nodes    : {n_chunks}")
    console.print(f"  Entity nodes   : {n_entities}")
    console.print(f"  Total edges    : {len(graph.edges)}")
    console.print(f"  LLM triple edges: {llm_edges}")

    # ── Ingestion ─────────────────────────────────────────────────────────────
    if not settings.chunks_path.exists():
        run_phase0(
            dataset_path=settings.dataset_path,
            output_path=settings.chunks_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            max_files=settings.max_files,
        )

    chunks = load_chunks(settings.chunks_path)
    chunk_lookup = build_chunk_lookup(settings.chunks_path)

    # ── Build ChromaDB index with fine-tuned BGE ──────────────────────────────
    index_summary = build_index(
        chunks_path=settings.chunks_path,
        chroma_path=settings.chroma_path,
        collection_name=COLLECTION_NAME,
        model_name=str(ft_model_path),
    )

    # ── BM25 + embedding model ────────────────────────────────────────────────
    bm25_index  = BM25Index(chunks)
    embed_model = get_embedding_model(str(ft_model_path), device="cpu")

    # ── Golden dataset ────────────────────────────────────────────────────────
    golden_path = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    if not golden_dataset:
        console.print("[red]No golden dataset found.[/red]")
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

        llm_expanded = any(m.get("graph_expanded") and m.get("entity_overlap", 0) >= 2
                           for m in matches)
        results.append({
            "query":        query,
            "matches":      matches,
            "answer":       answer,
            "llm_expanded": llm_expanded,
        })

    n_llm_used = sum(1 for r in results if r["llm_expanded"])
    console.print(f"[green]LLM triple expansion contributed to {n_llm_used}/{len(results)} queries[/green]")

    # ── RAGAS + Citation ──────────────────────────────────────────────────────
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
            ragas_scores["aggregate"]["citation_quality"] = \
                citation_scores["aggregate"]["citation_quality"]
            ragas_scores["citation_per_question"] = citation_scores["per_question"]

            ragas_path = Path(settings.output_dir) / "exp07_llm_graph_rag_ragas_scores.json"
            save_ragas_scores(ragas_scores, ragas_path)
            console.print(f"[green]RAGAS scores saved to {ragas_path}[/green]")
        except Exception as exc:
            console.print(f"[red]RAGAS evaluation failed: {exc}[/red]")

    # ── Save full results ─────────────────────────────────────────────────────
    output_path = Path(settings.output_dir) / "exp07_llm_graph_rag_results.json"
    payload = {
        "experiment":  "exp07_llm_graph_rag",
        "description": "LLM Graph RAG: Fine-tuned BGE + BM25 + RRF + GPT-4.1 typed triples + Rerank",
        "retrieval_config": {
            "embedding_model":   str(ft_model_path),
            "graph":             "policy_graph_llm.json",
            "llm_triple_edges":  llm_edges,
            "retrieval":         f"BM25 + Dense → RRF → LLM Graph ({GRAPH_HOPS} hops) → {RERANKER_MODEL}",
            "candidate_pool":    CANDIDATE_POOL,
            "graph_max_expanded": GRAPH_MAX_EXP,
            "final_top_k":       settings.top_k,
        },
        "graph_stats": {
            "chunk_nodes":   n_chunks,
            "entity_nodes":  n_entities,
            "total_edges":   len(graph.edges),
            "llm_edges":     llm_edges,
        },
        "llm_expansion_rate": f"{n_llm_used}/{len(results)}",
        "index_summary":  index_summary,
        "results":        results,
        "ragas_scores":   ragas_scores,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved results to {output_path}[/green]")

    if ragas_scores.get("aggregate"):
        print_comparison(ragas_scores["aggregate"])
    else:
        console.print("[yellow]No RAGAS scores — check API credentials.[/yellow]")


if __name__ == "__main__":
    main()
