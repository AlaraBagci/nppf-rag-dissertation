"""
Multi-Hop Cross-Document Evaluation — the test Graph RAG was actually built for.

The original golden_dataset.json (60 pairs) is single-chunk-derived, so it
cannot distinguish "found the one right chunk" from "successfully combined
information across the NPPF and the Local Plan" — which is this project's
central architectural hypothesis. This harness re-evaluates five already-
built retrieval pipelines against golden_dataset_multihop.json (30 pairs,
each requiring synthesis of an NPPF chunk AND a Local Plan chunk).

Architectures evaluated (reusing already-built indexes, no re-training):
  Exp02  — Hybrid, general BGE            (non-graph baseline)
  Exp04b — Hybrid, fine-tuned BGE v2       (non-graph, domain-adapted)
  Exp05  — Regex Graph RAG, FT-BGE v2      (graph-augmented)
  Exp07  — LLM Graph RAG, FT-BGE v2        (graph-augmented, typed triples)
  Exp08  — Hierarchical + LLM Graph (late fusion), FT-BGE v2

Hypothesis:
  On single-hop questions, all tuned architectures perform comparably.
  On multi-hop cross-document questions, only the graph-augmented
  architectures (Exp05, Exp07, Exp08) should meaningfully outperform the
  non-graph baselines (Exp02, Exp04b) — this is the evidence for why
  Graph RAG is necessary, which the single-hop golden set could not show.

Prerequisites:
  1. scripts/generate_multihop_dataset.py   (produces golden_dataset_multihop.json)
  2. Exp02, Exp04b, Exp05, Exp07, Exp08 must have already been run at least
     once, so their ChromaDB collections exist.

Outputs:
  outputs/eval_multihop_results.json
  outputs/eval_multihop_ragas_scores.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table

import chromadb

from src.auto_merger import auto_merge, build_chunk_lookup, hybrid_hierarchical_query
from src.config import get_settings, resolve_endpoint_config
from src.evaluator import (
    evaluate_citation_quality,
    load_golden_dataset,
    run_ragas_evaluation,
    save_ragas_scores,
)
from src.generator import answer_question
from src.graph_retriever import PolicyGraph, hybrid_graph_query, graph_expand
from src.indexer import get_embedding_model, load_chunks
from src.retriever import BM25Index, hybrid_query, reciprocal_rank_fusion, rerank

console = Console()

MULTIHOP_PATH   = Path(os.getenv("DATASET_PATH", "data/dataset")).parent / "golden_dataset_multihop.json"
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
CANDIDATE_POOL  = 20
GRAPH_MAX_EXP   = 40
MERGE_RATIO     = 0.5

FT_BGE_V2       = Path("outputs") / "bge-planning-finetuned-v2"
REGEX_GRAPH     = Path("outputs") / "policy_graph.json"
LLM_GRAPH       = Path("outputs") / "policy_graph_llm.json"

# Exp08 hierarchical collection/files (Config B)
HIER_COLLECTION  = "hier_tuned_b_512_v2"
HIER_CHUNKS_FILE = "chunks_hier_b_4096_1024_512.jsonl"
HIER_LEAVES_FILE = "chunks_hier_b_4096_1024_512_leaves.jsonl"
FLAT_COLLECTION  = "llm_graph_rag_finetuned_bge_v2"


def dense_retrieve(query_embedding, chroma_path, collection_name, n_results):
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)
    raw = collection.query(query_embeddings=[query_embedding], n_results=n_results,
                            include=["documents", "metadatas", "distances"])
    return [
        {"chunk_id": raw["ids"][0][i], "text": raw["documents"][0][i],
         "metadata": raw["metadatas"][0][i], "distance": raw["distances"][0][i]}
        for i in range(len(raw["ids"][0]))
    ]


def combined_query_v2(query, hier_bm25, hier_embed_model, hier_chunk_lookup, chroma_path,
                       flat_bm25, graph, flat_chunk_lookup, top_k, candidate_pool):
    query_embedding = hier_embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0].tolist()

    hier_bm25_results  = hier_bm25.retrieve(query, top_k=candidate_pool)
    hier_dense_results = dense_retrieve(query_embedding, chroma_path, HIER_COLLECTION, candidate_pool)
    hier_seed = reciprocal_rank_fusion([hier_dense_results, hier_bm25_results], k=60, top_n=candidate_pool)
    hier_merged = auto_merge(hier_seed, hier_chunk_lookup, merge_ratio=MERGE_RATIO)

    flat_bm25_results  = flat_bm25.retrieve(query, top_k=candidate_pool)
    flat_dense_results = dense_retrieve(query_embedding, chroma_path, FLAT_COLLECTION, candidate_pool)
    flat_seed = reciprocal_rank_fusion([flat_dense_results, flat_bm25_results], k=60, top_n=candidate_pool)
    seed_ids  = [r["chunk_id"] for r in flat_seed]
    expanded  = graph_expand(seed_chunk_ids=seed_ids, graph=graph, chunk_lookup=flat_chunk_lookup,
                              hops=2, max_expanded=GRAPH_MAX_EXP)
    graph_candidates = list(flat_seed)
    seen_flat = set(seed_ids)
    for chunk in expanded:
        if chunk["chunk_id"] not in seen_flat:
            graph_candidates.append(chunk)
            seen_flat.add(chunk["chunk_id"])

    seen_prefix, merged = set(), []
    for candidate in hier_merged + graph_candidates:
        prefix = candidate.get("text", "")[:80].strip()
        if prefix not in seen_prefix:
            seen_prefix.add(prefix)
            merged.append(candidate)

    return rerank(query, merged, model_name=RERANKER_MODEL, top_k=top_k)


def run_architecture(
    label: str,
    retrieve_fn,
    queries: list[str],
    settings,
) -> list[dict]:
    console.rule(f"[bold cyan]{label}[/bold cyan]")
    results = []
    for i, query in enumerate(queries, 1):
        if i % 10 == 0:
            console.print(f"  Query {i}/{len(queries)}...")
        matches = retrieve_fn(query)
        answer = ""
        if settings.llm_base_url and settings.llm_api_key:
            try:
                answer = answer_question(
                    question=query, matches=matches, model=settings.llm_model,
                    base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                    api_version=settings.llm_api_version,
                )
            except Exception as exc:
                console.print(f"[red]LLM call failed: {exc}[/red]")
        results.append({"query": query, "matches": matches, "answer": answer})
    return results


def print_final_table(all_scores: dict[str, dict]) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "citation_quality"]
    table = Table(title="Multi-Hop Cross-Document Evaluation — Architecture Comparison")
    table.add_column("Metric", style="bold")
    for label in all_scores:
        table.add_column(label, justify="right")

    def fmt(d, m):
        v = d.get(m)
        return f"{v:.4f}" if isinstance(v, float) else "—"

    for m in metrics:
        table.add_row(m, *[fmt(all_scores[label], m) for label in all_scores])
    console.print(table)


def main() -> None:
    settings     = get_settings()
    project_root = Path(__file__).resolve().parent.parent
    output_dir   = Path(settings.output_dir)

    console.print("[bold blue]Multi-Hop Cross-Document Evaluation[/bold blue]")

    if not MULTIHOP_PATH.exists():
        console.print(f"[red]Multi-hop set not found at {MULTIHOP_PATH}[/red]")
        console.print("[yellow]Run: python scripts/generate_multihop_dataset.py first[/yellow]")
        return

    multihop_dataset = load_golden_dataset(MULTIHOP_PATH)
    queries = [item["question"] for item in multihop_dataset]
    console.print(f"Loaded {len(queries)} multi-hop questions\n")

    ft_model_path = project_root / FT_BGE_V2
    if not ft_model_path.exists():
        console.print(f"[red]FT-BGE v2 not found at {ft_model_path}[/red]")
        return

    flat_chunks       = load_chunks(settings.chunks_path)
    flat_bm25         = BM25Index(flat_chunks)
    flat_chunk_lookup = build_chunk_lookup(settings.chunks_path)

    general_embed = get_embedding_model("BAAI/bge-large-en-v1.5", device="cpu")
    ft_embed      = get_embedding_model(str(ft_model_path), device="cpu")

    regex_graph = PolicyGraph(project_root / REGEX_GRAPH)
    llm_graph   = PolicyGraph(project_root / LLM_GRAPH)

    hier_chunks_path = output_dir / HIER_CHUNKS_FILE
    leaf_chunks_path = output_dir / HIER_LEAVES_FILE
    hier_chunk_lookup = build_chunk_lookup(hier_chunks_path)
    leaf_chunks       = load_chunks(leaf_chunks_path)
    hier_bm25         = BM25Index(leaf_chunks)

    architectures = {
        "Exp02\nHybrid": lambda q: hybrid_query(
            query=q, bm25_index=flat_bm25, embed_model=general_embed,
            chroma_path=settings.chroma_path, collection_name="hybrid_naive",
            reranker_model=RERANKER_MODEL, top_k=settings.top_k, candidate_pool=CANDIDATE_POOL,
        ),
        "Exp04b\nFT-BGE v2": lambda q: hybrid_query(
            query=q, bm25_index=flat_bm25, embed_model=ft_embed,
            chroma_path=settings.chroma_path, collection_name="finetuned_bge_hybrid_v2",
            reranker_model=RERANKER_MODEL, top_k=settings.top_k, candidate_pool=CANDIDATE_POOL,
        ),
        "Exp05\nRegex Graph": lambda q: hybrid_graph_query(
            query=q, bm25_index=flat_bm25, embed_model=ft_embed,
            chroma_path=settings.chroma_path, collection_name="graph_rag_finetuned_bge_v2",
            graph=regex_graph, chunk_lookup=flat_chunk_lookup, reranker_model=RERANKER_MODEL,
            top_k=settings.top_k, candidate_pool=CANDIDATE_POOL, graph_hops=2, graph_max_expanded=GRAPH_MAX_EXP,
        ),
        "Exp07\nLLM Graph": lambda q: hybrid_graph_query(
            query=q, bm25_index=flat_bm25, embed_model=ft_embed,
            chroma_path=settings.chroma_path, collection_name="llm_graph_rag_finetuned_bge_v2",
            graph=llm_graph, chunk_lookup=flat_chunk_lookup, reranker_model=RERANKER_MODEL,
            top_k=settings.top_k, candidate_pool=CANDIDATE_POOL, graph_hops=2, graph_max_expanded=GRAPH_MAX_EXP,
        ),
        "Exp08\nCombined": lambda q: combined_query_v2(
            query=q, hier_bm25=hier_bm25, hier_embed_model=ft_embed, hier_chunk_lookup=hier_chunk_lookup,
            chroma_path=settings.chroma_path, flat_bm25=flat_bm25, graph=llm_graph,
            flat_chunk_lookup=flat_chunk_lookup, top_k=settings.top_k, candidate_pool=CANDIDATE_POOL,
        ),
    }

    openai_base_url, openai_url_version = resolve_endpoint_config(os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key     = os.getenv("OPENAI_API_KEY", "")
    openai_api_version = os.getenv("OPENAI_API_VERSION", "") or openai_url_version

    all_results: dict[str, list[dict]] = {}
    all_scores: dict[str, dict] = {}

    for label, retrieve_fn in architectures.items():
        results = run_architecture(label, retrieve_fn, queries, settings)
        all_results[label] = results

        ragas_scores: dict = {}
        if openai_base_url and openai_api_key:
            try:
                ragas_scores = run_ragas_evaluation(
                    results=results, golden_dataset=multihop_dataset, eval_model=settings.eval_model,
                    openai_api_key=openai_api_key, openai_base_url=openai_base_url,
                    openai_api_version=openai_api_version,
                )
                citation_scores = evaluate_citation_quality(
                    results=results, eval_model=settings.eval_model, openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url, openai_api_version=openai_api_version,
                )
                ragas_scores["aggregate"]["citation_quality"] = citation_scores["aggregate"]["citation_quality"]
            except Exception as exc:
                console.print(f"[red]RAGAS failed for {label}: {exc}[/red]")
        all_scores[label.replace(chr(10), " ")] = ragas_scores.get("aggregate", {})

    output_path = output_dir / "eval_multihop_results.json"
    output_path.write_text(json.dumps({
        "multihop_dataset_size": len(multihop_dataset),
        "architectures": list(architectures.keys()),
        "results": all_results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved results to {output_path}[/green]")

    scores_path = output_dir / "eval_multihop_ragas_scores.json"
    save_ragas_scores(all_scores, scores_path)

    print_final_table(all_scores)


if __name__ == "__main__":
    main()
