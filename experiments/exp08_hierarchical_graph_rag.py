"""
Experiment 08 — Hierarchical + LLM Graph RAG (Combined Architecture)
=====================================================================
The final and most complete architecture, combining the two best individual
techniques from the experiment series:

  Hierarchical RAG (exp06 Config B — best hierarchical setup):
    Retrieves leaves, then auto-merges sibling groups into parents/grandparents.
    Ensures the LLM receives complete policy clause context (not truncated fragments).
    Hierarchy: 4096w → 1024w → 512w leaves, FT-BGE embeddings.

  LLM Graph RAG (exp07 — best graph setup):
    Expands seed retrieval via GPT-4.1-typed triples (IMPLEMENTS, REQUIRES, etc.)
    Ensures cross-document NPPF ↔ Local Plan connections are captured with
    semantic precision, not just keyword co-occurrence.
    Graph: policy_graph_llm.json, FT-BGE embeddings.

Combination strategy — late fusion:
  Both pipelines run independently on their own pre-built indexes.
  Their result sets are merged and cross-encoder reranked together.
  This is the simplest, most robust combination approach — it does not
  require rebuilding the graph on hierarchical chunk IDs, and it lets
  the reranker arbitrate between context-rich merged chunks and
  graph-expanded cross-document chunks.

  Pipeline:
    A. Hierarchical path:
       BM25(leaves) + Dense(leaves) → RRF → AutoMerge → candidate set A
    B. Graph path:
       BM25(flat) + Dense(flat) → RRF → LLM graph expand → candidate set B
    C. Merge A ∪ B (deduplicate by content prefix)
    D. BGE cross-encoder reranks merged set → top_k final context

Why this should improve over both alone:
  - Exp06 Config B (hierarchical): retrieves complete policy clauses but misses
    cross-document NPPF connections
  - Exp07 (LLM graph): finds NPPF paragraphs but may retrieve short flat chunks
    without surrounding clause context
  - Combined: the reranker can select merged parents (full clause context) from
    the hierarchical path AND NPPF cross-document chunks from the graph path

Hypothesis:
  Best context_recall across all experiments, because the graph path recovers
  NPPF evidence and the hierarchical path ensures Local Plan chunks have
  sufficient context for RAGAS faithfulness scoring.

Prerequisites:
  1. python -m experiments.exp06_hierarchical_tuned  (Config B chunk files + index)
  2. python -m experiments.exp07_llm_graph_rag       (LLM graph + flat index)
  3. Both require FT-BGE: python scripts/finetune_embeddings.py

Outputs:
  outputs/exp08_hierarchical_graph_rag_results.json
  outputs/exp08_hierarchical_graph_rag_ragas_scores.json
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

from src.auto_merger import auto_merge, build_chunk_lookup
from src.config import get_settings, resolve_endpoint_config
from src.evaluator import (
    evaluate_citation_quality,
    load_golden_dataset,
    run_ragas_evaluation,
    save_ragas_scores,
)
from src.generator import answer_question
from src.graph_retriever import PolicyGraph, graph_expand
from src.indexer import get_embedding_model, load_chunks
from src.ingest import run_phase0, run_phase0_hierarchical
from src.retriever import BM25Index, reciprocal_rank_fusion, rerank

console = Console()

# ── Config B hierarchy (best from exp06) ──────────────────────────────────────
HIER_SIZES       = (4096, 1024, 512)
HIER_COLLECTION  = "hier_tuned_b_512"
HIER_CHUNKS_FILE = "chunks_hier_b_4096_1024_512.jsonl"
HIER_LEAVES_FILE = "chunks_hier_b_4096_1024_512_leaves.jsonl"

# ── LLM graph (from exp07) ────────────────────────────────────────────────────
FLAT_COLLECTION  = "llm_graph_rag_finetuned_bge"
LLM_GRAPH_PATH   = Path("outputs") / "policy_graph_llm.json"

# ── Shared ────────────────────────────────────────────────────────────────────
FINETUNED_BGE    = Path("outputs") / "bge-planning-finetuned"
RERANKER_MODEL   = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
CANDIDATE_POOL   = int(os.getenv("CANDIDATE_POOL", "20"))
MERGE_RATIO      = 0.5
GRAPH_MAX_EXP    = 40

PRIOR = {
    "exp01":  {"faithfulness": 0.8788, "answer_relevancy": 0.7904,
               "context_precision": 0.6475, "context_recall": 0.6883, "citation_quality": 0.9320},
    "exp02":  {"faithfulness": 0.8981, "answer_relevancy": 0.9115,
               "context_precision": 0.9481, "context_recall": 0.8508, "citation_quality": 0.9520},
    "exp04b": {"faithfulness": 0.9022, "answer_relevancy": 0.9111,
               "context_precision": 0.9459, "context_recall": 0.8608, "citation_quality": 0.9520},
    "exp05":  {"faithfulness": 0.9062, "answer_relevancy": 0.9109,
               "context_precision": 0.9477, "context_recall": 0.8758, "citation_quality": 0.9330},
    "exp07":  {"faithfulness": 0.9091, "answer_relevancy": 0.9117,
               "context_precision": 0.9538, "context_recall": 0.8858, "citation_quality": 0.9430},
}


def dense_retrieve(
    query_embedding: list[float],
    chroma_path: Path,
    collection_name: str,
    n_results: int,
) -> list[dict]:
    client     = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": raw["ids"][0][i],
            "text":     raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
            "distance": raw["distances"][0][i],
        }
        for i in range(len(raw["ids"][0]))
    ]


def combined_query(
    query: str,
    # Hierarchical path
    hier_bm25: BM25Index,
    hier_embed_model,
    hier_chunk_lookup: dict[str, dict],
    chroma_path: Path,
    # Graph path
    flat_bm25: BM25Index,
    graph: PolicyGraph,
    flat_chunk_lookup: dict[str, dict],
    # Shared
    reranker_model: str,
    top_k: int,
    candidate_pool: int,
    merge_ratio: float,
    graph_max_expanded: int,
    settings,
) -> list[dict]:
    """
    Late-fusion hierarchical + LLM graph retrieval.

    Path A — hierarchical:
      BM25(leaves) + Dense(leaves) → RRF → AutoMerge → candidate set A

    Path B — LLM graph:
      BM25(flat) + Dense(flat) → RRF → LLM graph expand → candidate set B

    Merge A ∪ B (deduplicate by content prefix) → cross-encoder rerank → top_k
    """
    query_embedding = hier_embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0].tolist()

    # ── Path A: Hierarchical ──────────────────────────────────────────────────
    hier_bm25_results   = hier_bm25.retrieve(query, top_k=candidate_pool)
    hier_dense_results  = dense_retrieve(
        query_embedding, chroma_path, HIER_COLLECTION, candidate_pool
    )
    hier_seed = reciprocal_rank_fusion(
        [hier_dense_results, hier_bm25_results], k=60, top_n=candidate_pool
    )
    hier_merged = auto_merge(hier_seed, hier_chunk_lookup, merge_ratio=merge_ratio)

    # ── Path B: LLM Graph ─────────────────────────────────────────────────────
    flat_bm25_results  = flat_bm25.retrieve(query, top_k=candidate_pool)
    flat_dense_results = dense_retrieve(
        query_embedding, chroma_path, FLAT_COLLECTION, candidate_pool
    )
    flat_seed = reciprocal_rank_fusion(
        [flat_dense_results, flat_bm25_results], k=60, top_n=candidate_pool
    )
    seed_ids  = [r["chunk_id"] for r in flat_seed]
    expanded  = graph_expand(
        seed_chunk_ids=seed_ids,
        graph=graph,
        chunk_lookup=flat_chunk_lookup,
        hops=2,
        max_expanded=graph_max_expanded,
    )
    graph_candidates = list(flat_seed)
    seen_flat = set(seed_ids)
    for chunk in expanded:
        if chunk["chunk_id"] not in seen_flat:
            graph_candidates.append(chunk)
            seen_flat.add(chunk["chunk_id"])

    # ── Late fusion: merge A ∪ B, deduplicate by text prefix ─────────────────
    seen_text_prefix: set[str] = set()
    merged: list[dict] = []

    for candidate in hier_merged + graph_candidates:
        prefix = candidate.get("text", "")[:80].strip()
        if prefix not in seen_text_prefix:
            seen_text_prefix.add(prefix)
            merged.append(candidate)

    # ── Cross-encoder rerank → top_k ─────────────────────────────────────────
    return rerank(query, merged, model_name=reranker_model, top_k=top_k)


def print_comparison(exp08_aggregate: dict) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "citation_quality"]

    table = Table(title="Final Comparison — All Architectures")
    table.add_column("Metric",           style="bold")
    table.add_column("Exp01\nBaseline",  justify="right")
    table.add_column("Exp02\nHybrid",    justify="right")
    table.add_column("Exp04b\nFT-BGE",  justify="right")
    table.add_column("Exp05\nRegex\nGraph", justify="right")
    table.add_column("Exp07\nLLM\nGraph", justify="right")
    table.add_column("Exp08\nHier+Graph", justify="right", style="bold green")

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
            fmt(PRIOR["exp07"],  m),
            fmt(exp08_aggregate, m),
        )
    console.print(table)


def main() -> None:
    settings     = get_settings()
    project_root = Path(__file__).resolve().parent.parent
    output_dir   = Path(settings.output_dir)

    console.print("[bold blue]Experiment 08: Hierarchical + LLM Graph RAG (Combined)[/bold blue]")
    console.print("  Path A : BM25 + FT-BGE → RRF → AutoMerge (4096/1024/512w)")
    console.print("  Path B : BM25 + FT-BGE → RRF → LLM graph expand")
    console.print("  Fusion : Late fusion → BGE cross-encoder rerank")

    # ── Resolve model path ────────────────────────────────────────────────────
    ft_model_path = project_root / FINETUNED_BGE
    if not ft_model_path.exists():
        console.print(f"[red]Fine-tuned BGE not found. Run scripts/finetune_embeddings.py[/red]")
        return

    # ── Check hierarchical chunks exist (from exp06 Config B) ────────────────
    hier_chunks_path = output_dir / HIER_CHUNKS_FILE
    leaf_chunks_path = output_dir / HIER_LEAVES_FILE

    if not hier_chunks_path.exists():
        console.print(f"[yellow]Hierarchical chunks not found — building now (4096/1024/512w)...[/yellow]")
        run_phase0_hierarchical(
            dataset_path=settings.dataset_path,
            output_path=hier_chunks_path,
            sizes=HIER_SIZES,
            max_files=settings.max_files,
        )

    if not leaf_chunks_path.exists():
        console.print("[yellow]Filtering leaf chunks...[/yellow]")
        with hier_chunks_path.open("r", encoding="utf-8") as src, \
             leaf_chunks_path.open("w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    chunk = json.loads(line)
                    if chunk.get("metadata", {}).get("level") == 2:
                        dst.write(line)

    # ── Check LLM graph exists (from exp07) ───────────────────────────────────
    llm_graph_path = project_root / LLM_GRAPH_PATH
    if not llm_graph_path.exists():
        console.print(f"[red]LLM graph not found. Run scripts/build_graph_llm.py first.[/red]")
        return

    # ── Check flat chunks + ChromaDB collections exist ────────────────────────
    if not settings.chunks_path.exists():
        console.print("[yellow]Flat chunks not found — running ingestion.[/yellow]")
        run_phase0(
            dataset_path=settings.dataset_path,
            output_path=settings.chunks_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            max_files=settings.max_files,
        )

    # Verify required ChromaDB collections exist
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_path))
    existing_collections = {c.name for c in chroma_client.list_collections()}
    if HIER_COLLECTION not in existing_collections:
        console.print(f"[red]ChromaDB collection '{HIER_COLLECTION}' not found.[/red]")
        console.print("[yellow]Run: python -m experiments.exp06_hierarchical_tuned first[/yellow]")
        return
    if FLAT_COLLECTION not in existing_collections:
        console.print(f"[red]ChromaDB collection '{FLAT_COLLECTION}' not found.[/red]")
        console.print("[yellow]Run: python -m experiments.exp07_llm_graph_rag first[/yellow]")
        return

    # ── Load everything ───────────────────────────────────────────────────────
    console.print("[bold]Loading indexes and graph...[/bold]")

    leaf_chunks      = load_chunks(leaf_chunks_path)
    flat_chunks      = load_chunks(settings.chunks_path)
    hier_chunk_lookup = build_chunk_lookup(hier_chunks_path)
    flat_chunk_lookup = build_chunk_lookup(settings.chunks_path)
    graph             = PolicyGraph(llm_graph_path)

    hier_bm25  = BM25Index(leaf_chunks)
    flat_bm25  = BM25Index(flat_chunks)
    embed_model = get_embedding_model(str(ft_model_path), device="cpu")

    llm_edges = sum(1 for e in graph.edges if e["relation"].startswith("LLM_"))
    console.print(f"  Hierarchical leaves : {len(leaf_chunks)}")
    console.print(f"  Flat chunks         : {len(flat_chunks)}")
    console.print(f"  LLM graph edges     : {llm_edges}")

    # ── Golden dataset ────────────────────────────────────────────────────────
    golden_path    = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    if not golden_dataset:
        console.print("[red]No golden dataset found.[/red]")
        return
    queries = [item["question"] for item in golden_dataset]
    console.print(f"Running {len(queries)} queries...")

    # ── Retrieval + Generation ────────────────────────────────────────────────
    results: list[dict] = []
    for i, query in enumerate(queries, 1):
        if i % 10 == 0:
            console.print(f"  Query {i}/{len(queries)}...")

        matches = combined_query(
            query=query,
            hier_bm25=hier_bm25,
            hier_embed_model=embed_model,
            hier_chunk_lookup=hier_chunk_lookup,
            chroma_path=settings.chroma_path,
            flat_bm25=flat_bm25,
            graph=graph,
            flat_chunk_lookup=flat_chunk_lookup,
            reranker_model=RERANKER_MODEL,
            top_k=settings.top_k,
            candidate_pool=CANDIDATE_POOL,
            merge_ratio=MERGE_RATIO,
            graph_max_expanded=GRAPH_MAX_EXP,
            settings=settings,
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

        # Track which path contributed each result
        hier_ids = {c["chunk_id"] for c in hier_chunk_lookup.values()
                    if c.get("metadata", {}).get("level", 2) == 2}
        path_a_count = sum(1 for m in matches if m["chunk_id"] in hier_ids)
        path_b_count = len(matches) - path_a_count

        results.append({
            "query":        query,
            "matches":      matches,
            "answer":       answer,
            "path_a_count": path_a_count,
            "path_b_count": path_b_count,
        })

    # Path contribution stats
    avg_a = sum(r["path_a_count"] for r in results) / len(results)
    avg_b = sum(r["path_b_count"] for r in results) / len(results)
    console.print(f"[green]Average final chunks from hierarchical path: {avg_a:.1f}[/green]")
    console.print(f"[green]Average final chunks from graph path: {avg_b:.1f}[/green]")

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

            ragas_path = output_dir / "exp08_hierarchical_graph_rag_ragas_scores.json"
            save_ragas_scores(ragas_scores, ragas_path)
            console.print(f"[green]RAGAS scores saved to {ragas_path}[/green]")
        except Exception as exc:
            console.print(f"[red]RAGAS failed: {exc}[/red]")

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = output_dir / "exp08_hierarchical_graph_rag_results.json"
    payload = {
        "experiment":  "exp08_hierarchical_graph_rag",
        "description": "Late fusion: Hierarchical (4096/1024/512w FT-BGE) + LLM Graph RAG",
        "retrieval_config": {
            "path_a": f"BM25 + FT-BGE → RRF → AutoMerge {HIER_SIZES} (merge_ratio={MERGE_RATIO})",
            "path_b": f"BM25 + FT-BGE → RRF → LLM graph expand (max={GRAPH_MAX_EXP})",
            "fusion": "Late fusion → BGE cross-encoder reranking",
            "reranker":     RERANKER_MODEL,
            "candidate_pool": CANDIDATE_POOL,
            "final_top_k":  settings.top_k,
        },
        "path_contribution": {"avg_hier": avg_a, "avg_graph": avg_b},
        "results":      results,
        "ragas_scores": ragas_scores,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved results to {output_path}[/green]")

    if ragas_scores.get("aggregate"):
        print_comparison(ragas_scores["aggregate"])
    else:
        console.print("[yellow]No RAGAS scores — check API credentials.[/yellow]")


if __name__ == "__main__":
    main()
