"""
Diagnostic: for each multi-hop question, trace whether BOTH required source
chunks (a) exist in the candidate pool after graph expansion, BEFORE
reranking, and (b) survive into the final top-5 AFTER reranking.

This distinguishes two very different failure modes:
  - Graph expansion never finds the complementary chunk (retrieval failure)
  - Graph expansion finds it, but the cross-encoder reranker discards it
    (reranking bottleneck — the graph mechanism works, downstream doesn't
    trust it)

Run from project root: python scripts/diagnose_multihop.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from src.auto_merger import build_chunk_lookup
from src.config import get_settings
from src.graph_retriever import PolicyGraph, graph_expand, rerank_with_graph_reservation
from src.indexer import get_embedding_model, load_chunks
from src.retriever import BM25Index, reciprocal_rank_fusion, rerank

console = Console()

MULTIHOP_PATH  = Path("data") / "golden_dataset_multihop.json"
FT_MODEL       = Path("outputs") / "bge-planning-finetuned-v2"
LLM_GRAPH_PATH = Path("outputs") / "policy_graph_llm.json"
FLAT_COLLECTION = "llm_graph_rag_finetuned_bge_v2"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
CANDIDATE_POOL  = 20
GRAPH_MAX_EXP   = 40

# Final number of chunks kept after reranking — this is the exact lever being
# tested: does relaxing the cutoff let complementary graph-expanded chunks
# survive reranking? Override via: python scripts/diagnose_multihop.py 8
FINAL_TOP_K = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# Mode: "plain" (default) uses ordinary top-k reranking, same as every
# existing experiment. "reserved" uses rerank_with_graph_reservation() —
# reserves the final slot for the best graph-expanded candidate not otherwise
# selected. Run: python scripts/diagnose_multihop.py 5 reserved
MODE = sys.argv[2] if len(sys.argv) > 2 else "plain"


def dense_retrieve(query_embedding, chroma_path, collection_name, n_results):
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)
    raw = collection.query(query_embeddings=[query_embedding], n_results=n_results,
                            include=["documents", "metadatas", "distances"])
    return [
        {"chunk_id": raw["ids"][0][i], "text": raw["documents"][0][i],
         "metadata": raw["metadatas"][0][i], "distance": raw["distances"][0][i]}
        for i in range(len(raw["ids"][0]))
    ]


def main() -> None:
    settings = get_settings()
    multihop = json.loads(MULTIHOP_PATH.read_text(encoding="utf-8"))
    console.print(f"[bold]Diagnosing {len(multihop)} multi-hop questions against Exp07 (LLM Graph), final_top_k={FINAL_TOP_K}, mode={MODE}[/bold]\n")

    flat_chunks       = load_chunks(settings.chunks_path)
    flat_bm25         = BM25Index(flat_chunks)
    flat_chunk_lookup = build_chunk_lookup(settings.chunks_path)
    embed_model       = get_embedding_model(str(FT_MODEL), device="cpu")
    graph             = PolicyGraph(LLM_GRAPH_PATH)

    table = Table(title=f"Multi-Hop Retrieval Diagnosis (Exp07 pipeline, final_top_k={FINAL_TOP_K}, mode={MODE})")
    table.add_column("Q#", justify="right")
    table.add_column("Chunk A\nin seed?", justify="center")
    table.add_column("Chunk B\nin seed?", justify="center")
    table.add_column("Chunk A\nafter graph?", justify="center")
    table.add_column("Chunk B\nafter graph?", justify="center")
    table.add_column("Chunk A\nin final top-5?", justify="center")
    table.add_column("Chunk B\nin final top-5?", justify="center")
    table.add_column("Both required\nchunks in answer?", justify="center")

    both_in_seed = both_in_graph_pool = both_in_final = 0

    for i, item in enumerate(multihop, 1):
        query = item["question"]
        chunk_a, chunk_b = item["source_chunk_ids"]

        query_embedding = embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0].tolist()

        bm25_results  = flat_bm25.retrieve(query, top_k=CANDIDATE_POOL)
        dense_results = dense_retrieve(query_embedding, settings.chroma_path, FLAT_COLLECTION, CANDIDATE_POOL)
        seed = reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_n=CANDIDATE_POOL)
        seed_ids = {r["chunk_id"] for r in seed}

        a_in_seed = chunk_a in seed_ids
        b_in_seed = chunk_b in seed_ids

        expanded = graph_expand(
            seed_chunk_ids=list(seed_ids), graph=graph, chunk_lookup=flat_chunk_lookup,
            hops=2, max_expanded=GRAPH_MAX_EXP,
        )
        expanded_ids = {c["chunk_id"] for c in expanded}
        pool_ids = seed_ids | expanded_ids

        a_in_pool = chunk_a in pool_ids
        b_in_pool = chunk_b in pool_ids

        merged = list(seed)
        seen = set(seed_ids)
        for c in expanded:
            if c["chunk_id"] not in seen:
                merged.append(c)
                seen.add(c["chunk_id"])

        if MODE == "reserved":
            final = rerank_with_graph_reservation(
                query=query, merged_candidates=merged, graph_expanded_ids=expanded_ids,
                reranker_model=RERANKER_MODEL, top_k=FINAL_TOP_K,
            )
        else:
            final = rerank(query, merged, model_name=RERANKER_MODEL, top_k=FINAL_TOP_K)
        final_ids = {m["chunk_id"] for m in final}

        a_in_final = chunk_a in final_ids
        b_in_final = chunk_b in final_ids
        both_final = a_in_final and b_in_final

        if a_in_seed and b_in_seed:
            both_in_seed += 1
        if a_in_pool and b_in_pool:
            both_in_graph_pool += 1
        if both_final:
            both_in_final += 1

        def mark(x): return "[green]YES[/green]" if x else "[red]no[/red]"

        table.add_row(
            str(i), mark(a_in_seed), mark(b_in_seed),
            mark(a_in_pool), mark(b_in_pool),
            mark(a_in_final), mark(b_in_final),
            "[bold green]BOTH[/bold green]" if both_final else "",
        )

    console.print(table)
    n = len(multihop)
    console.print(f"\n[bold]Summary across {n} multi-hop questions:[/bold]")
    console.print(f"  Both chunks in initial BM25+Dense seed (pre-graph):  {both_in_seed}/{n}")
    console.print(f"  Both chunks in pool AFTER graph expansion:          {both_in_graph_pool}/{n}")
    console.print(f"  Both chunks survive into FINAL top-{FINAL_TOP_K} (post-rerank): {both_in_final}/{n}")
    console.print(f"\n[bold]Graph expansion net contribution: {both_in_graph_pool - both_in_seed} additional questions where both chunks became available[/bold]")
    console.print(f"[bold]Reranker survival rate: {both_in_final}/{both_in_graph_pool} of the questions where the graph found both chunks[/bold]")


if __name__ == "__main__":
    main()
