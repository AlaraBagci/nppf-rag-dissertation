"""
AutoMerging retrieval: replaces sets of retrieved sibling leaf chunks with
their parent chunk when enough siblings were retrieved together.

This restores the full policy context that gets split across chunk boundaries,
improving faithfulness and answer relevance.

Merge logic (applied bottom-up, two levels):
  Leaf → Parent   : if retrieved_leaves / total_leaves >= merge_ratio
  Parent → Grandparent : if merged_parents / total_parents >= merge_ratio
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.indexer import load_chunks


def build_chunk_lookup(chunks_path: str | Path) -> dict[str, dict]:
    """Load all chunks (all levels) into a dict keyed by chunk_id."""
    records = load_chunks(Path(chunks_path))
    return {r["chunk_id"]: r for r in records}


def _make_result(chunk: dict, source_results: list[dict]) -> dict:
    """Build a result dict from a parent/grandparent chunk after merging."""
    return {
        "chunk_id": chunk["chunk_id"],
        "text": chunk["text"],
        "metadata": {
            **{k: chunk.get(k) for k in [
                "document_id", "source_path", "collection", "region",
                "top_folder", "sub_topic", "document_class",
                "page_start", "page_end", "chunk_index",
            ]},
            **chunk.get("metadata", {}),
        },
        "distance": min((r.get("distance", 1.0) for r in source_results), default=1.0),
        "merged_from": [r["chunk_id"] for r in source_results],
        "merge_level": chunk["metadata"].get("level", -1),
    }


def auto_merge(
    leaf_results: list[dict],
    chunk_lookup: dict[str, dict],
    merge_ratio: float = 0.5,
) -> list[dict]:
    """
    Given a list of retrieved leaf chunks, auto-merge sibling groups
    upward when >= merge_ratio of a parent's children were retrieved.

    Returns a list of result dicts — some may now be parent or grandparent
    chunks, with longer text for the LLM to use as context.
    """
    # ── Pass 1: Leaf → Parent ─────────────────────────────────────────────
    by_parent: dict[str, list[dict]] = defaultdict(list)
    parentless: list[dict] = []

    for result in leaf_results:
        parent_id = result.get("metadata", {}).get("parent_id")
        if parent_id and parent_id in chunk_lookup:
            by_parent[parent_id].append(result)
        else:
            parentless.append(result)

    level1_results: list[dict] = list(parentless)
    merged_to_parent: set[str] = set()

    for parent_id, children in by_parent.items():
        parent_chunk = chunk_lookup[parent_id]
        total_children = len(parent_chunk.get("metadata", {}).get("child_ids", []))
        if total_children > 0 and len(children) / total_children >= merge_ratio:
            level1_results.append(_make_result(parent_chunk, children))
            merged_to_parent.add(parent_id)
        else:
            level1_results.extend(children)

    # ── Pass 2: Parent → Grandparent ──────────────────────────────────────
    by_grandparent: dict[str, list[dict]] = defaultdict(list)
    gp_parentless: list[dict] = []

    for result in level1_results:
        chunk_id = result["chunk_id"]
        chunk = chunk_lookup.get(chunk_id, {})
        gp_id = chunk.get("metadata", {}).get("parent_id") or result.get("metadata", {}).get("grandparent_id")
        if gp_id and gp_id in chunk_lookup and chunk_id in merged_to_parent:
            by_grandparent[gp_id].append(result)
        else:
            gp_parentless.append(result)

    final_results: list[dict] = list(gp_parentless)

    for gp_id, parents in by_grandparent.items():
        gp_chunk = chunk_lookup[gp_id]
        total_parents = len(gp_chunk.get("metadata", {}).get("child_ids", []))
        if total_parents > 0 and len(parents) / total_parents >= merge_ratio:
            final_results.append(_make_result(gp_chunk, parents))
        else:
            final_results.extend(parents)

    return final_results


def hybrid_hierarchical_query(
    query: str,
    bm25_index,
    embed_model,
    chroma_path: str | Path,
    collection_name: str,
    chunk_lookup: dict[str, dict],
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    top_k: int = 5,
    candidate_pool: int = 30,
    merge_ratio: float = 0.5,
) -> list[dict]:
    """
    Hybrid-Hierarchical retrieval pipeline (Option B):
      1. BM25 sparse retrieval on leaf chunks
      2. Dense vector retrieval on leaf chunks (ChromaDB)
      3. RRF fusion of both ranked leaf lists
      4. AutoMerge: sibling leaves → parents → grandparents
      5. Cross-encoder reranking of merged results
    """
    import chromadb
    from src.retriever import reciprocal_rank_fusion, rerank

    # ── 1. BM25 on leaves ────────────────────────────────────────────────
    bm25_results = bm25_index.retrieve(query, top_k=candidate_pool)

    # ── 2. Dense retrieval on leaves ──────────────────────────────────────
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)
    query_embedding = embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0].tolist()
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_pool,
        include=["documents", "metadatas", "distances"],
    )
    dense_results = [
        {
            "chunk_id": raw["ids"][0][i],
            "text": raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
            "distance": raw["distances"][0][i],
        }
        for i in range(len(raw["ids"][0]))
    ]

    # ── 3. RRF fusion of leaf results ─────────────────────────────────────
    fused_leaves = reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_n=candidate_pool)

    # ── 4. AutoMerge: promote sibling leaves to parent/grandparent ────────
    merged = auto_merge(fused_leaves, chunk_lookup, merge_ratio=merge_ratio)

    # ── 5. Cross-encoder reranking of merged (potentially longer) chunks ──
    reranked = rerank(query, merged, model_name=reranker_model, top_k=top_k)

    return reranked
