"""
Hybrid retrieval module: BM25 + dense vector search fused with RRF,
then reranked with a cross-encoder.
"""
from __future__ import annotations

from pathlib import Path

import chromadb
import torch
from rank_bm25 import BM25Okapi
from rich.console import Console
from sentence_transformers import CrossEncoder

from src.indexer import get_embedding_model

console = Console()


# ── BM25 ─────────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        console.print(f"[green]BM25 index built over {len(chunks)} chunks[/green]")

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    k: chunk.get(k)
                    for k in ["document_id", "source_path", "collection", "region",
                               "top_folder", "sub_topic", "document_class",
                               "page_start", "page_end", "chunk_index"]
                },
                "distance": 0.0,
                "bm25_score": float(scores[idx]),
            })
        return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_n: int = 20,
) -> list[dict]:
    """
    Merge multiple ranked result lists using RRF.
    Score for document d = sum over lists of 1 / (k + rank(d)).
    """
    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_store:
                chunk_store[cid] = result

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_n]
    return [{**chunk_store[cid], "rrf_score": rrf_scores[cid]} for cid in sorted_ids]


# ── Cross-Encoder Reranking ───────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[dict],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_k: int = 5,
) -> list[dict]:
    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Reranking {len(candidates)} candidates with {model_name} on {resolved_device}[/dim]")
    reranker = CrossEncoder(model_name, device=resolved_device)
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
    return [{**candidate, "rerank_score": score} for candidate, score in ranked[:top_k]]


# ── Main hybrid query entry point ─────────────────────────────────────────────

def hybrid_query(
    query: str,
    bm25_index: BM25Index,
    embed_model,
    chroma_path: str | Path,
    collection_name: str,
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual",
    top_k: int = 5,
    candidate_pool: int = 20,
    rrf_k: int = 60,
) -> list[dict]:
    # 1. BM25 sparse retrieval
    bm25_results = bm25_index.retrieve(query, top_k=candidate_pool)

    # 2. Dense retrieval from ChromaDB
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

    # 3. Fuse with Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=rrf_k, top_n=candidate_pool)

    # 4. Cross-encoder reranking
    reranked = rerank(query, fused, model_name=reranker_model, top_k=top_k)

    return reranked
