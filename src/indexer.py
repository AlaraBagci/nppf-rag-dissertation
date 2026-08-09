from __future__ import annotations

import json
from pathlib import Path

import chromadb
import torch
from rich.console import Console
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


console = Console()


def load_chunks(chunks_path: Path) -> list[dict]:
    records: list[dict] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def get_embedding_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold]Loading embeddings on {resolved_device}[/bold] using {model_name}")
    return SentenceTransformer(model_name, device=resolved_device)


def build_index(chunks_path: str | Path, chroma_path: str | Path, collection_name: str = "baseline_naive", model_name: str = "BAAI/bge-large-en-v1.5", max_chunks: int | None = None) -> dict[str, int | str]:
    chunks_path = Path(chunks_path)
    chroma_path = Path(chroma_path)
    chunks = load_chunks(chunks_path)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    console.print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    client = chromadb.PersistentClient(path=str(chroma_path))
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        count = client.get_collection(collection_name).count()
        if count == len(chunks):
            console.print(f"[green]Index '{collection_name}' already exists with {count} chunks — skipping re-index.[/green]")
            return {"chunks_indexed": count, "collection_name": collection_name, "chroma_path": str(chroma_path), "model_name": model_name}
        console.print(f"[yellow]Index exists but chunk count differs ({count} vs {len(chunks)}) — rebuilding.[/yellow]")
        client.delete_collection(collection_name)

    model = get_embedding_model(model_name)
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    batch_size = 64
    total_inserted = 0
    for start in tqdm(range(0, len(chunks), batch_size), desc="Indexing chunks"):
        batch = chunks[start:start + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[{
                "document_id": item["document_id"],
                "source_path": item["source_path"],
                "collection": item["collection"],
                "region": item["region"],
                "top_folder": item["top_folder"],
                "sub_topic": item["sub_topic"],
                "document_class": item["document_class"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "chunk_index": item["chunk_index"],
            } for item in batch],
        )
        total_inserted += len(batch)

    summary = {
        "chunks_indexed": total_inserted,
        "collection_name": collection_name,
        "chroma_path": str(chroma_path),
        "model_name": model_name,
    }
    console.print(summary)
    return summary


def query_index(query: str, chroma_path: str | Path, collection_name: str, model_name: str, top_k: int = 5, device: str = "cpu") -> list[dict]:
    model = get_embedding_model(model_name, device=device)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)
    query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].tolist()
    result = collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
    matches: list[dict] = []
    for index, chunk_id in enumerate(result["ids"][0]):
        matches.append({
            "chunk_id": chunk_id,
            "text": result["documents"][0][index],
            "metadata": result["metadatas"][0][index],
            "distance": result["distances"][0][index],
        })
    return matches
