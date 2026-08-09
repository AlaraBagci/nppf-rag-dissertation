from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import fitz
from docx import Document
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from src.schemas import ChunkRecord


console = Console()
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def iter_documents(root: Path, max_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith(".") and "~$" not in path.name and "__MACOSX" not in str(path):
            files.append(path)
            if max_files is not None and len(files) >= max_files:
                break
    return files


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8", errors="ignore")).hexdigest()


def classify_path(file_path: Path, root: Path) -> dict[str, str | None]:
    relative_parts = file_path.relative_to(root).parts
    top_folder = relative_parts[0] if relative_parts else None
    parent_folder = file_path.parent.name if file_path.parent != root else None
    document_class: str | None = None
    region: str | None = None

    if top_folder == "general":
        document_class = "National Planning Policy"
        region = "UK"
    elif top_folder == "Coventry":
        region = "Coventry"
        if len(relative_parts) > 1 and relative_parts[1].startswith("1. Adopted Coventry Local Plan"):
            document_class = "Statutory Policy"
        elif len(relative_parts) > 1 and relative_parts[1].startswith("2. Supplementary Planning Documents"):
            document_class = "Supplementary Planning Document"
        else:
            document_class = "Coventry Planning Document"
    else:
        document_class = "Unknown"

    sub_topic = None
    if parent_folder and parent_folder not in {top_folder, root.name}:
        sub_topic = parent_folder

    return {
        "collection": top_folder or "unknown",
        "region": region,
        "top_folder": top_folder,
        "sub_topic": sub_topic,
        "document_class": document_class,
    }


def extract_pdf(path: Path) -> list[dict[str, str | int | None]]:
    page_records: list[dict[str, str | int | None]] = []
    try:
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    page_records.append({"page": page_number, "text": text})
    except Exception as exc:
        console.print(f"[yellow]Skipping unreadable PDF:[/yellow] {path.name} ({exc})")
        return []
    return page_records


def extract_docx(path: Path) -> list[dict[str, str | int | None]]:
    doc = Document(path)
    text = "\n".join(paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip())
    return [{"page": None, "text": text}] if text else []


def extract_document_text(path: Path) -> list[dict[str, str | int | None]]:
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    if path.suffix.lower() == ".docx":
        return extract_docx(path)
    return []


def chunk_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(root: Path, chunk_size: int, overlap: int, max_files: int | None = None) -> tuple[list[ChunkRecord], dict[str, int]]:
    files = iter_documents(root, max_files=max_files)
    seen_document_hashes: set[str] = set()
    chunk_records: list[ChunkRecord] = []
    stats = {
        "files_scanned": len(files),
        "files_deduplicated": 0,
        "files_with_text": 0,
        "files_without_text": 0,
        "chunks_created": 0,
    }

    for path in tqdm(files, desc="Processing documents"):
        raw_bytes = path.read_bytes()
        document_hash = sha256_bytes(raw_bytes)
        if document_hash in seen_document_hashes:
            stats["files_deduplicated"] += 1
            continue
        seen_document_hashes.add(document_hash)

        page_records = extract_document_text(path)
        if not page_records:
            stats["files_without_text"] += 1
            console.print(f"[yellow]No content extracted:[/yellow] {path.name}")
            continue

        stats["files_with_text"] += 1
        path_tags = classify_path(path, root)
        full_text = "\n\n".join(record["text"] for record in page_records if record["text"])
        pages = [record["page"] for record in page_records if record["page"] is not None]
        page_start = min(pages) if pages else None
        page_end = max(pages) if pages else None

        text_chunks = chunk_words(full_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_hash = sha256_text(f"{document_hash}:{chunk_index}:{chunk_text}")
            record = ChunkRecord(
                chunk_id=f"{document_hash[:12]}-{chunk_index:04d}",
                document_id=document_hash,
                source_path=str(path.relative_to(root)),
                collection=str(path_tags["collection"]),
                region=path_tags["region"],
                top_folder=path_tags["top_folder"],
                sub_topic=path_tags["sub_topic"],
                document_class=path_tags["document_class"],
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_index,
                text=chunk_text,
                hash=chunk_hash,
                metadata={
                    "path_tags": path_tags,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                },
            )
            chunk_records.append(record)
            stats["chunks_created"] += 1

    return chunk_records, stats


def write_chunks_jsonl(chunks: Iterable[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def iter_documents_raw(root: Path) -> list[Path]:
    """Collect ALL files with no filtering — includes duplicates, junk, and temp files."""
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]


def extract_pdf_raw(path: Path) -> list[dict]:
    """Extract PDF pages with no filtering — includes empty, whitespace-only, and junk pages."""
    records = []
    try:
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")  # no strip(), no empty check
                records.append({"page": page_number, "text": text})
    except Exception as exc:
        console.print(f"[yellow]Warning (raw):[/yellow] {path.name} ({exc})")
    return records


def build_chunks_raw(root: Path, chunk_size: int, overlap: int) -> tuple[list[ChunkRecord], dict[str, int]]:
    """Ingest with NO quality gates: no deduplication, no empty filtering, no junk removal."""
    files = iter_documents_raw(root)
    chunk_records: list[ChunkRecord] = []
    stats = {
        "files_scanned": len(files),
        "files_deduplicated": 0,   # always 0 — dedup disabled
        "files_with_text": 0,
        "files_without_text": 0,
        "chunks_created": 0,
    }

    #for path in tqdm(files, desc="Raw ingestion (no preprocessing)"):
    for doc_idx, path in enumerate(tqdm(files, desc="Raw ingestion (no preprocessing)")):
        raw_bytes = path.read_bytes()
        document_hash = sha256_bytes(raw_bytes)
        # ── No deduplication check ─────────────────────────────────────────

        if path.suffix.lower() == ".pdf":
            page_records = extract_pdf_raw(path)      # no empty-page filter
        else:
            page_records = extract_docx(path)

        if not page_records:
            stats["files_without_text"] += 1
            continue

        stats["files_with_text"] += 1
        path_tags = classify_path(path, root)
        full_text = "\n\n".join(r["text"] for r in page_records)  # no empty check
        pages = [r["page"] for r in page_records if r["page"] is not None]
        page_start = min(pages) if pages else None
        page_end   = max(pages) if pages else None

        # ── No minimum chunk length — creates chunks from headers, page numbers, whitespace ──
        text_chunks = chunk_words(full_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_hash = sha256_text(f"{document_hash}:{chunk_index}:{chunk_text}")
            record = ChunkRecord(
                #chunk_id=f"raw-{document_hash[:12]}-{chunk_index:04d}",
                chunk_id=f"raw-{doc_idx:04d}-{document_hash[:12]}-{chunk_index:04d}",
                document_id=document_hash,
                source_path=str(path.relative_to(root)),
                collection=str(path_tags["collection"]),
                region=path_tags["region"],
                top_folder=path_tags["top_folder"],
                sub_topic=path_tags["sub_topic"],
                document_class=path_tags["document_class"],
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_index,
                text=chunk_text,
                hash=chunk_hash,
                metadata={
                    "path_tags": path_tags,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "preprocessed": False,
                },
            )
            chunk_records.append(record)
            stats["chunks_created"] += 1

    return chunk_records, stats


def run_phase0_raw(dataset_path: str | Path, output_path: str | Path, chunk_size: int = 512, overlap: int = 64) -> dict[str, int | str]:
    """Run ingestion with NO preprocessing — for ablation study comparison."""
    root = Path(dataset_path)
    output_path = Path(output_path)
    chunks, stats = build_chunks_raw(root=root, chunk_size=chunk_size, overlap=overlap)
    write_chunks_jsonl(chunks, output_path)

    summary = {
        **stats,
        "dataset_path": str(root),
        "output_path": str(output_path),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "preprocessed": False,
    }

    table = Table(title="Phase 0 Raw Ingestion (No Preprocessing)")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)
    return summary


def build_chunks_hierarchical(
    root: Path,
    sizes: tuple[int, int, int] = (2048, 512, 128),
    max_files: int | None = None,
) -> tuple[list[ChunkRecord], dict[str, int]]:
    """
    Build a 3-level hierarchy of chunks per document:
      Level 0 (grandparent) : sizes[0] words  — rich context for the LLM
      Level 1 (parent)      : sizes[1] words  — intermediate grouping
      Level 2 (leaf)        : sizes[2] words  — embedded & retrieved

    Parent-child boundaries are exact (no overlap) so auto-merging is clean.
    All three levels are written to JSONL; only leaves are indexed in ChromaDB.
    """
    gp_size, parent_size, leaf_size = sizes
    files = iter_documents(root, max_files=max_files)
    seen_hashes: set[str] = set()
    all_records: list[ChunkRecord] = []
    stats = {
        "files_scanned": len(files),
        "files_deduplicated": 0,
        "files_with_text": 0,
        "files_without_text": 0,
        "grandparent_chunks": 0,
        "parent_chunks": 0,
        "leaf_chunks": 0,
    }

    for path in tqdm(files, desc="Hierarchical ingestion"):
        raw_bytes = path.read_bytes()
        doc_hash = sha256_bytes(raw_bytes)
        if doc_hash in seen_hashes:
            stats["files_deduplicated"] += 1
            continue
        seen_hashes.add(doc_hash)

        page_records = extract_document_text(path)
        if not page_records:
            stats["files_without_text"] += 1
            continue
        stats["files_with_text"] += 1

        path_tags = classify_path(path, root)
        full_text = "\n\n".join(r["text"] for r in page_records if r["text"])
        words = full_text.split()
        pages = [r["page"] for r in page_records if r["page"] is not None]
        page_start = min(pages) if pages else None
        page_end   = max(pages) if pages else None

        # ── Level 0: grandparent chunks (non-overlapping, gp_size words each) ──
        for gp_idx, gp_start in enumerate(range(0, len(words), gp_size)):
            gp_words = words[gp_start: gp_start + gp_size]
            if not gp_words:
                continue
            gp_text = " ".join(gp_words)
            gp_id   = f"{doc_hash[:12]}-G{gp_idx:04d}"
            parent_ids: list[str] = []

            # ── Level 1: parent chunks inside this grandparent ────────────────
            for p_idx, p_start in enumerate(range(0, len(gp_words), parent_size)):
                p_words = gp_words[p_start: p_start + parent_size]
                if not p_words:
                    continue
                p_text = " ".join(p_words)
                p_id   = f"{doc_hash[:12]}-G{gp_idx:04d}-P{p_idx:04d}"
                parent_ids.append(p_id)
                leaf_ids: list[str] = []

                # ── Level 2: leaf chunks inside this parent ────────────────────
                for l_idx, l_start in enumerate(range(0, len(p_words), leaf_size)):
                    l_words = p_words[l_start: l_start + leaf_size]
                    if not l_words:
                        continue
                    l_text = " ".join(l_words)
                    l_id   = f"{doc_hash[:12]}-G{gp_idx:04d}-P{p_idx:04d}-L{l_idx:04d}"
                    leaf_ids.append(l_id)
                    all_records.append(ChunkRecord(
                        chunk_id=l_id,
                        document_id=doc_hash,
                        source_path=str(path.relative_to(root)),
                        collection=str(path_tags["collection"]),
                        region=path_tags["region"],
                        top_folder=path_tags["top_folder"],
                        sub_topic=path_tags["sub_topic"],
                        document_class=path_tags["document_class"],
                        page_start=page_start,
                        page_end=page_end,
                        chunk_index=stats["leaf_chunks"],
                        text=l_text,
                        hash=sha256_text(l_id + l_text),
                        metadata={
                            "level": 2,
                            "parent_id": p_id,
                            "grandparent_id": gp_id,
                            "leaf_index": l_idx,
                            "path_tags": path_tags,
                        },
                    ))
                    stats["leaf_chunks"] += 1

                # Write parent record (stores its leaf IDs for auto-merging)
                all_records.append(ChunkRecord(
                    chunk_id=p_id,
                    document_id=doc_hash,
                    source_path=str(path.relative_to(root)),
                    collection=str(path_tags["collection"]),
                    region=path_tags["region"],
                    top_folder=path_tags["top_folder"],
                    sub_topic=path_tags["sub_topic"],
                    document_class=path_tags["document_class"],
                    page_start=page_start,
                    page_end=page_end,
                    chunk_index=stats["parent_chunks"],
                    text=p_text,
                    hash=sha256_text(p_id + p_text),
                    metadata={
                        "level": 1,
                        "parent_id": gp_id,
                        "grandparent_id": gp_id,
                        "child_ids": leaf_ids,
                        "path_tags": path_tags,
                    },
                ))
                stats["parent_chunks"] += 1

            # Write grandparent record
            all_records.append(ChunkRecord(
                chunk_id=gp_id,
                document_id=doc_hash,
                source_path=str(path.relative_to(root)),
                collection=str(path_tags["collection"]),
                region=path_tags["region"],
                top_folder=path_tags["top_folder"],
                sub_topic=path_tags["sub_topic"],
                document_class=path_tags["document_class"],
                page_start=page_start,
                page_end=page_end,
                chunk_index=stats["grandparent_chunks"],
                text=gp_text,
                hash=sha256_text(gp_id + gp_text),
                metadata={
                    "level": 0,
                    "parent_id": None,
                    "child_ids": parent_ids,
                    "path_tags": path_tags,
                },
            ))
            stats["grandparent_chunks"] += 1

    stats["chunks_created"] = stats["leaf_chunks"] + stats["parent_chunks"] + stats["grandparent_chunks"]
    return all_records, stats


def run_phase0_hierarchical(
    dataset_path: str | Path,
    output_path: str | Path,
    sizes: tuple[int, int, int] = (2048, 512, 128),
    max_files: int | None = None,
) -> dict[str, int | str]:
    """Hierarchical ingestion — produces a 3-level chunk tree for AutoMerging RAG."""
    root = Path(dataset_path)
    output_path = Path(output_path)
    chunks, stats = build_chunks_hierarchical(root=root, sizes=sizes, max_files=max_files)
    write_chunks_jsonl(chunks, output_path)

    summary = {**stats, "dataset_path": str(root), "output_path": str(output_path), "sizes": list(sizes)}
    table = Table(title="Hierarchical Ingestion Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)
    return summary


def run_phase0(dataset_path: str | Path, output_path: str | Path, chunk_size: int = 512, overlap: int = 64, max_files: int | None = None) -> dict[str, int | str]:
    root = Path(dataset_path)
    output_path = Path(output_path)
    chunks, stats = build_chunks(root=root, chunk_size=chunk_size, overlap=overlap, max_files=max_files)
    write_chunks_jsonl(chunks, output_path)

    summary = {
        **stats,
        "dataset_path": str(root),
        "output_path": str(output_path),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }

    table = Table(title="Phase 0 Ingestion Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)
    return summary