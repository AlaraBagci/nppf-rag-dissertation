"""
Generate a fine-tuning training set that is strictly disjoint from the golden
evaluation dataset.

Why this script exists:
  scripts/finetune_embeddings.py previously trained the embedding model on
  data/golden_dataset.json — the SAME 60 Q&A pairs used to evaluate every
  RAGAS experiment from exp04b onward. That is train/test leakage: the model
  was directly taught (question, correct_chunk) pairs for the exact questions
  it was later "tested" on, which inflates every downstream result built on
  bge-planning-finetuned (exp04b, exp05, exp06a, exp06b, exp07, exp08).

  This script generates a larger, independent set of (question, chunk) pairs
  for training, explicitly excluding every chunk_id used as a source_chunk_id
  in the golden dataset. finetune_embeddings.py additionally asserts zero
  overlap before training, so this cannot silently regress.

Output: data/finetune_train_pairs.json
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import random
import sys
import time
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import track

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import resolve_endpoint_config

console = Console()

# ── Config ──────────────────────────────────────────────────────────────────
CHUNKS_PATH  = Path(os.getenv("CHUNKS_PATH", "outputs/chunks.jsonl"))
GOLDEN_PATH  = Path(os.getenv("DATASET_PATH", "data/dataset")).parent / "golden_dataset.json"
OUTPUT_PATH  = Path(os.getenv("DATASET_PATH", "data/dataset")).parent / "finetune_train_pairs.json"

OPENAI_BASE_URL    = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "")
EVAL_MODEL         = os.getenv("EVAL_MODEL", "gpt-4.1")

TARGET_PAIRS    = 300   # training pairs — much larger than the 60 golden pairs
SAMPLE_PER_RUN  = 420   # oversample to allow for skip / rejection rate
MIN_CHUNK_WORDS = 80
SLEEP_BETWEEN   = 0.6


SYSTEM_PROMPT = """You are creating training data for fine-tuning a sentence embedding model on UK planning policy text.

Given a chunk of text from a planning document, generate ONE natural-language question that this chunk directly and completely answers.

Rules:
- The question must be answerable solely from the provided chunk.
- The question must be specific (include policy codes, document names, or numbers where present) — avoid generic phrasing.
- Do NOT ask vague questions like "What does this document say?".
- If the chunk is too administrative (e.g. just an adoption statement, a table of contents, or a page header), reply with {"skip": true}.

Reply ONLY with valid JSON in one of these two formats:
{"question": "..."}
{"skip": true}"""


def build_client() -> OpenAI:
    if not OPENAI_BASE_URL or not OPENAI_API_KEY:
        raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY must be set in .env")
    resolved_base_url, resolved_url_version = resolve_endpoint_config(OPENAI_BASE_URL)
    resolved_api_version = OPENAI_API_VERSION or resolved_url_version
    kwargs: dict = {
        "base_url": resolved_base_url,
        "api_key": OPENAI_API_KEY,
        "default_headers": {"api-key": OPENAI_API_KEY},
    }
    if resolved_api_version:
        kwargs["default_query"] = {"api-version": resolved_api_version}
    return OpenAI(**kwargs)


def load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_excluded_chunk_ids() -> set[str]:
    """Every chunk_id used as a source for the golden evaluation set — must never appear in training."""
    if not GOLDEN_PATH.exists():
        console.print(f"[red]Golden dataset not found at {GOLDEN_PATH} — cannot verify exclusion. Aborting.[/red]")
        raise SystemExit(1)
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    excluded = {item["source_chunk_id"] for item in golden if item.get("source_chunk_id")}
    console.print(f"[bold]Excluding {len(excluded)} chunk IDs already used in the golden evaluation set[/bold]")
    return excluded


def sample_chunks(chunks: list[dict], n: int, excluded_ids: set[str]) -> list[dict]:
    """Sample diverse chunks for training, strictly disjoint from the golden evaluation set."""
    filtered = [
        c for c in chunks
        if c["chunk_id"] not in excluded_ids
        and len(c.get("text", "").split()) >= MIN_CHUNK_WORDS
        and c.get("document_class") not in {"Unknown"}
        and "adoption statement" not in c.get("text", "").lower()[:100]
    ]
    console.print(f"Candidate pool after exclusion + filtering: {len(filtered)} chunks")

    general  = [c for c in filtered if c.get("collection") == "general"]
    coventry = [c for c in filtered if c.get("collection") == "Coventry"]

    general_n  = min(len(general),  n // 3)
    coventry_n = min(len(coventry), n - general_n)

    sampled = random.sample(general, general_n) + random.sample(coventry, coventry_n)
    random.shuffle(sampled)
    return sampled[:n]


def generate_question(client: OpenAI, chunk: dict) -> dict | None:
    text = chunk["text"][:3000]
    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Document class: {chunk.get('document_class')}\nRegion: {chunk.get('region')}\n\nChunk:\n{text}"},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        if parsed.get("skip"):
            return None
        return {
            "question":        parsed["question"],
            "source_chunk_id": chunk["chunk_id"],
            "document_class":  chunk.get("document_class"),
            "region":          chunk.get("region"),
        }
    except Exception as e:
        console.print(f"[yellow]Skipped chunk {chunk.get('chunk_id')}: {e}[/yellow]")
        return None


def main() -> None:
    console.print("[bold blue]Fine-Tuning Training Set Generator (leak-free)[/bold blue]")
    console.print(f"Model : {EVAL_MODEL}  |  Target: {TARGET_PAIRS} pairs")
    console.print(f"Output: {OUTPUT_PATH}\n")

    excluded_ids = load_excluded_chunk_ids()

    client = build_client()
    chunks = load_chunks()
    console.print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    sampled = sample_chunks(chunks, SAMPLE_PER_RUN, excluded_ids)
    console.print(f"Sampled {len(sampled)} diverse chunks for generation\n")

    pairs: list[dict] = []
    for chunk in track(sampled, description="Generating training questions..."):
        if len(pairs) >= TARGET_PAIRS:
            break
        pair = generate_question(client, chunk)
        if pair:
            pairs.append(pair)
        time.sleep(SLEEP_BETWEEN)

    # ── Final safety check before writing to disk ────────────────────────────
    overlap = {p["source_chunk_id"] for p in pairs} & excluded_ids
    if overlap:
        console.print(f"[red]FATAL: {len(overlap)} training pairs overlap with the golden dataset. Aborting write.[/red]")
        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"\n[green]Done. Generated {len(pairs)} leak-free training pairs → {OUTPUT_PATH}[/green]")
    console.print(f"[green]Verified zero overlap with {len(excluded_ids)} golden evaluation chunk IDs.[/green]")
    console.print("[dim]Next: python scripts/finetune_embeddings.py[/dim]")


if __name__ == "__main__":
    main()
