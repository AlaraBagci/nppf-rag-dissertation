from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import random
import time
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import track

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import resolve_endpoint_config

console = Console()

# ── Config ────────────────────────────────────────────────────────────
CHUNKS_PATH   = Path(os.getenv("CHUNKS_PATH", "outputs/chunks.jsonl"))
OUTPUT_PATH   = Path(os.getenv("DATASET_PATH", "data/dataset")).parent / "golden_dataset.json"
OPENAI_BASE_URL    = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "")
EVAL_MODEL         = os.getenv("EVAL_MODEL", "gpt-4.1")

TARGET_PAIRS   = 100  # how many Q&A pairs to generate
SAMPLE_PER_RUN = 130  # chunks to sample (more than target to allow rejections)
MIN_CHUNK_WORDS = 80  # skip very short chunks (headers, adoption statements, etc.)
SLEEP_BETWEEN  = 1.0  # seconds between API calls to avoid rate limits


SYSTEM_PROMPT = """You are a UK planning policy expert creating evaluation questions for a RAG system.
Given a chunk of text from a planning document, generate ONE factual question and its correct answer.

Rules:
- The question must be answerable solely from the provided chunk.
- The question must be specific (include policy codes, document names, or numbers where present).
- The answer must be concise (1-3 sentences), cite the key facts, AND include the exact document name and policy/paragraph reference (e.g. "NPPF paragraph 11", "Coventry Local Plan Policy DS1", "SPD Section 2.3").
- The ground_truth answer must end with a reference line in this format: [Source: <document name>, <policy/section identifier>]
- Do NOT ask vague questions like "What does this document say?"
- Do NOT ask questions whose answer is "See the document for details."
- If the chunk is too administrative (e.g. just an adoption statement or table of contents), reply with {"skip": true}.

Example of a good answer:
"Policy DS1 of the Coventry Local Plan requires all new residential developments over 10 units to achieve a minimum density of 40 dwellings per hectare in accessible locations. [Source: Coventry Local Plan, Policy DS1]"

Reply ONLY with valid JSON in one of these two formats:
{"question": "...", "ground_truth": "..."}
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


def sample_chunks(chunks: list[dict], n: int) -> list[dict]:
    """Sample diverse chunks: mix of collections, document classes, skip very short ones."""
    filtered = [
        c for c in chunks
        if len(c.get("text", "").split()) >= MIN_CHUNK_WORDS
        and c.get("document_class") not in {"Unknown"}
        and "adoption statement" not in c.get("text", "").lower()[:100]
    ]

    # Stratify by collection so we get both general + Coventry
    general = [c for c in filtered if c.get("collection") == "general"]
    coventry = [c for c in filtered if c.get("collection") == "Coventry"]

    general_n  = min(len(general),  n // 3)
    coventry_n = min(len(coventry), n - general_n)

    sampled = random.sample(general, general_n) + random.sample(coventry, coventry_n)
    random.shuffle(sampled)
    return sampled[:n]


def generate_pair(client: OpenAI, chunk: dict) -> dict | None:
    text = chunk["text"][:3000]  # truncate very long chunks
    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Document class: {chunk.get('document_class')}\nRegion: {chunk.get('region')}\n\nChunk:\n{text}"},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw = response.choices[0].message.content or ""
        # Strip markdown code fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        if parsed.get("skip"):
            return None
        return {
            "question":    parsed["question"],
            "ground_truth": parsed["ground_truth"],
            "source_chunk_id": chunk["chunk_id"],
            "document_class":  chunk.get("document_class"),
            "region":          chunk.get("region"),
            "source_path":     chunk.get("source_path"),
        }
    except Exception as e:
        console.print(f"[yellow]Skipped chunk {chunk.get('chunk_id')}: {e}[/yellow]")
        return None


def main() -> None:
    console.print(f"[bold blue]Golden Dataset Generator[/bold blue]")
    console.print(f"Model : {EVAL_MODEL}  |  Target: {TARGET_PAIRS} pairs")
    console.print(f"Output: {OUTPUT_PATH}\n")

    client = build_client()
    chunks = load_chunks()
    console.print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    sampled = sample_chunks(chunks, SAMPLE_PER_RUN)
    console.print(f"Sampled {len(sampled)} diverse chunks for generation\n")

    pairs: list[dict] = []
    for chunk in track(sampled, description="Generating Q&A pairs..."):
        if len(pairs) >= TARGET_PAIRS:
            break
        pair = generate_pair(client, chunk)
        if pair:
            pairs.append(pair)
        time.sleep(SLEEP_BETWEEN)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"\n[green]Done. Generated {len(pairs)} Q&A pairs → {OUTPUT_PATH}[/green]")
    console.print("[yellow]Review the file before running RAGAS — remove any low-quality pairs.[/yellow]")


if __name__ == "__main__":
    main()
