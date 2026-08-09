"""
Generate a genuinely multi-hop, cross-document evaluation set.

Why this exists:
  The original golden_dataset.json (60 pairs) was generated one chunk at a
  time — every ground truth is answerable from a SINGLE source chunk. That
  makes it structurally incapable of testing this project's central
  hypothesis: that Graph RAG (CROSS_DOC edges, LLM-typed triples) improves
  retrieval for questions that require combining information from BOTH the
  NPPF and the Coventry Local Plan. context_recall saturating at 1.0000 for
  well-tuned single-hop retrieval is a direct symptom of this gap.

How this set is built:
  1. Load the LLM-enriched policy graph (policy_graph_llm.json) and take its
     pre-computed CROSS_DOC edges — chunk-to-chunk links between an NPPF
     chunk and a Local Plan chunk that share a planning topic.
  2. For each candidate pair, prompt GPT-4.1 with BOTH chunks' text and ask
     for a question that can only be answered by combining information from
     both — reject (skip) any pair where the question turns out answerable
     from just one side.
  3. Save with a "source_chunk_ids" (plural) field, distinguishing this
     dataset's schema from the single-hop golden_dataset.json.

Leakage safety:
  Candidate chunks are excluded if they were used as a source chunk in
  EITHER the single-hop golden_dataset.json (60 pairs) OR the fine-tuning
  training set (finetune_train_pairs.json, 300 pairs) — the fine-tuned
  model saw those 300 chunks directly during training, so evaluating
  multi-hop questions built from them would leak exactly the same way the
  original bug did.

Output: data/golden_dataset_multihop.json
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
CHUNKS_PATH      = Path(os.getenv("CHUNKS_PATH", "outputs/chunks.jsonl"))
GRAPH_PATH       = Path("outputs") / "policy_graph_llm.json"
GOLDEN_PATH      = Path(os.getenv("DATASET_PATH", "data/dataset")).parent / "golden_dataset.json"
TRAIN_PATH       = Path(os.getenv("DATASET_PATH", "data/dataset")).parent / "finetune_train_pairs.json"
OUTPUT_PATH      = Path(os.getenv("DATASET_PATH", "data/dataset")).parent / "golden_dataset_multihop.json"

OPENAI_BASE_URL    = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "")
EVAL_MODEL         = os.getenv("EVAL_MODEL", "gpt-4.1")

TARGET_PAIRS   = 30    # multi-hop questions to generate
CANDIDATE_CAP  = 150   # how many CROSS_DOC edges to try before giving up
MIN_CHUNK_WORDS = 80
SLEEP_BETWEEN  = 1.0


SYSTEM_PROMPT = """You are a UK planning policy expert creating MULTI-HOP evaluation questions for a RAG system.

You will be given TWO chunks of text: one from the NPPF (national policy) and one from a Coventry Local Plan document. They were pre-identified as topically related.

Your task: write ONE question that can ONLY be answered correctly by combining information from BOTH chunks — not from either chunk alone. Good multi-hop questions typically ask the reader to compare, connect, or explain the relationship between the national policy and the local policy (e.g. "How does Coventry Local Plan Policy X implement or go beyond NPPF paragraph Y's requirement for Z?").

Rules:
- The question MUST require genuine synthesis of both chunks. If it can be fully answered using only one chunk, you must reply {"skip": true}.
- The answer must cite BOTH sources explicitly, e.g. ending with: [Sources: NPPF paragraph <N>; Coventry Local Plan, Policy <code>]
- The answer must be concise (2-4 sentences) and factually grounded only in the two chunks provided.
- Do NOT invent policy codes, paragraph numbers, or facts not present in the text.
- If the two chunks are not actually related enough to support a genuine multi-hop question, reply {"skip": true}.

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


def load_chunk_lookup(path: Path) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                lookup[chunk["chunk_id"]] = chunk
    return lookup


def load_excluded_chunk_ids() -> set[str]:
    """Chunks already used in single-hop eval (60) or fine-tuning training (300) — never reuse for multi-hop eval."""
    excluded: set[str] = set()
    if GOLDEN_PATH.exists():
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        excluded |= {item["source_chunk_id"] for item in golden if item.get("source_chunk_id")}
    if TRAIN_PATH.exists():
        train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
        excluded |= {item["source_chunk_id"] for item in train if item.get("source_chunk_id")}
    console.print(f"[bold]Excluding {len(excluded)} chunk IDs (single-hop golden set + fine-tuning training set)[/bold]")
    return excluded


def load_cross_doc_pairs(excluded_ids: set[str]) -> list[tuple[str, str]]:
    """CROSS_DOC edges from the LLM-enriched graph, filtered to exclude any leaked chunk on either side."""
    if not GRAPH_PATH.exists():
        console.print(f"[red]Graph not found at {GRAPH_PATH} — run scripts/build_graph_llm.py first[/red]")
        raise SystemExit(1)
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = graph["nodes"]
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset] = set()
    for edge in graph["edges"]:
        if edge["relation"] != "CROSS_DOC":
            continue
        src, dst = edge["src"], edge["dst"]
        if nodes.get(src, {}).get("type") != "chunk" or nodes.get(dst, {}).get("type") != "chunk":
            continue
        if src in excluded_ids or dst in excluded_ids:
            continue
        key = frozenset((src, dst))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((src, dst))
    console.print(f"Eligible CROSS_DOC pairs after exclusion: {len(pairs)}")
    random.shuffle(pairs)
    return pairs[:CANDIDATE_CAP]


def generate_multihop_pair(client: OpenAI, chunk_a: dict, chunk_b: dict) -> dict | None:
    text_a = chunk_a["text"][:2000]
    text_b = chunk_b["text"][:2000]
    user_content = (
        f"CHUNK A — {chunk_a.get('document_class')} ({chunk_a.get('region')}):\n{text_a}\n\n"
        f"CHUNK B — {chunk_b.get('document_class')} ({chunk_b.get('region')}):\n{text_b}"
    )
    try:
        response = client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=350,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        if parsed.get("skip"):
            return None
        return {
            "question":         parsed["question"],
            "ground_truth":     parsed["ground_truth"],
            "source_chunk_ids": [chunk_a["chunk_id"], chunk_b["chunk_id"]],
            "document_classes": [chunk_a.get("document_class"), chunk_b.get("document_class")],
            "regions":          [chunk_a.get("region"), chunk_b.get("region")],
            "source_paths":     [chunk_a.get("source_path"), chunk_b.get("source_path")],
        }
    except Exception as e:
        console.print(f"[yellow]Skipped pair ({chunk_a.get('chunk_id')}, {chunk_b.get('chunk_id')}): {e}[/yellow]")
        return None


def main() -> None:
    console.print("[bold blue]Multi-Hop Cross-Document Evaluation Set Generator[/bold blue]")
    console.print(f"Model : {EVAL_MODEL}  |  Target: {TARGET_PAIRS} multi-hop questions")
    console.print(f"Output: {OUTPUT_PATH}\n")

    excluded_ids = load_excluded_chunk_ids()
    chunk_lookup = load_chunk_lookup(CHUNKS_PATH)
    console.print(f"Loaded {len(chunk_lookup)} chunk texts")

    candidate_pairs = load_cross_doc_pairs(excluded_ids)
    if not candidate_pairs:
        console.print("[red]No eligible CROSS_DOC pairs found after exclusion. Aborting.[/red]")
        return

    client = build_client()

    results: list[dict] = []
    for src, dst in track(candidate_pairs, description="Generating multi-hop questions..."):
        if len(results) >= TARGET_PAIRS:
            break
        chunk_a = chunk_lookup.get(src)
        chunk_b = chunk_lookup.get(dst)
        if not chunk_a or not chunk_b:
            continue
        if len(chunk_a["text"].split()) < MIN_CHUNK_WORDS or len(chunk_b["text"].split()) < MIN_CHUNK_WORDS:
            continue
        pair = generate_multihop_pair(client, chunk_a, chunk_b)
        if pair:
            results.append(pair)
        time.sleep(SLEEP_BETWEEN)

    # ── Safety check: no chunk in this set overlaps the excluded sets ────────
    used_ids = {cid for r in results for cid in r["source_chunk_ids"]}
    overlap = used_ids & excluded_ids
    if overlap:
        console.print(f"[red]FATAL: {len(overlap)} multi-hop chunk IDs overlap with excluded sets. Aborting write.[/red]")
        raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"\n[green]Done. Generated {len(results)} multi-hop questions → {OUTPUT_PATH}[/green]")
    console.print(f"[green]Verified zero overlap with {len(excluded_ids)} single-hop/training chunk IDs.[/green]")
    console.print("[dim]Next: python -m experiments.eval_multihop[/dim]")


if __name__ == "__main__":
    main()
