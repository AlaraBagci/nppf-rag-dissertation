"""
Build an LLM-enriched policy graph using GPT-4.1 to extract typed triples.

This script extends the v2 regex graph (outputs/policy_graph.json) with
semantically-typed edges extracted by GPT-4.1 reading each chunk.

Why this matters:
  Regex can find "Policy H4" and "NPPF paragraph 63" in the same chunk but
  cannot tell you *how* they are related. GPT-4.1 reads the text and extracts:
    (Policy H4, IMPLEMENTS, NPPF paragraph 63)
    (Policy H4, REQUIRES, 25% affordable housing provision)
  These typed, directional relationships allow the graph expansion to rank
  cross-document connections by their semantic strength, not just co-occurrence.

Triple extraction:
  - GPT-4.1 is called once per chunk
  - Results are cached to outputs/llm_triples_cache.json (resume-safe)
  - Extracted subjects/objects are resolved to existing graph nodes where possible
  - New concept nodes are created for everything else

Output: outputs/policy_graph_llm.json

Run time: ~15–25 min for ~800 chunks (rate-limited to 1 call/sec)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AzureOpenAI
from rich.console import Console
from rich.progress import track

console = Console()

BASE_GRAPH_PATH  = PROJECT_ROOT / "outputs" / "policy_graph.json"
CACHE_PATH       = PROJECT_ROOT / "outputs" / "llm_triples_cache.json"
OUTPUT_PATH      = PROJECT_ROOT / "outputs" / "policy_graph_llm.json"
CHUNKS_PATH      = PROJECT_ROOT / "outputs" / "chunks.jsonl"

# Relation weights used later in graph expansion
RELATION_WEIGHTS = {
    "IMPLEMENTS":  3,   # local policy derives authority from national policy
    "REQUIRES":    2,   # policy places obligation on an applicant
    "RESTRICTS":   2,   # policy restricts a type of development
    "SUPERSEDES":  2,   # policy replaces or overrides another
    "APPLIES_TO":  2,   # policy applies to a specific area / development type
    "REFERENCES":  1,   # text cites another document / policy without implementing it
}

EXTRACTION_PROMPT = """You are a planning policy analyst. Read the text below and extract up to 5 relationships between planning policy entities.

Return a JSON array only — no explanation, no markdown, no extra text.

Each item: {{"subject": "...", "relation": "...", "object": "..."}}

Allowed relations:
- IMPLEMENTS  : this local policy implements or derives authority from a national policy / NPPF paragraph
- REQUIRES    : this policy requires something specific from a developer or applicant
- RESTRICTS   : this policy restricts a type of development or use
- APPLIES_TO  : this policy applies to a specific development type, site, or geographic area
- REFERENCES  : this text cites another specific policy code or NPPF paragraph (without implementing it)
- SUPERSEDES  : this policy replaces or overrides another policy

Rules:
- Subject and object must be SPECIFIC: exact policy codes (DS1, H4), paragraph numbers (NPPF paragraph 63), or short noun phrases (3 words max)
- Do NOT use vague terms like "it", "this policy", "the plan"
- If no clear relationships exist, return []

Text:
{text}"""


# ── Entity resolution ─────────────────────────────────────────────────────────

POLICY_NORM_RE  = re.compile(r'\b(?:Policy\s+)?([A-Z]{1,3}\d+[a-z]?)\b')
NPPF_PARA_RE    = re.compile(r'\b(?:NPPF\s+)?[Pp]aragraph[s]?\s+(\d+(?:[-–]\d+)?)\b')


def resolve_entity(text: str, existing_nodes: set[str]) -> str:
    """Map a raw extracted entity string to an existing graph node id or new concept id."""
    text = text.strip()

    # Try policy code
    m = POLICY_NORM_RE.search(text)
    if m:
        candidate = f"policy:{m.group(1).upper()}"
        if candidate in existing_nodes:
            return candidate

    # Try NPPF paragraph
    m = NPPF_PARA_RE.search(text)
    if m:
        para = m.group(1)
        if "–" in para or "-" in para:
            sep = "–" if "–" in para else "-"
            start = para.split(sep)[0]
            return f"nppf:NPPF_para_{start}"
        return f"nppf:NPPF_para_{para}"

    # Generic concept node
    slug = re.sub(r'[^a-z0-9_]', '_', text.lower())[:60].strip('_')
    return f"concept:{slug}"


# ── GPT-4.1 triple extraction ─────────────────────────────────────────────────

def extract_triples(
    chunk_id: str,
    text: str,
    client: AzureOpenAI,
    eval_model: str,
) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(text=text[:1500])  # cap at 1500 chars
    try:
        response = client.chat.completions.create(
            model=eval_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        triples = json.loads(raw)
        if not isinstance(triples, list):
            return []
        valid = []
        for t in triples:
            if (
                isinstance(t, dict)
                and "subject" in t and "relation" in t and "object" in t
                and t["relation"].upper() in RELATION_WEIGHTS
                and len(t["subject"]) >= 2
                and len(t["object"]) >= 2
            ):
                valid.append({
                    "subject":  t["subject"].strip(),
                    "relation": t["relation"].upper().strip(),
                    "object":   t["object"].strip(),
                })
        return valid[:5]
    except Exception as exc:
        console.print(f"[yellow]  Triple extraction failed for {chunk_id}: {exc}[/yellow]")
        return []


# ── Graph enrichment ──────────────────────────────────────────────────────────

def enrich_graph(
    base_graph: dict,
    all_triples: dict[str, list[dict]],   # chunk_id → raw triples from GPT-4.1
) -> dict:
    nodes = dict(base_graph["nodes"])
    edges = list(base_graph["edges"])

    existing_node_ids = set(nodes.keys())
    llm_edge_count = 0

    for chunk_id, triples in all_triples.items():
        if chunk_id not in nodes:
            continue
        for triple in triples:
            relation = triple["relation"]
            weight   = RELATION_WEIGHTS.get(relation, 1)

            # Resolve subject — typically the chunk itself or a policy code in it
            subj_raw = triple["subject"]
            subj_id  = resolve_entity(subj_raw, existing_node_ids)

            # Resolve object
            obj_raw = triple["object"]
            obj_id  = resolve_entity(obj_raw, existing_node_ids)

            # Create missing nodes
            for nid, label in [(subj_id, subj_raw), (obj_id, obj_raw)]:
                if nid not in nodes and not nid.startswith("chunk"):
                    ntype = "concept"
                    if nid.startswith("policy:"):
                        ntype = "policy"
                    elif nid.startswith("nppf:"):
                        ntype = "nppf_para"
                    nodes[nid] = {"type": ntype, "label": label, "llm_created": True}
                    existing_node_ids.add(nid)

            # Edge: chunk → object (the chunk is the source of the relationship)
            edges.append({
                "src":      chunk_id,
                "dst":      obj_id,
                "relation": f"LLM_{relation}",
                "weight":   weight,
                "subject":  subj_raw,
            })

            # If subject resolves to an existing policy node (not the chunk itself),
            # also add a typed entity→entity edge for policy-level graph traversal
            if subj_id != chunk_id and subj_id in existing_node_ids:
                edges.append({
                    "src":      subj_id,
                    "dst":      obj_id,
                    "relation": f"LLM_{relation}",
                    "weight":   weight,
                })

            llm_edge_count += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "meta":  {
            **base_graph.get("meta", {}),
            "llm_triple_edges": llm_edge_count,
            "llm_model": "gpt-4.1",
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("[bold blue]Building LLM-enriched Policy Graph (exp07)[/bold blue]")

    # ── Load base (v2 regex) graph ────────────────────────────────────────────
    if not BASE_GRAPH_PATH.exists():
        console.print(f"[red]Base graph not found. Run scripts/build_graph.py first.[/red]")
        return
    base_graph = json.loads(BASE_GRAPH_PATH.read_text(encoding="utf-8"))
    console.print(f"Base graph: {len(base_graph['nodes'])} nodes, {len(base_graph['edges'])} edges")

    # ── Load chunks ───────────────────────────────────────────────────────────
    chunks: list[dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    console.print(f"Loaded {len(chunks)} chunks to process")

    # ── Load triple cache (resume support) ────────────────────────────────────
    cache: dict[str, list[dict]] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        console.print(f"[green]Cache loaded: {len(cache)} chunks already processed[/green]")

    # ── Set up Azure OpenAI client ────────────────────────────────────────────
    from src.config import get_settings, resolve_endpoint_config
    settings = get_settings()
    openai_base_url, openai_url_version = resolve_endpoint_config(
        os.getenv("OPENAI_BASE_URL", "")
    )
    openai_api_key     = os.getenv("OPENAI_API_KEY", "")
    openai_api_version = os.getenv("OPENAI_API_VERSION", "") or openai_url_version

    if not openai_base_url or not openai_api_key:
        console.print("[red]OPENAI_BASE_URL / OPENAI_API_KEY not set — cannot extract triples.[/red]")
        return

    client = AzureOpenAI(
        azure_endpoint=openai_base_url,
        api_key=openai_api_key,
        api_version=openai_api_version,
    )
    eval_model = settings.eval_model
    console.print(f"Using model: {eval_model}")

    # ── Process chunks ────────────────────────────────────────────────────────
    to_process = [c for c in chunks if c["chunk_id"] not in cache]
    console.print(f"Chunks to process: {len(to_process)} (skipping {len(cache)} cached)")

    new_count = 0
    total_triples = 0

    for chunk in track(to_process, description="Extracting triples"):
        chunk_id = chunk["chunk_id"]
        triples = extract_triples(
            chunk_id=chunk_id,
            text=chunk["text"],
            client=client,
            eval_model=eval_model,
        )
        cache[chunk_id] = triples
        new_count += 1
        total_triples += len(triples)

        # Save cache every 50 chunks (resume safety)
        if new_count % 50 == 0:
            CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            console.print(f"  [dim]Cache saved ({new_count} new, {total_triples} triples so far)[/dim]")

        time.sleep(0.5)   # ~2 req/sec to avoid 429s

    # Final cache save
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # Summarise triple quality
    all_triples_flat = [t for triples in cache.values() for t in triples]
    relation_counts: dict[str, int] = defaultdict(int)
    for t in all_triples_flat:
        relation_counts[t["relation"]] += 1

    console.print(f"\n[bold]Triple extraction summary:[/bold]")
    console.print(f"  Chunks processed : {len(cache)}")
    console.print(f"  Total triples    : {len(all_triples_flat)}")
    for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
        console.print(f"    {rel:<15} : {count}")

    # ── Build enriched graph ──────────────────────────────────────────────────
    console.print("\n[bold]Enriching graph with LLM triples...[/bold]")
    enriched = enrich_graph(base_graph, cache)

    by_rel: dict[str, int] = defaultdict(int)
    for e in enriched["edges"]:
        by_rel[e["relation"]] += 1

    console.print(f"\n[bold]Enriched graph summary:[/bold]")
    console.print(f"  Total nodes : {len(enriched['nodes'])}")
    console.print(f"  Total edges : {len(enriched['edges'])}")
    for rel, count in sorted(by_rel.items()):
        console.print(f"    {rel:<25} : {count}")

    OUTPUT_PATH.write_text(json.dumps(enriched, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]LLM-enriched graph saved to {OUTPUT_PATH}[/green]")


if __name__ == "__main__":
    main()
