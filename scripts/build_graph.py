"""
Build a property graph over all planning policy chunks.

Nodes:
  - chunk      : each text chunk (chunk_id)
  - policy     : policy codes like "DS1", "H4", "GB1"
  - topic      : planning topics like "Affordable Housing", "Green Belt"
  - nppf_para  : NPPF paragraph references like "paragraph 11"

Edges:
  - chunk  -[MENTIONS]->     policy / topic / nppf_para
  - policy -[CO_OCCURS]->    policy  (two policies co-occur in the same chunk)
  - chunk  -[CROSS_DOC]->    chunk   (Local Plan chunk explicitly references NPPF paragraph
                                       → linked to the NPPF chunk that IS that paragraph)

Cross-document linking (v2):
  Problem: the original graph only fired NPPF_RE on text like "NPPF paragraph 11"
  in Local Plan chunks. NPPF chunks that ARE paragraph 11 never tagged themselves
  with nppf:NPPF_para_11, so the bridge was one-sided.

  Fix 1 — NPPF self-tagging:
    Detect paragraph numbers within NPPF-sourced chunks using the paragraph
    numbering style ("11.  The planning system..."). Tag those chunks with
    nppf:NPPF_para_N so the same node is reachable from both sides.

  Fix 2 — CROSS_DOC edges:
    For every topic shared between ≥1 NPPF chunk and ≥1 Local Plan chunk, add
    direct CROSS_DOC edges (Local Plan → NPPF) so the graph traversal can hop
    across documents in a single step without needing topic nodes as an
    intermediate.

Output: outputs/policy_graph.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.progress import track

console = Console()

CHUNKS_PATH = PROJECT_ROOT / "outputs" / "chunks.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "policy_graph.json"

# ── Entity extraction patterns ────────────────────────────────────────────────

# Local Plan policy codes: "Policy DS1", "DS1 of the Local Plan", etc.
POLICY_RE = re.compile(
    r'\bPolicy\s+([A-Z]{1,3}\d+[a-z]?)\b'
    r'|\b([A-Z]{1,3}\d+[a-z]?)\s+(?:of\s+the\s+)?(?:Local\s+Plan|Coventry)',
    re.IGNORECASE,
)

# References to NPPF paragraph numbers FROM any chunk
# e.g. "NPPF paragraph 11", "paragraph 11 of the NPPF"
NPPF_REF_RE = re.compile(
    r'\bNPPF\s+[Pp]aragraph[s]?\s+(\d+(?:[-–]\d+)?)\b'
    r'|\b[Pp]aragraph[s]?\s+(\d+(?:[-–]\d+)?)\s+of\s+the\s+(?:NPPF|National\s+Planning\s+Policy\s+Framework)\b',
)

# Paragraph numbers WITHIN NPPF source chunks — detects "11.  The..." or "11  The..."
# at the start of a line, followed by an uppercase letter.
# Kept strict (1–3 digit, ≤ 250) to avoid false positives from years / section refs.
NPPF_SELF_PARA_RE = re.compile(r'(?:^|\n)\s*(\d{1,3})[.\s]\s+[A-Z]')

# National-policy reference phrases in Local Plan chunks
NATIONAL_POLICY_RE = re.compile(
    r'\b(?:national\s+(?:planning\s+)?policy|NPPF|National\s+Planning\s+Policy\s+Framework)\b',
    re.IGNORECASE,
)

PLANNING_TOPICS = [
    "Affordable Housing", "Green Belt", "Housing Density", "Sustainable Development",
    "Heritage", "Flood Risk", "Transport", "Open Space", "Biodiversity",
    "Climate Change", "Design Quality", "Employment Land", "Retail",
    "Traveller Sites", "Health", "Education", "Renewable Energy",
    "Infrastructure", "Community Facilities", "Town Centre",
]


# ── Document source classification ───────────────────────────────────────────

def is_nppf_chunk(chunk: dict) -> bool:
    return (
        chunk.get("collection") == "general"
        or "National Planning Policy" in (chunk.get("document_class") or "")
        or "nppf" in (chunk.get("source_path") or "").lower()
    )


def is_local_plan_chunk(chunk: dict) -> bool:
    return chunk.get("collection") == "Coventry"


# ── Entity extraction ─────────────────────────────────────────────────────────

def extract_entities(text: str, chunk: dict) -> dict[str, list[str]]:
    policies: set[str] = set()
    for m in POLICY_RE.finditer(text):
        code = (m.group(1) or m.group(2) or "").strip().upper()
        if code and len(code) >= 2:
            policies.add(code)

    nppf_paras: set[str] = set()

    # References to NPPF paragraphs from any document
    for m in NPPF_REF_RE.finditer(text):
        para = (m.group(1) or m.group(2) or "").strip()
        if para:
            # Handle ranges like "11-14" → tag each individually
            if "–" in para or "-" in para:
                sep = "–" if "–" in para else "-"
                parts = para.split(sep)
                try:
                    start, end = int(parts[0]), int(parts[1])
                    for n in range(start, min(end + 1, start + 10)):  # cap range at 10
                        nppf_paras.add(f"NPPF_para_{n}")
                except ValueError:
                    nppf_paras.add(f"NPPF_para_{para}")
            else:
                nppf_paras.add(f"NPPF_para_{para}")

    # Self-tagging: NPPF chunks tag themselves with their own paragraph numbers
    if is_nppf_chunk(chunk):
        for m in NPPF_SELF_PARA_RE.finditer(text):
            num = int(m.group(1))
            if 1 <= num <= 250:
                nppf_paras.add(f"NPPF_para_{num}")

    topics: set[str] = set()
    text_lower = text.lower()
    for topic in PLANNING_TOPICS:
        if topic.lower() in text_lower:
            topics.add(topic)

    # Tag Local Plan chunks that reference national policy generically
    references_national = bool(NATIONAL_POLICY_RE.search(text)) and is_local_plan_chunk(chunk)

    return {
        "policies":          sorted(policies),
        "nppf_paras":        sorted(nppf_paras),
        "topics":            sorted(topics),
        "references_national": references_national,
    }


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(chunks: list[dict]) -> dict:
    nodes: dict[str, dict]  = {}
    edges: list[dict]       = []
    chunk_entities: dict[str, dict] = {}

    # ── Pass 1: nodes + MENTIONS edges ────────────────────────────────────────
    for chunk in track(chunks, description="Extracting entities"):
        chunk_id = chunk["chunk_id"]
        nodes[chunk_id] = {
            "type":           "chunk",
            "source_path":    chunk.get("source_path", ""),
            "document_class": chunk.get("document_class", ""),
            "collection":     chunk.get("collection", ""),
            "region":         chunk.get("region", ""),
            "is_nppf":        is_nppf_chunk(chunk),
            "is_local_plan":  is_local_plan_chunk(chunk),
            "text_preview":   chunk["text"][:120],
        }
        entities = extract_entities(chunk["text"], chunk)
        chunk_entities[chunk_id] = entities

        for policy in entities["policies"]:
            node_id = f"policy:{policy}"
            if node_id not in nodes:
                nodes[node_id] = {"type": "policy", "label": policy}
            edges.append({"src": chunk_id, "dst": node_id, "relation": "MENTIONS"})

        for para in entities["nppf_paras"]:
            node_id = f"nppf:{para}"
            if node_id not in nodes:
                nodes[node_id] = {"type": "nppf_para", "label": para}
            edges.append({"src": chunk_id, "dst": node_id, "relation": "MENTIONS"})

        for topic in entities["topics"]:
            node_id = f"topic:{topic.replace(' ', '_')}"
            if node_id not in nodes:
                nodes[node_id] = {"type": "topic", "label": topic}
            edges.append({"src": chunk_id, "dst": node_id, "relation": "MENTIONS"})

    # ── Pass 2: CO_OCCURS edges (policy–policy within same chunk) ─────────────
    for chunk_id, entities in chunk_entities.items():
        policies = entities["policies"]
        for i, p1 in enumerate(policies):
            for p2 in policies[i + 1:]:
                edges.append({
                    "src": f"policy:{p1}",
                    "dst": f"policy:{p2}",
                    "relation": "CO_OCCURS",
                })

    # ── Pass 3: CROSS_DOC edges (Local Plan ↔ NPPF via shared topic) ──────────
    # For each planning topic, collect NPPF chunk ids and Local Plan chunk ids.
    # Add a direct edge from each Local Plan chunk to each NPPF chunk sharing
    # the topic so graph expansion can hop cross-document in a single step.
    topic_nppf:  dict[str, list[str]] = defaultdict(list)
    topic_local: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        if edge["relation"] != "MENTIONS":
            continue
        cid      = edge["src"]
        entity   = edge["dst"]
        if not entity.startswith("topic:"):
            continue
        n = nodes.get(cid, {})
        if n.get("is_nppf"):
            topic_nppf[entity].append(cid)
        elif n.get("is_local_plan"):
            topic_local[entity].append(cid)

    cross_doc_count = 0
    for topic_id, lp_chunks in topic_local.items():
        nppf_chunks = topic_nppf.get(topic_id, [])
        if not nppf_chunks:
            continue
        # Cap at 10×10 per topic to prevent edge explosion on broad topics
        for lp_cid in lp_chunks[:10]:
            for nppf_cid in nppf_chunks[:10]:
                edges.append({
                    "src":    lp_cid,
                    "dst":    nppf_cid,
                    "relation": "CROSS_DOC",
                })
                cross_doc_count += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "cross_doc_edges": cross_doc_count,
        },
    }


def main() -> None:
    console.print("[bold blue]Building Policy Knowledge Graph (v2 — cross-document linking)[/bold blue]")

    if not CHUNKS_PATH.exists():
        console.print(f"[red]Chunks not found at {CHUNKS_PATH}[/red]")
        return

    chunks: list[dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    console.print(f"Loaded {len(chunks)} chunks")

    n_nppf  = sum(1 for c in chunks if is_nppf_chunk(c))
    n_local = sum(1 for c in chunks if is_local_plan_chunk(c))
    console.print(f"  NPPF chunks      : {n_nppf}")
    console.print(f"  Local Plan chunks: {n_local}")

    graph = build_graph(chunks)

    n_chunk_nodes  = sum(1 for n in graph["nodes"].values() if n["type"] == "chunk")
    n_policy_nodes = sum(1 for n in graph["nodes"].values() if n["type"] == "policy")
    n_topic_nodes  = sum(1 for n in graph["nodes"].values() if n["type"] == "topic")
    n_nppf_nodes   = sum(1 for n in graph["nodes"].values() if n["type"] == "nppf_para")

    by_rel: dict[str, int] = defaultdict(int)
    for e in graph["edges"]:
        by_rel[e["relation"]] += 1

    console.print(f"\n[bold]Graph summary:[/bold]")
    console.print(f"  Chunk nodes        : {n_chunk_nodes}")
    console.print(f"  Policy nodes       : {n_policy_nodes}")
    console.print(f"  Topic nodes        : {n_topic_nodes}")
    console.print(f"  NPPF para nodes    : {n_nppf_nodes}")
    console.print(f"  Total edges        : {len(graph['edges'])}")
    for rel, count in sorted(by_rel.items()):
        console.print(f"    {rel:<20} : {count}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Graph saved to {OUTPUT_PATH}[/green]")


if __name__ == "__main__":
    main()
