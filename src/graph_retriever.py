"""
Graph-enhanced retrieval using the policy knowledge graph.

Two graph variants are supported:
  PolicyGraph      — loads the v2 regex graph (policy_graph.json)
  LLMPolicyGraph   — loads the LLM-enriched graph (policy_graph_llm.json),
                     which adds GPT-4.1-extracted typed triples on top of v2.

Multi-hop expansion logic:
  1. Standard hybrid search returns seed chunks (same as exp02/exp04)
  2. For each seed chunk, look up entities it mentions (policy codes, topics, NPPF paras)
  3. Hop 1: collect all chunks sharing any of those entities
  4. Hop 2a: follow CO_OCCURS edges (policy → co-occurring policy → chunks)
  5. Hop 2b: follow CROSS_DOC edges (Local Plan ↔ NPPF direct bridge)
  6. Hop 2c [LLM graph only]: follow LLM_* triple edges — weighted by relation type:
       IMPLEMENTS / REQUIRES / RESTRICTS / SUPERSEDES → weight 3
       APPLIES_TO                                     → weight 2
       REFERENCES                                     → weight 1
     These give semantically-typed cross-document connections ranked higher
     than keyword co-occurrence, answering the examiner question of why
     not use LLM-extracted triples.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class PolicyGraph:
    """In-memory property graph loaded from build_graph.py output JSON."""

    def __init__(self, graph_path: str | Path) -> None:
        data = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        self.nodes: dict[str, dict] = data["nodes"]
        self.edges: list[dict]      = data["edges"]

        # Build adjacency: entity_node_id → set of chunk_ids that MENTION it
        self._entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        # Build adjacency: chunk_id → set of entity_node_ids it mentions
        self._chunk_to_entities: dict[str, set[str]] = defaultdict(set)
        # CO_OCCURS: policy_node_id → set of co-occurring policy_node_ids
        self._co_occurs: dict[str, set[str]] = defaultdict(set)
        # CROSS_DOC: chunk_id → set of cross-document chunk_ids (Local Plan ↔ NPPF)
        self._cross_doc: dict[str, set[str]] = defaultdict(set)
        # LLM triples: chunk_id → {target_id → weight}  (exp07 only, empty for exp05)
        self._llm_targets: dict[str, dict[str, int]] = defaultdict(dict)

        LLM_WEIGHTS = {
            "LLM_IMPLEMENTS": 3, "LLM_REQUIRES": 3, "LLM_RESTRICTS": 3,
            "LLM_SUPERSEDES": 3, "LLM_APPLIES_TO": 2, "LLM_REFERENCES": 1,
        }

        for edge in self.edges:
            src, dst, rel = edge["src"], edge["dst"], edge["relation"]
            if rel == "MENTIONS":
                self._entity_to_chunks[dst].add(src)
                self._chunk_to_entities[src].add(dst)
            elif rel == "CO_OCCURS":
                self._co_occurs[src].add(dst)
                self._co_occurs[dst].add(src)
            elif rel == "CROSS_DOC":
                self._cross_doc[src].add(dst)
                self._cross_doc[dst].add(src)
            elif rel in LLM_WEIGHTS:
                weight = edge.get("weight", LLM_WEIGHTS[rel])
                # chunk → entity/concept (for expanding from a seed chunk)
                if src in self.nodes and self.nodes[src].get("type") == "chunk":
                    self._llm_targets[src][dst] = max(
                        self._llm_targets[src].get(dst, 0), weight
                    )
                    # Also register entity→chunk for reverse lookup
                    self._entity_to_chunks[dst].add(src)
                # entity → entity (e.g. policy:H4 IMPLEMENTS nppf:NPPF_para_63)
                if src in self.nodes and self.nodes[src].get("type") != "chunk":
                    self._llm_targets[src][dst] = max(
                        self._llm_targets[src].get(dst, 0), weight
                    )

    def entities_for_chunk(self, chunk_id: str) -> set[str]:
        return self._chunk_to_entities.get(chunk_id, set())

    def chunks_for_entity(self, entity_id: str) -> set[str]:
        return self._entity_to_chunks.get(entity_id, set())

    def co_occurring_policies(self, policy_node_id: str) -> set[str]:
        return self._co_occurs.get(policy_node_id, set())

    def cross_doc_chunks(self, chunk_id: str) -> set[str]:
        return self._cross_doc.get(chunk_id, set())

    def llm_targets(self, node_id: str) -> dict[str, int]:
        """Return {target_node_id: weight} for all LLM triple edges from this node."""
        return self._llm_targets.get(node_id, {})


def graph_expand(
    seed_chunk_ids: list[str],
    graph: PolicyGraph,
    chunk_lookup: dict[str, dict],
    hops: int = 2,
    max_expanded: int = 40,
) -> list[dict]:
    """
    Expand seed chunks via the knowledge graph.

    Hop 1: find all chunks that share a policy/topic entity with any seed chunk
    Hop 2: follow CO_OCCURS edges — find policies co-occurring with seed policies,
            then retrieve their chunks

    Returns expanded chunks (excluding seeds, preserving order by entity overlap score).
    """
    seed_set = set(seed_chunk_ids)

    # ── Collect entities from seeds ───────────────────────────────────────────
    seed_entities: set[str] = set()
    for cid in seed_chunk_ids:
        seed_entities.update(graph.entities_for_chunk(cid))

    # ── Hop 1: chunks sharing entities with seeds ─────────────────────────────
    hop1_candidates: dict[str, int] = defaultdict(int)  # chunk_id → overlap count
    for entity in seed_entities:
        for cid in graph.chunks_for_entity(entity):
            if cid not in seed_set:
                hop1_candidates[cid] += 1

    expanded_ids: set[str] = set(hop1_candidates.keys())

    # ── Hop 2a: CO_OCCURS expansion (policy → co-occurring policy → chunks) ──
    if hops >= 2:
        policy_entities = {e for e in seed_entities if e.startswith("policy:")}
        hop2_entities: set[str] = set()
        for policy_e in policy_entities:
            for co_policy in graph.co_occurring_policies(policy_e):
                if co_policy not in seed_entities:
                    hop2_entities.add(co_policy)

        for entity in hop2_entities:
            for cid in graph.chunks_for_entity(entity):
                if cid not in seed_set and cid not in expanded_ids:
                    hop1_candidates[cid] += 0  # lower priority than hop-1 matches
                    expanded_ids.add(cid)

    # ── Hop 2b: CROSS_DOC expansion (Local Plan chunk → NPPF chunk, and vice versa)
    if hops >= 2:
        for cid in list(seed_chunk_ids):
            for cross_cid in graph.cross_doc_chunks(cid):
                if cross_cid not in seed_set and cross_cid not in expanded_ids:
                    hop1_candidates[cross_cid] += 1
                    expanded_ids.add(cross_cid)

    # ── Hop 2c: LLM triple expansion (weighted by relation type) ─────────────
    # Only active when graph was built with build_graph_llm.py.
    # For each seed chunk, follow LLM_* edges to target entities, then find
    # all chunks connected to those entities. Weight is relation-specific:
    #   IMPLEMENTS/REQUIRES/RESTRICTS → 3  (strong semantic dependency)
    #   APPLIES_TO                    → 2
    #   REFERENCES                    → 1
    if hops >= 2:
        for cid in list(seed_chunk_ids):
            # Direct LLM triples from this chunk
            for target_id, weight in graph.llm_targets(cid).items():
                # target may be an entity node or another chunk
                if target_id in graph.nodes and graph.nodes[target_id].get("type") == "chunk":
                    if target_id not in seed_set:
                        hop1_candidates[target_id] += weight
                        expanded_ids.add(target_id)
                else:
                    # target is an entity node — find chunks connected to it
                    for linked_cid in graph.chunks_for_entity(target_id):
                        if linked_cid not in seed_set:
                            hop1_candidates[linked_cid] += weight
                            expanded_ids.add(linked_cid)

        # Also follow entity→entity LLM triples from seed entities
        # e.g. policy:H4 -[LLM_IMPLEMENTS]-> nppf:NPPF_para_63
        for entity in seed_entities:
            for target_id, weight in graph.llm_targets(entity).items():
                for linked_cid in graph.chunks_for_entity(target_id):
                    if linked_cid not in seed_set:
                        hop1_candidates[linked_cid] += weight
                        expanded_ids.add(linked_cid)

    # ── Sort by overlap count (descending) and cap ────────────────────────────
    ranked = sorted(expanded_ids, key=lambda cid: hop1_candidates.get(cid, 0), reverse=True)
    ranked = ranked[:max_expanded]

    # ── Retrieve chunk texts from lookup ──────────────────────────────────────
    results: list[dict] = []
    for cid in ranked:
        chunk = chunk_lookup.get(cid)
        if chunk is None:
            continue
        results.append({
            "chunk_id":   cid,
            "text":       chunk.get("text", ""),
            "metadata":   {
                **{k: chunk.get(k) for k in [
                    "document_id", "source_path", "collection", "region",
                    "top_folder", "sub_topic", "document_class",
                ]},
                **chunk.get("metadata", {}),
            },
            "distance":   0.5,  # neutral score — will be reranked by cross-encoder
            "graph_expanded": True,
            "entity_overlap": hop1_candidates.get(cid, 0),
        })
    return results


def rerank_with_graph_reservation(
    query: str,
    merged_candidates: list[dict],
    graph_expanded_ids: set[str],
    reranker_model: str,
    top_k: int,
) -> list[dict]:
    """
    Reranking with one slot reserved for graph-recovered complementary evidence.

    Rationale: pointwise cross-encoder rerankers (e.g. bge-reranker-v2-m3) score
    each candidate independently against the query — they have no mechanism to
    reward a candidate for being useful only *in combination with* another
    candidate (Cao et al., 2007, "Learning to Rank: From Pairwise Approach to
    Listwise Approach"; Liu, 2009). For multi-hop questions requiring synthesis
    of two complementary chunks, a graph-recovered chunk may legitimately score
    lower in isolation than a chunk that is individually relevant but
    insufficient alone — causing the reranker to systematically discard exactly
    the evidence multi-hop graph expansion was built to recover.

    Fix: score the full merged pool exactly as before (the reranker's judgment
    is not overridden). Take the top (top_k - 1) by score. For the final slot,
    reserve it for the highest-scoring candidate that (a) came from graph
    expansion and (b) is not already selected — a coverage-aware selection
    step conceptually related to Maximal Marginal Relevance (Carbonell &
    Goldstein, 1998) and to evidence-preservation strategies in multi-hop RAG
    (e.g. IRCoT, Trivedi et al. 2023). If no graph-only candidate remains
    outside the top (top_k - 1), behaviour is identical to plain reranking.
    """
    from src.retriever import rerank

    # Score the entire pool (not just top_k) so we can inspect candidates
    # beyond the naive cutoff when reserving the final slot.
    fully_ranked = rerank(query, merged_candidates, model_name=reranker_model, top_k=len(merged_candidates))

    primary = fully_ranked[: top_k - 1]
    primary_ids = {c["chunk_id"] for c in primary}

    reserved = None
    for candidate in fully_ranked[top_k - 1 :]:
        if candidate["chunk_id"] in graph_expanded_ids and candidate["chunk_id"] not in primary_ids:
            reserved = candidate
            break

    if reserved is not None:
        return primary + [reserved]

    # No eligible graph-only candidate beyond the cutoff — fall back to plain top_k.
    return fully_ranked[:top_k]


def hybrid_graph_query(
    query: str,
    bm25_index,
    embed_model,
    chroma_path: str | Path,
    collection_name: str,
    graph: PolicyGraph,
    chunk_lookup: dict[str, dict],
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    top_k: int = 5,
    candidate_pool: int = 20,
    graph_hops: int = 2,
    graph_max_expanded: int = 40,
    reserve_graph_slot: bool = False,
) -> list[dict]:
    """
    Graph-RAG retrieval pipeline:
      1. BM25 sparse retrieval (seed)
      2. Dense vector retrieval (seed)
      3. RRF fusion of both seed lists
      4. Graph expansion via entity co-occurrence + CO_OCCURS (multi-hop)
      5. Merge seed + expanded, cross-encoder rerank → top_k

    reserve_graph_slot: if True, use rerank_with_graph_reservation() for step 5
    instead of plain reranking — reserves the final context slot for the
    highest-scoring graph-expanded candidate not otherwise selected. Defaults
    to False so existing experiments (Exp05/Exp07/Exp08) are unaffected unless
    explicitly opted in.
    """
    import chromadb
    from src.retriever import reciprocal_rank_fusion, rerank

    # ── 1. BM25 seed retrieval ────────────────────────────────────────────────
    bm25_results = bm25_index.retrieve(query, top_k=candidate_pool)

    # ── 2. Dense seed retrieval ───────────────────────────────────────────────
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
            "text":     raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
            "distance": raw["distances"][0][i],
        }
        for i in range(len(raw["ids"][0]))
    ]

    # ── 3. RRF fusion of seed results ─────────────────────────────────────────
    seed_results = reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_n=candidate_pool)

    # ── 4. Graph expansion ────────────────────────────────────────────────────
    seed_ids = [r["chunk_id"] for r in seed_results]
    expanded = graph_expand(
        seed_chunk_ids=seed_ids,
        graph=graph,
        chunk_lookup=chunk_lookup,
        hops=graph_hops,
        max_expanded=graph_max_expanded,
    )

    # ── 5. Merge seed + expanded, rerank ─────────────────────────────────────
    seen: set[str] = set(seed_ids)
    merged = list(seed_results)
    expanded_ids: set[str] = set()
    for chunk in expanded:
        if chunk["chunk_id"] not in seen:
            merged.append(chunk)
            seen.add(chunk["chunk_id"])
            expanded_ids.add(chunk["chunk_id"])

    if reserve_graph_slot:
        return rerank_with_graph_reservation(
            query=query, merged_candidates=merged, graph_expanded_ids=expanded_ids,
            reranker_model=reranker_model, top_k=top_k,
        )

    reranked = rerank(query, merged, model_name=reranker_model, top_k=top_k)
    return reranked
