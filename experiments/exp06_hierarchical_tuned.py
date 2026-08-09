"""
Experiment 06 — Hierarchical RAG (Tuned): Fine-tuned BGE + Parameter Search
============================================================================
Exp03 used general BAAI/bge-large-en-v1.5 and underperformed exp02 (hybrid).
The hypothesis here is that the regression was caused by the embedding model,
not the hierarchical architecture itself.

Two configurations tested:

  Config A — same structure as exp03, upgraded embedding:
    Hierarchy  : 2048w → 512w → 256w  (same as exp03)
    Embedding  : Fine-tuned BGE  (planning domain — best from exp04b)
    Merge ratio: 0.5
    Pool       : 40  (was 30 in exp03 — larger pool gives AutoMerge more material)

  Config B — larger chunks at every level:
    Hierarchy  : 4096w → 1024w → 512w
    Embedding  : Fine-tuned BGE
    Merge ratio: 0.5
    Pool       : 40
    Rationale  : 512-word leaves capture a complete policy clause — richer
                 than 256 words for embedding. 4096-word grandparents cover
                 an entire policy chapter when merged.

Both configs use the same hybrid retrieval pipeline:
  BM25 + Dense → RRF → AutoMerge → BGE cross-encoder rerank

Outputs:
  outputs/exp06a_hier_tuned_results.json
  outputs/exp06b_hier_tuned_results.json
  outputs/exp06_hier_tuned_ragas_scores.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table

from src.auto_merger import build_chunk_lookup, hybrid_hierarchical_query
from src.config import get_settings, resolve_endpoint_config
from src.evaluator import (
    evaluate_citation_quality,
    load_golden_dataset,
    run_ragas_evaluation,
    save_ragas_scores,
)
from src.generator import answer_question
from src.indexer import build_index, get_embedding_model, load_chunks
from src.ingest import run_phase0_hierarchical
from src.retriever import BM25Index

console = Console()

FINETUNED_BGE  = Path("outputs") / "bge-planning-finetuned"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MERGE_RATIO    = 0.5
CANDIDATE_POOL = 40

CONFIGS = {
    "A": {
        "label":            "FT-BGE + 2048/512/256 (pool=40)",
        "hierarchy":        (2048, 512, 256),
        "collection_name":  "hier_tuned_a_256",
        "chunks_file":      "chunks_hier_a_2048_512_256.jsonl",
        "leaves_file":      "chunks_hier_a_2048_512_256_leaves.jsonl",
        "output_key":       "exp06a",
    },
    "B": {
        "label":            "FT-BGE + 4096/1024/512 (pool=40)",
        "hierarchy":        (4096, 1024, 512),
        "collection_name":  "hier_tuned_b_512",
        "chunks_file":      "chunks_hier_b_4096_1024_512.jsonl",
        "leaves_file":      "chunks_hier_b_4096_1024_512_leaves.jsonl",
        "output_key":       "exp06b",
    },
}

# Prior experiment scores for comparison table
PRIOR = {
    "exp01": {"faithfulness": 0.8788, "answer_relevancy": 0.7904,
              "context_precision": 0.6475, "context_recall": 0.6883, "citation_quality": 0.9320},
    "exp02": {"faithfulness": 0.8981, "answer_relevancy": 0.9115,
              "context_precision": 0.9481, "context_recall": 0.8508, "citation_quality": 0.9520},
    "exp03": {"faithfulness": 0.8553, "answer_relevancy": 0.8837,
              "context_precision": 0.9136, "context_recall": 0.7850, "citation_quality": 0.9010},
    "exp04b": {"faithfulness": 0.9022, "answer_relevancy": 0.9111,
               "context_precision": 0.9459, "context_recall": 0.8608, "citation_quality": 0.9520},
    "exp05": {"faithfulness": 0.9074, "answer_relevancy": 0.9113,
              "context_precision": 0.9506, "context_recall": 0.8542, "citation_quality": 0.9270},
}


def run_config(
    cfg: dict,
    settings,
    ft_model_path: Path,
    golden_dataset: list[dict],
    openai_api_key: str,
    openai_base_url: str,
    openai_api_version: str,
) -> dict:
    label      = cfg["label"]
    hierarchy  = cfg["hierarchy"]
    collection = cfg["collection_name"]
    output_dir = Path(settings.output_dir)

    hier_chunks_path  = output_dir / cfg["chunks_file"]
    leaf_chunks_path  = output_dir / cfg["leaves_file"]

    console.rule(f"[bold cyan]Config {cfg['output_key'].upper()}: {label}[/bold cyan]")
    console.print(f"  Hierarchy : {hierarchy[0]}w → {hierarchy[1]}w → {hierarchy[2]}w (leaves)")
    console.print(f"  Embedding : {ft_model_path.name}")
    console.print(f"  Pool      : {CANDIDATE_POOL}  |  Merge ratio: {MERGE_RATIO}  |  Top-k: {settings.top_k}")

    # ── Build hierarchy ──────────────────────────────────────────────────────
    if not hier_chunks_path.exists():
        console.print(f"[yellow]Building hierarchical chunks {hierarchy}...[/yellow]")
        run_phase0_hierarchical(
            dataset_path=settings.dataset_path,
            output_path=hier_chunks_path,
            sizes=hierarchy,
            max_files=settings.max_files,
        )
    else:
        console.print(f"[green]Hierarchical chunks exist — skipping ingestion.[/green]")

    # ── Filter leaves ────────────────────────────────────────────────────────
    if not leaf_chunks_path.exists():
        console.print("[yellow]Filtering leaf chunks (level 2)...[/yellow]")
        with hier_chunks_path.open("r", encoding="utf-8") as src, \
             leaf_chunks_path.open("w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    chunk = json.loads(line)
                    if chunk.get("metadata", {}).get("level") == 2:
                        dst.write(line)

    leaf_chunks = load_chunks(leaf_chunks_path)
    console.print(f"[green]{len(leaf_chunks)} leaf chunks ready[/green]")

    # ── Build ChromaDB index with fine-tuned BGE ─────────────────────────────
    index_summary = build_index(
        chunks_path=leaf_chunks_path,
        chroma_path=settings.chroma_path,
        collection_name=collection,
        model_name=str(ft_model_path),
    )

    # ── BM25 + embedding model ────────────────────────────────────────────────
    bm25_index  = BM25Index(leaf_chunks)
    embed_model = get_embedding_model(str(ft_model_path), device="cpu")

    # ── Full hierarchy lookup for auto-merging ────────────────────────────────
    chunk_lookup = build_chunk_lookup(hier_chunks_path)
    console.print(f"[green]Loaded {len(chunk_lookup)} total chunks (all levels)[/green]")

    # ── Retrieval + Generation ────────────────────────────────────────────────
    queries = [item["question"] for item in golden_dataset]
    console.print(f"Running {len(queries)} queries...")

    results: list[dict] = []
    for i, query in enumerate(queries, 1):
        if i % 10 == 0:
            console.print(f"  Query {i}/{len(queries)}...")

        matches = hybrid_hierarchical_query(
            query=query,
            bm25_index=bm25_index,
            embed_model=embed_model,
            chroma_path=settings.chroma_path,
            collection_name=collection,
            chunk_lookup=chunk_lookup,
            reranker_model=RERANKER_MODEL,
            top_k=settings.top_k,
            candidate_pool=CANDIDATE_POOL,
            merge_ratio=MERGE_RATIO,
        )

        answer = ""
        if settings.llm_base_url and settings.llm_api_key:
            try:
                answer = answer_question(
                    question=query,
                    matches=matches,
                    model=settings.llm_model,
                    base_url=settings.llm_base_url,
                    api_key=settings.llm_api_key,
                    api_version=settings.llm_api_version,
                )
            except Exception as exc:
                console.print(f"[red]LLM call failed: {exc}[/red]")

        results.append({"query": query, "matches": matches, "answer": answer})

    # ── RAGAS + Citation ──────────────────────────────────────────────────────
    ragas_scores: dict = {}
    if openai_base_url and openai_api_key:
        try:
            ragas_scores = run_ragas_evaluation(
                results=results,
                golden_dataset=golden_dataset,
                eval_model=settings.eval_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                openai_api_version=openai_api_version,
            )
            citation_scores = evaluate_citation_quality(
                results=results,
                eval_model=settings.eval_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                openai_api_version=openai_api_version,
            )
            ragas_scores["aggregate"]["citation_quality"] = citation_scores["aggregate"]["citation_quality"]
            ragas_scores["citation_per_question"] = citation_scores["per_question"]

            score_path = output_dir / f"{cfg['output_key']}_ragas_scores.json"
            save_ragas_scores(ragas_scores, score_path)
        except Exception as exc:
            console.print(f"[red]RAGAS failed: {exc}[/red]")

    # ── Save results ──────────────────────────────────────────────────────────
    result_path = output_dir / f"{cfg['output_key']}_hier_tuned_results.json"
    payload = {
        "experiment":  cfg["output_key"],
        "label":       label,
        "description": f"Hierarchical RAG (FT-BGE, {hierarchy}, pool={CANDIDATE_POOL})",
        "retrieval_config": {
            "embedding_model":   str(ft_model_path),
            "hierarchy_words":   list(hierarchy),
            "indexed_level":     f"leaf ({hierarchy[2]} words)",
            "retrieval":         f"BM25 + FT-BGE → RRF → AutoMerge → {RERANKER_MODEL}",
            "candidate_pool":    CANDIDATE_POOL,
            "merge_ratio":       MERGE_RATIO,
            "final_top_k":       settings.top_k,
        },
        "index_summary":  index_summary,
        "results":        results,
        "ragas_scores":   ragas_scores,
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved to {result_path}[/green]")

    return {"label": label, "ragas_scores": ragas_scores}


def print_comparison(all_results: dict) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision",
               "context_recall", "citation_quality"]

    table = Table(title="Hierarchical RAG — Tuning Comparison")
    table.add_column("Metric",          style="bold")
    table.add_column("Exp01\nBaseline", justify="right")
    table.add_column("Exp02\nHybrid",   justify="right")
    table.add_column("Exp03\nGeneral\nBGE", justify="right")
    table.add_column("Exp04b\nFT-BGE",  justify="right")
    table.add_column("Exp05\nGraph RAG", justify="right")
    table.add_column("Exp06a\nFT+256w", justify="right", style="cyan")
    table.add_column("Exp06b\nFT+512w", justify="right", style="green")

    def fmt(d: dict, m: str) -> str:
        v = d.get(m)
        return f"{v:.4f}" if isinstance(v, float) else "—"

    agg_a = all_results.get("A", {}).get("ragas_scores", {}).get("aggregate", {})
    agg_b = all_results.get("B", {}).get("ragas_scores", {}).get("aggregate", {})

    for m in metrics:
        table.add_row(
            m,
            fmt(PRIOR["exp01"],  m),
            fmt(PRIOR["exp02"],  m),
            fmt(PRIOR["exp03"],  m),
            fmt(PRIOR["exp04b"], m),
            fmt(PRIOR["exp05"],  m),
            fmt(agg_a, m),
            fmt(agg_b, m),
        )
    console.print(table)


def main() -> None:
    settings = get_settings()
    project_root = Path(__file__).resolve().parent.parent

    console.print("[bold blue]Experiment 06: Hierarchical RAG (Fine-tuned BGE, Parameter Tuning)[/bold blue]")
    console.print("  Hypothesis: exp03 regressed because it used general BGE.")
    console.print("  Fix: use fine-tuned BGE (exp04b model) + larger candidate pool.")
    console.print("  Also test 512-word leaves (Config B) vs 256-word leaves (Config A).")

    # ── Resolve fine-tuned model ──────────────────────────────────────────────
    ft_model_path = project_root / FINETUNED_BGE
    if not ft_model_path.exists():
        console.print(f"[red]Fine-tuned BGE not found at {ft_model_path}[/red]")
        console.print("[yellow]Run: python scripts/finetune_embeddings.py first[/yellow]")
        return

    # ── Load golden dataset ───────────────────────────────────────────────────
    golden_path = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    if not golden_dataset:
        console.print("[red]No golden dataset found.[/red]")
        return

    openai_base_url, openai_url_version = resolve_endpoint_config(os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key     = os.getenv("OPENAI_API_KEY", "")
    openai_api_version = os.getenv("OPENAI_API_VERSION", "") or openai_url_version

    # ── Run both configs ──────────────────────────────────────────────────────
    all_results: dict = {}
    for key, cfg in CONFIGS.items():
        result = run_config(
            cfg=cfg,
            settings=settings,
            ft_model_path=ft_model_path,
            golden_dataset=golden_dataset,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_api_version=openai_api_version,
        )
        all_results[key] = result

    print_comparison(all_results)


if __name__ == "__main__":
    main()
