from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before src imports so os.getenv() sees .env values

from rich.console import Console

from src.config import get_settings
from src.config import resolve_endpoint_config
from src.evaluator import load_golden_dataset, run_ragas_evaluation, save_ragas_scores, evaluate_citation_quality
from src.generator import answer_question
from src.indexer import build_index, query_index
from src.ingest import run_phase0

console = Console()

TEST_QUERIES = [
    "What is Coventry's policy on housing density?",
    "What are the design requirements for new residential development in Coventry?",
    "How does Coventry's Local Plan address affordable housing compared with national policy?",
    "What is the national Green Belt policy and how does Coventry interpret it?",
    "What does the NPPF say about the presumption in favour of sustainable development?",
]

INDEX_LIMIT = int(os.getenv("INDEX_LIMIT", "500")) or None


def _run_phase0(settings) -> None:
    run_phase0(
        dataset_path=settings.dataset_path,
        output_path=settings.chunks_path,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        max_files=settings.max_files,
    )


def _validate_jsonl(path: Path) -> tuple[bool, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    json.loads(line)
    except Exception as exc:
        return False, f"line {line_number}: {exc}"
    return True, None


def _rebuild_chroma_dir(chroma_path: Path) -> None:
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    settings = get_settings()
    console.print("[bold blue]Phase 1: Naive Baseline RAG[/bold blue]")
    console.print(f"Dataset: {settings.dataset_path}")
    console.print("Venv GPU check will use the active torch install automatically.")

    if not settings.chunks_path.exists():
        console.print("[yellow]No chunks file found. Running phase 0 ingestion first.[/yellow]")
        _run_phase0(settings)

    valid_chunks, chunks_error = _validate_jsonl(settings.chunks_path)
    if not valid_chunks:
        console.print(f"[yellow]Chunks file is malformed ({chunks_error}). Re-running phase 0 ingestion.[/yellow]")
        _run_phase0(settings)

    try:
        index_summary = build_index(
            chunks_path=settings.chunks_path,
            chroma_path=settings.chroma_path,
            collection_name="baseline_naive",
            model_name=settings.embedding_model,
            max_chunks=INDEX_LIMIT,
        )
    except Exception as exc:
        if "database disk image is malformed" in str(exc).lower():
            console.print("[yellow]Detected corrupted Chroma database. Rebuilding index storage and retrying.[/yellow]")
            _rebuild_chroma_dir(settings.chroma_path)
            index_summary = build_index(
                chunks_path=settings.chunks_path,
                chroma_path=settings.chroma_path,
                collection_name="baseline_naive",
                model_name=settings.embedding_model,
                max_chunks=INDEX_LIMIT,
            )
        else:
            raise

    # Use golden dataset questions if available — ensures RAGAS can always evaluate
    golden_path = Path(settings.dataset_path).parent / "golden_dataset.json"
    golden_dataset = load_golden_dataset(golden_path)
    queries = [item["question"] for item in golden_dataset] if golden_dataset else TEST_QUERIES
    console.print(f"Running {len(queries)} queries ({'from golden dataset' if golden_dataset else 'hardcoded fallback'})")

    results: list[dict] = []
    for query in queries:
        matches = query_index(
            query=query,
            chroma_path=settings.chroma_path,
            collection_name="baseline_naive",
            model_name=settings.embedding_model,
            top_k=settings.top_k,
            device="cpu",
        )
        answer = ""
        if settings.llm_base_url and settings.llm_api_key:
            console.print(f"[dim]Calling LLM at {settings.llm_base_url} ...[/dim]")
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
        else:
            console.print("[yellow]LLM_BASE_URL or LLM_API_KEY not set — skipping generation.[/yellow]")
        results.append({"query": query, "matches": matches, "answer": answer})

    ragas_scores: dict = {}
    openai_base_url, openai_url_version = resolve_endpoint_config(os.getenv("OPENAI_BASE_URL", ""))
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_version = os.getenv("OPENAI_API_VERSION", "") or openai_url_version
    can_run_ragas = bool(golden_dataset and openai_base_url and openai_api_key)

    if can_run_ragas:
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
            ragas_output = Path(settings.output_dir) / "exp01_ragas_scores.json"
            save_ragas_scores(ragas_scores, ragas_output)
            console.print(f"[green]RAGAS output path: {ragas_output}[/green]")
        except Exception as exc:
            console.print(f"[red]RAGAS evaluation failed: {exc}[/red]")
    else:
        missing: list[str] = []
        if not golden_dataset:
            missing.append(f"golden dataset at {golden_path}")
        if not openai_base_url:
            missing.append("OPENAI_BASE_URL")
        if not openai_api_key:
            missing.append("OPENAI_API_KEY")
        missing_str = ", ".join(missing)
        console.print(f"[yellow]Skipping RAGAS. Missing prerequisites: {missing_str}[/yellow]")

    output_path = Path(settings.output_dir) / "exp01_baseline_results.json"
    payload = {
        "config": {
            "dataset_path": str(settings.dataset_path),
            "chunks_path": str(settings.chunks_path),
            "chroma_path": str(settings.chroma_path),
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "top_k": settings.top_k,
        },
        "index_summary": index_summary,
        "results": results,
        "ragas_scores": ragas_scores,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Saved baseline results to {output_path}[/green]")


if __name__ == "__main__":
    main()