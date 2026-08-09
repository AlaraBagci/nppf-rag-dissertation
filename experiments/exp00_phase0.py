from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from src.config import get_settings
from src.ingest import run_phase0


load_dotenv()
console = Console()


def main() -> None:
    settings = get_settings()
    summary = run_phase0(
        dataset_path=settings.dataset_path,
        output_path=settings.chunks_path,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        max_files=settings.max_files,
    )
    output_path = Path(settings.output_dir) / "phase0_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"[green]Saved phase 0 summary to {output_path}[/green]")


if __name__ == "__main__":
    main()
