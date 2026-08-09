from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    source_path: str
    collection: str
    region: str | None
    top_folder: str | None
    sub_topic: str | None
    document_class: str | None
    page_start: int | None
    page_end: int | None
    chunk_index: int
    text: str
    hash: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
