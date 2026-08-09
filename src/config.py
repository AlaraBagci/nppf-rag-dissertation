from __future__ import annotations

import os
from urllib.parse import parse_qs, unquote, urlparse
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_endpoint(value: str) -> str:
    if not value:
        return value
    parsed = urlparse(value)
    if parsed.netloc.endswith("safelinks.protection.outlook.com"):
        wrapped = parse_qs(parsed.query).get("url", [""])[0]
        if wrapped:
            return unquote(wrapped)
    return value

def resolve_endpoint_config(value: str) -> tuple[str, str]:
    if not value:
        return "", ""

    parsed = urlparse(value)
    if parsed.netloc.endswith("safelinks.protection.outlook.com"):
        wrapped = parse_qs(parsed.query).get("url", [""])[0]
        if wrapped:
            value = unquote(wrapped)
            parsed = urlparse(value)

    api_version = parse_qs(parsed.query).get("api-version", [""])[0]
    endpoint = normalize_endpoint(value)
    return endpoint, api_version


@dataclass(frozen=True)
class Settings:
    dataset_path: Path = Path(os.getenv("DATASET_PATH", PROJECT_ROOT / "data" / "dataset"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    chunks_path: Path = Path(os.getenv("CHUNKS_PATH", PROJECT_ROOT / "outputs" / "chunks.jsonl"))
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", PROJECT_ROOT / "outputs" / "chroma_db"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    llm_base_url: str = normalize_endpoint(os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "")))
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_api_version: str = os.getenv("LLM_API_VERSION", "")
    llm_model: str = os.getenv("LLM_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")
    eval_model: str = os.getenv("EVAL_MODEL", "gpt-4.1")
    openai_api_version: str = os.getenv("OPENAI_API_VERSION", "")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_files: int | None = int(os.getenv("MAX_FILES", "0")) or None


def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
