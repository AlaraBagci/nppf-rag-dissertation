from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI

from src.config import resolve_endpoint_config


def build_client(base_url: str | None = None, api_key: str | None = None, api_version: str | None = None) -> OpenAI:
    resolved_base_url, resolved_url_version = resolve_endpoint_config(
        base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    )
    resolved_api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    resolved_api_version = api_version or os.getenv("LLM_API_VERSION") or resolved_url_version or ""
    if not resolved_base_url:
        raise ValueError("LLM_BASE_URL is required for generation")
    azure_headers = {"api-key": resolved_api_key}
    if resolved_api_version:
        return OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            default_headers=azure_headers,
            default_query={"api-version": resolved_api_version},
        )
    return OpenAI(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        default_headers=azure_headers,
    )


def compose_context(matches: Iterable[dict], max_chars: int = 16000) -> str:
    blocks: list[str] = []
    total = 0
    for idx, match in enumerate(matches, start=1):
        meta = match.get("metadata", {})
        header = f"[Source {idx}] {meta.get('source_path', 'unknown')} | {meta.get('document_class', 'unknown')}"
        body = match.get("text", "")
        block = f"{header}\n{body}".strip()
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def answer_question(question: str, matches: Iterable[dict], model: str, base_url: str, api_key: str, api_version: str = "") -> str:
    client = build_client(base_url=base_url, api_key=api_key, api_version=api_version)
    context = compose_context(matches)
    prompt = f"""You are a UK planning policy expert. Answer the question using ONLY the provided source documents.

STRICT RULES — every single sentence that contains a fact MUST have an inline citation. No exceptions.
- Inline citation format: (Document Name, Policy/Paragraph/Section identifier)
  Examples: (NPPF, paragraph 11) | (Coventry Local Plan, Policy DS1) | (Coventry SPD, Section 3.2)
- If a sentence has no citation, it must be removed.
- If the context does not contain enough information, respond with: "The provided documents do not contain sufficient information to answer this question."
- Do NOT add any background knowledge, general statements, or introductory sentences that are not directly sourced from the context.
- End every answer with a References section.

Question:
{question}

Context:
{context}

Answer format:
[Every sentence cites its source inline, e.g.: "New residential developments must achieve a minimum density of 40 dwellings per hectare in accessible locations (Coventry Local Plan, Policy DS1)."]

References:
- [Source N]: [Exact document name] — [policy code / paragraph / section]
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful RAG assistant for UK planning policy."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""
