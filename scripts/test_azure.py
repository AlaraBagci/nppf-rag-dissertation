from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from src.config import resolve_endpoint_config

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_VERSION = os.getenv("LLM_API_VERSION", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "")
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4.1")

print("=== Config ===")
print(f"LLM_BASE_URL     : {LLM_BASE_URL}")
print(f"LLM_API_VERSION  : {LLM_API_VERSION}")
print(f"LLM_MODEL        : {LLM_MODEL}")
print(f"LLM_API_KEY      : {LLM_API_KEY[:8]}...{LLM_API_KEY[-4:]}" if LLM_API_KEY else "LLM_API_KEY      : NOT SET")
print()
print(f"OPENAI_BASE_URL  : {OPENAI_BASE_URL}")
print(f"OPENAI_API_VERSION: {OPENAI_API_VERSION}")
print(f"EVAL_MODEL       : {EVAL_MODEL}")
print(f"OPENAI_API_KEY   : {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "OPENAI_API_KEY   : NOT SET")
print()


def test_endpoint(label: str, base_url: str, api_key: str, api_version: str, model: str) -> None:
    print(f"--- Testing {label} ---")
    if not base_url or not api_key:
        print("SKIP: URL or key not set\n")
        return
    try:
        resolved_base_url, resolved_url_version = resolve_endpoint_config(base_url)
        resolved_api_version = api_version or resolved_url_version
        client = OpenAI(
            base_url=resolved_base_url,
            api_key=api_key,
            default_headers={"api-key": api_key},
            default_query={"api-version": resolved_api_version} if resolved_api_version else {},
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
            temperature=0,
        )
        print(f"SUCCESS: {response.choices[0].message.content!r}\n")
    except Exception as e:
        print(f"FAILED : {e}\n")


test_endpoint(
    label="Llama-4-Maverick (Azure AI Foundry)",
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    api_version=LLM_API_VERSION,
    model=LLM_MODEL,
)

test_endpoint(
    label="GPT-4.1 (Azure OpenAI)",
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    model=EVAL_MODEL,
)
