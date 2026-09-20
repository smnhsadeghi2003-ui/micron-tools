from __future__ import annotations

import requests

from backend.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TAGS_URL,
    OLLAMA_URL,
    OPENAI_API_KEY,
    USE_OLLAMA,
    XAI_API_KEY,
)


SYSTEM_RULES = """
You are the technical sales consultant of Micron Tools (industrial cutting tools, Tehran).

Task:
Write a very short and precise explanation why the selected product(s) match the customer's request.

STRICT RULES:
1) Use ONLY the provided product information in Context. Never invent specs, sizes, materials or features.
2) Do not repeat the full product name.
3) Do not write links, prices, SKU or contact information.
4) Do not talk about any product that is not in the Context.
5) Maximum 2-3 short sentences.
6) Reply in the SAME language as the customer (Persian or English).
7) If evidence is insufficient, write exactly:
   - Persian: «بر اساس اطلاعات موجود، این محصول با درخواست شما مطابقت دارد.»
   - English: «Based on the available information, this product matches your request.»
8) For diameter and thread: only confirm what is explicitly present in Context.
9) Never use uncertain language ("probably", "seems", "might", "احتمالاً", "به نظر می‌رسد").
10) Be professional, clear and helpful.
"""


def _call_ollama(user_message: str, context: str) -> str | None:
    if not USE_OLLAMA:
        return None

    prompt = f"""{SYSTEM_RULES}

Customer question:
{user_message}

Real product information:
{context}

Write only the final answer.
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        answer = (data.get("response") or "").strip()
        return answer or None
    except Exception as exc:
        print("OLLAMA ERROR:", repr(exc))
        return None


def _call_cloud(user_message: str, context: str) -> str | None:
    api_key = OPENAI_API_KEY or GROQ_API_KEY or XAI_API_KEY
    if not api_key:
        return None

    base_url = "https://api.openai.com/v1"
    model = LLM_MODEL or "gpt-4o-mini"

    if GROQ_API_KEY and not OPENAI_API_KEY:
        base_url = "https://api.groq.com/openai/v1"
        model = LLM_MODEL or "llama-3.3-70b-versatile"
    elif XAI_API_KEY and not OPENAI_API_KEY and not GROQ_API_KEY:
        base_url = "https://api.x.ai/v1"
        model = LLM_MODEL or "grok-2-latest"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_RULES},
                {
                    "role": "user",
                    "content": (
                        f"Customer question:\n{user_message}\n\n"
                        f"Real product information:\n{context}\n\n"
                        "Write only the final answer."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:
        print("CLOUD LLM ERROR:", repr(exc))
        return None


def generate_llm_answer(user_message: str, context: str) -> str | None:
    """Try Ollama first, then optional cloud providers."""
    answer = _call_ollama(user_message, context)
    if answer:
        return answer
    return _call_cloud(user_message, context)


def llm_health() -> dict:
    status: dict = {
        "ollama": {
            "enabled": USE_OLLAMA,
            "available": False,
            "model": OLLAMA_MODEL,
        },
        "cloud": {
            "openai": bool(OPENAI_API_KEY),
            "groq": bool(GROQ_API_KEY),
            "xai": bool(XAI_API_KEY),
        },
    }

    if USE_OLLAMA:
        try:
            response = requests.get(OLLAMA_TAGS_URL, timeout=3)
            response.raise_for_status()
            status["ollama"]["available"] = True
        except Exception as exc:
            status["ollama"]["error"] = str(exc)

    return status
