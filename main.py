from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import ALLOWED_ORIGINS, APP_NAME, APP_VERSION
from backend.consultant import build_advice, classify_non_product_request
from backend.llm import generate_llm_answer, llm_health
from backend.rag import ProductRAG

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Professional deterministic product RAG + LLM consultant for Micron Tools.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

rag = ProductRAG()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


def public_product(product: dict) -> dict:
    return {
        "name": product.get("name", ""),
        "brand": product.get("brand", ""),
        "sku": product.get("sku", ""),
        "price": product.get("price", ""),
        "url": product.get("url", ""),
        "image": product.get("image", ""),
        "category": product.get("category", ""),
        "score": product.get("_score"),
    }


@app.get("/")
def home():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "status": "ok",
        "message": f"{APP_NAME} is running.",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "products": len(rag.products),
        "rag_ready": rag.ready,
        "llm": llm_health(),
        "version": APP_VERSION,
    }


@app.get("/search")
def search(q: str, top_k: int = 5):
    top_k = max(1, min(top_k, 10))
    hits = rag.search(q, top_k=top_k)
    return {
        "query": q,
        "results": [public_product(h) for h in hits],
        "results_found": len(hits),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    message = req.message.strip()

    # 1) Fast path – greeting / contact / thanks / goodbye / about
    non_product = classify_non_product_request(message)
    if non_product:
        return {
            "answer": non_product,
            "products": [],
            "products_found": 0,
            "source": "deterministic",
        }

    # 2) Deterministic RAG search
    hits = rag.search(message, top_k=req.top_k)

    # 3) No verified match
    if not hits:
        return {
            "answer": rag.no_match_message(message),
            "products": [],
            "products_found": 0,
            "source": "rag",
        }

    # 4) Specific request → only top 1 product
    #    General request → up to 3 products
    parsed = rag.parse_request(message)
    is_specific = bool(
        parsed.get("diameter")
        or parsed.get("thread")
        or parsed.get("material")
    )
    selected = hits[:1] if is_specific else hits[:3]

    # 5) LLM only explains the selection (never invents)
    context = rag.build_context(selected)
    answer = generate_llm_answer(user_message=message, context=context)

    # 6) Safe fallback if LLM is down
    if not answer:
        answer = build_advice(message, selected)

    return {
        "answer": answer,
        "products": [public_product(p) for p in selected],
        "products_found": len(selected),
        "source": "rag+llm" if answer else "rag",
        "parsed": {
            "intent": parsed.get("intent"),
            "diameter": parsed.get("diameter"),
            "thread": parsed.get("thread"),
            "material": parsed.get("material"),
        },
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

