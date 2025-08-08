from __future__ import annotations
from typing import List
import numpy as np
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import GEMINI_API_KEY, HTTP_TIMEOUT_SECS


class EmbeddingError(Exception):
    pass


async def _embed_text_once(client: httpx.AsyncClient, text: str) -> List[float]:
    if not GEMINI_API_KEY:
        raise EmbeddingError("GEMINI_API_KEY not set")

    v1_url = "https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent"
    v1_payload = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}

    try:
        r = await client.post(f"{v1_url}?key={GEMINI_API_KEY}", json=v1_payload)
        r.raise_for_status()
        data = r.json()
        values = data.get("embedding", {}).get("values") or data.get("embedding", {}).get("value")
        if not values and "embeddings" in data:
            values = data["embeddings"][0]["values"]
        if not values:
            raise EmbeddingError("No embedding values returned")
        return values
    except httpx.HTTPStatusError:
        v1beta_url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedText"
        v1beta_payload = {"text": text}
        rb = await client.post(f"{v1beta_url}?key={GEMINI_API_KEY}", json=v1beta_payload)
        rb.raise_for_status()
        data = rb.json()
        values = data.get("embedding", {}).get("value") or data.get("embedding", {}).get("values")
        if not values and "embeddings" in data:
            values = data["embeddings"][0]["values"]
        if not values:
            raise EmbeddingError("No embedding values returned (v1beta)")
        return values


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True,
       retry=retry_if_exception_type((httpx.HTTPError, EmbeddingError)))
async def embed_texts(texts: List[str]) -> np.ndarray:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
        vectors: List[List[float]] = []
        for t in texts:
            values = await _embed_text_once(client, t)
            vectors.append(values)
        arr = np.array(vectors, dtype=np.float32)
        return arr


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True,
       retry=retry_if_exception_type((httpx.HTTPError, EmbeddingError)))
async def embed_query(text: str) -> np.ndarray:
    vecs = await embed_texts([text])
    return vecs[0]
