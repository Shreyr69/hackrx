from __future__ import annotations
from typing import List
import numpy as np
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import OPENAI_API_KEY, HTTP_TIMEOUT_SECS


class EmbeddingError(Exception):
    pass


async def _embed_text_once(client: httpx.AsyncClient, text: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise EmbeddingError("OPENAI_API_KEY not set")

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "text-embedding-3-small",  # Best performing embedding model
        "input": text,
        "encoding_format": "float"
    }

    try:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        values = data.get("data", [{}])[0].get("embedding")
        if not values:
            raise EmbeddingError("No embedding values returned")
        return values
    except Exception as e:
        raise EmbeddingError(f"Embedding failed: {str(e)}")


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
