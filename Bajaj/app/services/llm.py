from __future__ import annotations
from typing import List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import GEMINI_API_KEY, HTTP_TIMEOUT_SECS


SYSTEM_TEMPLATE = (
    "You are a compliance assistant. Based on the following extracted policy/contract/document content, "
    "answer the user’s question precisely.\n\nDocument Excerpts:\n{context}\n\nUser Question:\n{question}\n\nAnswer concisely and factually. If applicable, mention conditions or limitations."
)


class LLMError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), reraise=True,
       retry=retry_if_exception_type(httpx.HTTPError))
async def answer_with_gemini(context_blocks: List[str], question: str) -> str:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY not set")

    prompt = SYSTEM_TEMPLATE.format(context="\n".join(context_blocks), question=question)

    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
        r = await client.post(f"{url}?key={GEMINI_API_KEY}", json=payload)
        r.raise_for_status()
        data = r.json()
        # Extract text from candidates
        text = ""
        try:
            candidates = data.get("candidates", [])
            for c in candidates:
                parts = ((c.get("content") or {}).get("parts") or [])
                for p in parts:
                    if "text" in p:
                        text += p["text"]
            text = text.strip()
        except Exception:
            pass
        if not text:
            # Fallback for alternate shapes
            text = (data.get("text") or "").strip()
        return text or ""
