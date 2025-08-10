from __future__ import annotations
from typing import List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE, HTTP_TIMEOUT_SECS


SYSTEM_TEMPLATE = (
    "You are a compliance assistant. Based on the following extracted policy/contract/document content, "
    "answer the user's question precisely.\n\nDocument Excerpts:\n{context}\n\nUser Question:\n{question}\n\nAnswer concisely and factually. If applicable, mention conditions or limitations."
)


class LLMError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), reraise=True,
       retry=retry_if_exception_type(httpx.HTTPError))
async def answer_with_openai(context_blocks: List[str], question: str) -> str:
    if not OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY not set")

    prompt = SYSTEM_TEMPLATE.format(context="\n".join(context_blocks), question=question)

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert compliance and policy analysis assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": OPENAI_MAX_TOKENS,
        "temperature": OPENAI_TEMPERATURE,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1
    }

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        
        # Extract response text from OpenAI API response
        text = ""
        try:
            choices = data.get("choices", [])
            if choices and len(choices) > 0:
                text = choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass
        
        if not text:
            text = (data.get("text") or "").strip()
        
        return text or ""
