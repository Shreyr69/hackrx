from __future__ import annotations
from typing import List
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import GEMINI_API_KEY, HTTP_TIMEOUT_SECS, MAX_CONCURRENT_LLM_CALLS

# IMPROVED SYSTEM TEMPLATE for better accuracy
SYSTEM_TEMPLATE = (
    "You are an expert compliance and policy analysis assistant. Your task is to provide precise, factual answers "
    "based on the provided document excerpts.\n\n"
    "IMPORTANT GUIDELINES:\n"
    "1. Answer ONLY based on the provided document content\n"
    "2. Be specific with numbers, dates, and conditions\n"
    "3. If information is not in the document, say 'Information not found in the document'\n"
    "4. Use bullet points for multiple conditions or requirements\n"
    "5. Quote exact text when possible\n\n"
    "Document Excerpts:\n{context}\n\n"
    "User Question: {question}\n\n"
    "Provide a clear, structured answer:"
)

# SEMAPHORE for controlling concurrent LLM calls
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)


class LLMError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), reraise=True,
       retry=retry_if_exception_type(httpx.HTTPError))
async def answer_with_gemini(context_blocks: List[str], question: str) -> str:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY not set")

    # Limit context length to prevent token overflow
    max_context_length = 8000  # characters
    context_text = "\n".join(context_blocks)
    if len(context_text) > max_context_length:
        # Truncate while keeping most relevant parts
        context_text = context_text[:max_context_length] + "\n[Content truncated for length]"

    prompt = SYSTEM_TEMPLATE.format(context=context_text, question=question)

    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,  # Lower temperature for more consistent answers
            "maxOutputTokens": 1000,  # Limit response length
            "topP": 0.8,
            "topK": 40
        }
    }

    async with _llm_semaphore:  # Control concurrent calls
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
            r = await client.post(f"{url}?key={GEMINI_API_KEY}", json=payload)
            r.raise_for_status()
            data = r.json()
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
                text = (data.get("text") or "").strip()
            return text or "Information not found in the document."


# NEW: Batch processing for multiple questions
async def answer_multiple_questions(context_blocks: List[str], questions: List[str]) -> List[str]:
    """Process multiple questions concurrently with controlled parallelism"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
    
    async def answer_single(q: str) -> str:
        async with semaphore:
            return await answer_with_gemini(context_blocks, q)
    
    # Process questions in batches
    batch_size = MAX_CONCURRENT_LLM_CALLS
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_results = await asyncio.gather(*[answer_single(q) for q in batch])
        results.extend(batch_results)
    
    return results
