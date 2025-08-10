from __future__ import annotations
from typing import List
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE, HTTP_TIMEOUT_SECS, MAX_CONCURRENT_LLM_CALLS

# IMPROVED SYSTEM TEMPLATE for concise, clean responses
SYSTEM_TEMPLATE = (
    "You are an expert compliance and policy analysis assistant. Provide precise, factual answers "
    "based on the provided document excerpts.\n\n"
    "IMPORTANT GUIDELINES:\n"
    "1. Answer ONLY based on the provided document content\n"
    "2. Be specific with numbers, dates, and conditions\n"
    "3. If information is not in the document, say 'Information not found in the document'\n"
    "4. Provide concise, single-paragraph answers without bullet points or formatting\n"
    "5. Use clear, professional language\n"
    "6. Include key details but keep responses focused and direct\n\n"
    "Document Excerpts:\n{context}\n\n"
    "User Question: {question}\n\n"
    "Provide a clear, concise answer in a single paragraph:"
)

# SEMAPHORE for controlling concurrent LLM calls
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)


class LLMError(Exception):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), reraise=True,
       retry=retry_if_exception_type(httpx.HTTPError))
async def answer_with_openai(context_blocks: List[str], question: str) -> str:
    if not OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY not set")

    # Limit context length to prevent token overflow
    max_context_length = 12000  # Increased for OpenAI's better context handling
    context_text = "\n".join(context_blocks)
    if len(context_text) > max_context_length:
        # Truncate while keeping most relevant parts
        context_text = context_text[:max_context_length] + "\n[Content truncated for length]"

    prompt = SYSTEM_TEMPLATE.format(context=context_text, question=question)

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

    async with _llm_semaphore:  # Control concurrent calls
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
            
            # Clean up the response to ensure single-paragraph format
            if text:
                # Remove bullet points and excessive formatting
                text = text.replace("*", "").replace("•", "").replace("-", "")
                # Remove extra newlines and spaces
                text = " ".join(text.split())
                # Ensure it's a single paragraph
                text = text.replace("\n", " ").strip()
            
            return text or "Information not found in the document."
