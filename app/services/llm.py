from __future__ import annotations
from typing import List
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Simple in-memory cache for performance optimization
_response_cache = {}
_embedding_cache = {}

def get_cache_key(text: str, question: str) -> str:
    """Generate cache key for responses"""
    import hashlib
    combined = f"{text[:100]}_{question[:100]}"
    return hashlib.md5(combined.encode()).hexdigest()

def get_embedding_cache_key(text: str) -> str:
    """Generate cache key for embeddings"""
    import hashlib
    return hashlib.md5(text.encode()).hexdigest()

async def get_cached_embedding(text: str) -> List[float]:
    """Get cached embedding if available"""
    cache_key = get_embedding_cache_key(text)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    return None

def cache_embedding(text: str, embedding: List[float]):
    """Cache embedding for future use"""
    cache_key = get_embedding_cache_key(text)
    _embedding_cache[cache_key] = embedding
    # Limit cache size to prevent memory issues
    if len(_embedding_cache) > 1000:
        # Remove oldest entries
        oldest_keys = list(_embedding_cache.keys())[:100]
        for key in oldest_keys:
            del _embedding_cache[key]

from ..config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE, HTTP_TIMEOUT_SECS, MAX_CONCURRENT_LLM_CALLS

# ENHANCED INSURANCE-SPECIFIC SYSTEM TEMPLATE for better policy analysis
SYSTEM_TEMPLATE = (
    "You are an expert insurance policy analysis assistant specializing in Indian health insurance policies. "
    "Your task is to provide precise, factual answers based on the provided document excerpts.\n\n"
    "INSURANCE POLICY ANALYSIS REQUIREMENTS:\n"
    "1. Answer ONLY based on the provided document content - do not make assumptions\n"
    "2. Be extremely specific with policy numbers, IRDAI registration numbers, CIN numbers, and UIN codes\n"
    "3. Include exact coverage amounts, waiting periods, exclusions, and conditions\n"
    "4. Reference specific policy sections, clauses, and sub-clauses when mentioned\n"
    "5. Distinguish between different policy types (Easy Health, Arogya Sanjeevani, etc.)\n"
    "6. Include all relevant exceptions, sub-limits, and co-payment details\n"
    "7. If information is not explicitly in the document, say 'Information not found in the document'\n"
    "8. Use exact language from the document when possible for policy terms\n"
    "9. Include specific amounts, percentages, or time periods exactly as mentioned\n\n"
    "Document Excerpts:\n{context}\n\n"
    "User Question: {question}\n\n"
    "Provide a comprehensive, accurate answer with specific policy details:"
)

# SEMAPHORE for controlling concurrent LLM calls
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)


class LLMError(Exception):
    pass


def get_dynamic_max_tokens(question: str, context_length: int) -> int:
    """Dynamically allocate tokens based on question complexity and context"""
    base_tokens = OPENAI_MAX_TOKENS
    
    # Adjust based on question complexity
    question_lower = question.lower()
    if any(word in question_lower for word in ["explain", "detailed", "describe", "how does"]):
        base_tokens = min(2500, base_tokens + 500)  # More tokens for explanation questions
    elif any(word in question_lower for word in ["what is", "define", "list"]):
        base_tokens = max(1500, base_tokens - 500)  # Fewer tokens for simple questions
    
    # Adjust based on context length
    if context_length > 8000:
        base_tokens = min(3000, base_tokens + 500)  # More tokens for longer context
    
    return base_tokens

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), reraise=True,
       retry=retry_if_exception_type(httpx.HTTPError))
async def answer_with_openai(context_blocks: List[str], question: str) -> str:
    if not OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY not set")

    # Check cache first
    cache_key = get_cache_key("\n".join(context_blocks), question)
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    # Limit context length to prevent token overflow
    max_context_length = 16000  # Increased for OpenAI's better context handling
    context_text = "\n".join(context_blocks)
    if len(context_text) > max_context_length:
        # Truncate while keeping most relevant parts
        context_text = context_text[:max_context_length] + "\n[Content truncated for length]"

    prompt = SYSTEM_TEMPLATE.format(context=context_text, question=question)
    
    # Dynamic token allocation
    dynamic_max_tokens = get_dynamic_max_tokens(question, len(context_text))

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": dynamic_max_tokens,
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
            
            # Cache the response
            _response_cache[cache_key] = text
            return text or "Information not found in the document."

async def answer_with_openai_traceable(context_blocks: List[str], question: str) -> dict:
    """Enhanced version with traceability - main function for external use"""
    answer = await answer_with_openai(context_blocks, question)
    return format_answer_with_traceability(answer, context_blocks, question)

def format_answer_with_traceability(answer: str, source_chunks: List[str], question: str) -> dict:
    """Format answer with traceability information"""
    # Extract policy identifiers from source chunks
    policy_info = extract_policy_identifiers("\n".join(source_chunks))
    
    # Calculate confidence based on source relevance
    confidence_score = calculate_confidence_score(source_chunks, question)
    
    return {
        "answer": answer,
        "source_references": [
            {
                "chunk_id": f"chunk_{i}",
                "relevance_score": 0.85,  # Placeholder - can be enhanced later
                "policy_section": extract_policy_section(chunk)
            } for i, chunk in enumerate(source_chunks[:3])  # Top 3 most relevant
        ],
        "confidence_score": confidence_score,
        "policy_identifiers": policy_info
    }

def extract_policy_identifiers(text: str) -> dict:
    """Extract insurance policy identifiers from text"""
    import re
    
    return {
        "irdai_reg_no": re.findall(r"IRDAI Reg\.? No\.?[:\s]*([A-Z0-9]+)", text, re.IGNORECASE),
        "cin": re.findall(r"CIN[:\s]*([A-Z0-9]+)", text, re.IGNORECASE),
        "uin": re.findall(r"UIN[:\s]*([A-Z0-9]+)", text, re.IGNORECASE),
        "policy_name": re.findall(r"([A-Za-z\s]+Policy)", text, re.IGNORECASE)
    }

def extract_policy_section(chunk_text: str) -> str:
    """Extract policy section information from chunk text"""
    # Simple extraction - can be enhanced with more sophisticated parsing
    if "section" in chunk_text.lower():
        import re
        section_match = re.search(r"Section\s+([0-9\.]+)", chunk_text, re.IGNORECASE)
        if section_match:
            return f"Section {section_match.group(1)}"
    return "General Policy Information"

def calculate_confidence_score(source_chunks: List[str], question: str) -> float:
    """Calculate confidence score based on source relevance"""
    if not source_chunks:
        return 0.0
    
    # Simple confidence calculation - can be enhanced
    total_chunks = len(source_chunks)
    if total_chunks >= 5:
        return 0.95
    elif total_chunks >= 3:
        return 0.85
    elif total_chunks >= 2:
        return 0.75
    else:
        return 0.65
