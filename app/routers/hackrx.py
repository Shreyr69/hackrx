from __future__ import annotations
import asyncio
from typing import List

from fastapi import APIRouter, Header, HTTPException

from ..config import (
    REQUIRED_BEARER_TOKEN,
    DEFAULT_CHUNK_WORDS,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    TOP_K,
)
from ..models.schemas import RunRequest, RunResponse
from ..services.document_ingestion import ingest_document
from ..utils.chunking import build_chunks
from ..services.embeddings import embed_texts, embed_query
from ..services.retrieval import Retriever, Chunk
from ..services.llm import answer_with_openai, answer_with_openai_traceable

router = APIRouter()


@router.post("/hackrx/run", response_model=RunResponse)
async def run_endpoint(
    payload: RunRequest,
    authorization: str = Header(default=""),
):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if authorization.split(" ", 1)[1] != REQUIRED_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # OPTIMIZED: Process document ingestion and chunking
    full_text = await ingest_document(payload.documents)
    if not full_text:
        raise HTTPException(status_code=400, detail="Failed to parse document")

    chunk_tuples = build_chunks(full_text, DEFAULT_CHUNK_WORDS, DEFAULT_CHUNK_OVERLAP_WORDS)
    if not chunk_tuples:
        raise HTTPException(status_code=400, detail="No content after parsing")

    chunks: List[Chunk] = [Chunk(id=cid, text=ct) for (ct, cid) in chunk_tuples]
    chunk_texts: List[str] = [c.text for c in chunks]

    # OPTIMIZED: Generate embeddings for all chunks at once
    chunk_embeddings = await embed_texts(chunk_texts)
    retriever = Retriever(chunk_embeddings, chunks)

    # ENHANCED: Process all questions with improved retrieval and traceability
    async def process_question(q: str) -> str:
        q_vec = await embed_query(q)
        top_chunks = retriever.search(q_vec, TOP_K)
        
        # IMPROVED: Better context formatting with chunk relevance scores
        if not top_chunks:
            return "Information not found in the document."
        
        # Filter out low-quality chunks and format context
        relevant_chunks = []
        chunk_texts_for_trace = []
        for chunk, score in top_chunks:
            if score > 0.25:  # Using optimized threshold from config
                relevant_chunks.append(f"[Chunk {chunk.id} - Score: {score:.2f}] {chunk.text}")
                chunk_texts_for_trace.append(chunk.text)
        
        if not relevant_chunks:
            return "Information not found in the document."
        
        # Use traceable response for enhanced information
        traceable_response = await answer_with_openai_traceable(chunk_texts_for_trace, q)
        return traceable_response["answer"]  # Keep backward compatibility

    # OPTIMIZED: Process questions concurrently with controlled parallelism
    semaphore = asyncio.Semaphore(3)  # Limit concurrent LLM calls
    
    async def process_with_semaphore(q: str) -> str:
        async with semaphore:
            return await process_question(q)
    
    # Process all questions concurrently
    answers = await asyncio.gather(*[process_with_semaphore(q) for q in payload.questions])

    return RunResponse(answers=list(answers))
