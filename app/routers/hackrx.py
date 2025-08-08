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
from ..services.llm import answer_with_gemini

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

    full_text = await ingest_document(payload.documents)
    if not full_text:
        raise HTTPException(status_code=400, detail="Failed to parse document")

    chunk_tuples = build_chunks(full_text, DEFAULT_CHUNK_WORDS, DEFAULT_CHUNK_OVERLAP_WORDS)
    if not chunk_tuples:
        raise HTTPException(status_code=400, detail="No content after parsing")

    chunks: List[Chunk] = [Chunk(id=cid, text=ct) for (ct, cid) in chunk_tuples]
    chunk_texts: List[str] = [c.text for c in chunks]

    chunk_embeddings = await embed_texts(chunk_texts)

    retriever = Retriever(chunk_embeddings, chunks)

    async def answer_one(q: str) -> str:
        q_vec = await embed_query(q)
        top = retriever.search(q_vec, TOP_K)
        ctx_blocks = [f"[Chunk {c.id}] {c.text}" for (c, _score) in top]
        ans = await answer_with_gemini(ctx_blocks, q)
        return ans.strip() or ""

    answers = await asyncio.gather(*[answer_one(q) for q in payload.questions])

    return RunResponse(answers=list(answers))
