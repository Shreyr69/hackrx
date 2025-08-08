from __future__ import annotations
import asyncio
import io
from typing import Tuple

import httpx

from ..config import HTTP_TIMEOUT_SECS
from ..utils.chunking import clean_text


async def download_blob(url: str) -> Tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECS) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        return resp.content, content_type


def detect_type(url: str, content_type: str) -> str:
    if url.lower().endswith(".pdf"):
        return "pdf"
    if url.lower().endswith(".docx"):
        return "docx"
    if url.lower().endswith(".eml") or url.lower().endswith(".msg"):
        return "email"

    if "pdf" in content_type:
        return "pdf"
    if "word" in content_type or "docx" in content_type:
        return "docx"
    if "message" in content_type or "rfc822" in content_type:
        return "email"
    return "pdf"


def parse_pdf(data: bytes) -> str:
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        texts = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                texts.append(page_text)
    return clean_text("\n\n".join(texts))


def parse_docx(data: bytes) -> str:
    from docx import Document

    fh = io.BytesIO(data)
    doc = Document(fh)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return clean_text("\n\n".join(paragraphs))


def parse_email(data: bytes) -> str:
    from email import message_from_bytes

    msg = message_from_bytes(data)
    texts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                try:
                    texts.append(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="ignore"))
                except Exception:
                    continue
    else:
        if msg.get_content_type() == "text/plain":
            texts.append(msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", errors="ignore"))
    return clean_text("\n\n".join(texts))


async def ingest_document(url: str) -> str:
    data, content_type = await download_blob(url)
    kind = detect_type(url, content_type)

    if kind == "pdf":
        return await asyncio.to_thread(parse_pdf, data)
    if kind == "docx":
        return await asyncio.to_thread(parse_docx, data)
    if kind == "email":
        return await asyncio.to_thread(parse_email, data)

    try:
        return await asyncio.to_thread(parse_pdf, data)
    except Exception:
        try:
            text = data.decode("utf-8", errors="ignore")
            return clean_text(text)
        except Exception:
            return ""
