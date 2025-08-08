import re
from typing import List, Tuple


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace and remove excessive newlines
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter based on punctuation; robust enough for policy docs
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def build_chunks(
    text: str,
    chunk_words: int,
    overlap_words: int,
) -> List[Tuple[str, int]]:
    """
    Returns list of (chunk_text, local_chunk_id)
    """
    sentences = split_into_sentences(text)
    chunks: List[Tuple[str, int]] = []

    current_words: List[str] = []
    current_len = 0
    chunk_id = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) <= chunk_words:
            current_words.extend(words)
            current_len += len(words)
        else:
            if current_words:
                chunks.append((" ".join(current_words), chunk_id))
                chunk_id += 1
            # overlap
            if overlap_words > 0 and current_words:
                overlap = current_words[-overlap_words:]
            else:
                overlap = []
            current_words = overlap + words
            current_len = len(current_words)

    if current_words:
        chunks.append((" ".join(current_words), chunk_id))

    return chunks
