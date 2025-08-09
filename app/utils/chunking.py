import re
from typing import List, Tuple


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    # IMPROVED: Better sentence splitting that preserves context
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs for better context preservation"""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def build_chunks(text: str, chunk_words: int, overlap_words: int) -> List[Tuple[str, int]]:
    # IMPROVED: Hybrid approach - use paragraphs when possible, sentences when needed
    paragraphs = split_into_paragraphs(text)
    chunks: List[Tuple[str, int]] = []
    chunk_id = 0

    for paragraph in paragraphs:
        # If paragraph is small enough, keep it as one chunk
        if len(paragraph.split()) <= chunk_words:
            chunks.append((paragraph, chunk_id))
            chunk_id += 1
        else:
            # Split large paragraphs into sentences
            sentences = split_into_sentences(paragraph)
            current_words: List[str] = []
            current_len = 0

            for sent in sentences:
                words = sent.split()
                if current_len + len(words) <= chunk_words:
                    current_words.extend(words)
                    current_len += len(words)
                else:
                    if current_words:
                        chunks.append((" ".join(current_words), chunk_id))
                        chunk_id += 1
                    
                    # IMPROVED: Better overlap handling
                    if overlap_words > 0 and current_words:
                        overlap = current_words[-overlap_words:] if len(current_words) >= overlap_words else current_words
                        current_words = overlap + words
                        current_len = len(current_words)
                    else:
                        current_words = words
                        current_len = len(words)

            if current_words:
                chunks.append((" ".join(current_words), chunk_id))
                chunk_id += 1

    # IMPROVED: Filter out very short chunks that might not be meaningful
    filtered_chunks = []
    for chunk_text, cid in chunks:
        if len(chunk_text.split()) >= 10:  # Minimum 10 words for meaningful chunks
            filtered_chunks.append((chunk_text, cid))
    
    return filtered_chunks
