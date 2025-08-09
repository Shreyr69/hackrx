from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

from ..config import CHUNK_SIMILARITY_THRESHOLD


@dataclass
class Chunk:
    id: int
    text: str


class Retriever:
    def __init__(self, embeddings: np.ndarray, chunks: List[Chunk]):
        self.chunks = chunks
        self.embeddings = self._normalize(embeddings.astype(np.float32))
        self.index = None
        if _HAS_FAISS:
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.embeddings)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        q = query_vector.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        
        if self.index is not None:
            scores, indices = self.index.search(q.reshape(1, -1), min(top_k * 2, len(self.chunks)))
            result: List[Tuple[Chunk, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunks) and score >= CHUNK_SIMILARITY_THRESHOLD:
                    result.append((self.chunks[idx], float(score)))
            # Re-rank by score and return top_k
            result.sort(key=lambda x: x[1], reverse=True)
            return result[:top_k]
        
        # Fallback to numpy implementation
        sims = self.embeddings @ q
        # Filter by similarity threshold
        valid_indices = np.where(sims >= CHUNK_SIMILARITY_THRESHOLD)[0]
        if len(valid_indices) == 0:
            # If no chunks meet threshold, return top_k anyway
            top_idx = np.argsort(-sims)[:top_k]
            return [(self.chunks[i], float(sims[i])) for i in top_idx]
        
        # Get top_k from valid chunks
        valid_sims = sims[valid_indices]
        top_valid_idx = np.argsort(-valid_sims)[:top_k]
        return [(self.chunks[valid_indices[i]], float(valid_sims[i])) for i in top_valid_idx]
