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
            scores, indices = self.index.search(q.reshape(1, -1), top_k)
            result: List[Tuple[Chunk, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunks):
                    result.append((self.chunks[idx], float(score)))
            return result
        # numpy fallback (inner product)
        sims = self.embeddings @ q
        top_idx = np.argsort(-sims)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]
