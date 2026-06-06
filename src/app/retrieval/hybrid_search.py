 
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from src.app.utils.logging import logger

class BiomedicalHybridSearcher:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.chunks = chunks
        self.embeddings = np.array(embeddings)
        self.corpus_texts = [c["text"] for c in chunks]
        
        logger.info("Initializing Sparse TF-IDF Index for keyword retrieval matching.")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.sparse_matrix = self.vectorizer.fit_transform(self.corpus_texts)

    def search(self, query_text: str, query_vector: List[float], top_k: int = 2) -> List[Dict[str, Any]]:
        """Executes a hybrid search merging keyword weights with dense vector similarity."""
        logger.info(f"Executing hybrid search for query: '{query_text}'")
        
        # 1. Compute Sparse Scores (Keyword Match) via Dot Product
        query_sparse = self.vectorizer.transform([query_text])
        sparse_scores = (self.sparse_matrix * query_sparse.T).toarray().flatten()

        # 2. Compute Dense Scores (Semantic Match) via Cosine Similarity
        q_vec = np.array(query_vector)
        norm_corpus = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(q_vec)
        
        # Guard against zero-division errors
        if norm_query == 0 or np.any(norm_corpus == 0):
            dense_scores = np.zeros(len(self.chunks))
        else:
            dense_scores = np.dot(self.embeddings, q_vec) / (norm_corpus * norm_query)

        # 3. Fuse Scores (50/50 linear combination weighting)
        hybrid_scores = (0.5 * sparse_scores) + (0.5 * dense_scores)

        # 4. Sort and return top matching chunks
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "text": self.chunks[idx]["text"],
                "score": float(hybrid_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "dense_score": float(dense_scores[idx])
            })
        return results
