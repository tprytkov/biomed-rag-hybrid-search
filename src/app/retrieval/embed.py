 
from typing import List
from sentence_transformers import SentenceTransformer
from src.app.utils.config import settings
from src.app.utils.logging import logger

class MedicalEmbeddingEngine:
    def __init__(self):
        logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL_NAME}")
        # Automatically loads on GPU if available, otherwise drops back to CPU
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Converts a batch of text strings into an array of vector embeddings."""
        if not texts:
            return []
        
        logger.info(f"Generating dense vector embeddings for a batch of {len(texts)} chunks.")
        # Generate embeddings as lists of floats
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings
