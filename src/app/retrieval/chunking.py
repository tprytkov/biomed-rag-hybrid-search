 from typing import List, Dict, Any
from src.app.utils.logging import logger

class MedicalTextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Splits a single medical document text into overlapping chunks."""
        if not text or len(text.strip()) == 0:
            logger.warning(f"Skipping empty text block for doc_id: {doc_id}")
            return []

        chunks = []
        start = 0
        text_len = len(text)
        chunk_index = 0

        while start < text_len:
            # Calculate end boundary
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Pack chunk payload with cross-referencing metadata
            chunks.append({
                "chunk_id": f"{doc_id}_chk_{chunk_index}",
                "parent_id": doc_id,
                "text": chunk_text.strip(),
                "metadata": metadata or {}
            })
            
            # Advance sliding window by chunk step
            start += (self.chunk_size - self.chunk_overlap)
            chunk_index += 1

        logger.info(f"Successfully processed doc_id {doc_id} into {len(chunks)} text chunks.")
        return chunks

