import sys
import os

# Append project root directory to path to prevent absolute import failures
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.retrieval.chunking import MedicalTextChunker
from src.app.retrieval.embed import MedicalEmbeddingEngine
from src.app.utils.logging import logger

def run_pipeline():
    logger.info("--- Starting Biomedical RAG Ingestion Pipeline Run ---")
    
    # Mock data representing a medical abstract or corpus file
    sample_document_id = "PMID_3489102"
    sample_medical_text = (
        "The patient presented with elevated baseline biomarkers indicating potential "
        "resistance to standard kinase inhibitors. Genetic screening revealed a rare mutation "
        "in the EGFR tyrosine kinase domain. Combined hybrid therapeutic regimens utilizing third-generation "
        "inhibitors showed substantial reduction in tumor volume without significant toxicity markers."
    )
    
    # 1. Initialize components
    chunker = MedicalTextChunker(chunk_size=100, chunk_overlap=20)
    embedder = MedicalEmbeddingEngine()
    
    # 2. Slice text into chunks
    chunks = chunker.chunk_document(doc_id=sample_document_id, text=sample_medical_text)
    
    if not chunks:
        logger.error("Pipeline aborted: No text chunks extracted.")
        return
        
    # 3. Extract text strings for vector transformations
    text_strings = [c["text"] for c in chunks]
    
    # 4. Process dense embeddings vectors
    vectors = embedder.get_embeddings(text_strings)
    
    # 5. Link vectors back to metadata chunks
    for idx, chunk in enumerate(chunks):
        chunk["vector_preview"] = vectors[idx][:3]  # Log just the first 3 numbers to save terminal space
        logger.info(f"Chunk ID: {chunk['chunk_id']} | Vector Preview: {chunk['vector_preview']}...")

    logger.info(f"--- Pipeline complete! Processed {len(chunks)} embedded vector chunks. ---")

if __name__ == "__main__":
    run_pipeline()

