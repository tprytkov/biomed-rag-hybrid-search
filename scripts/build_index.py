import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.retrieval.chunking import MedicalTextChunker
from src.app.retrieval.embed import MedicalEmbeddingEngine
from src.app.retrieval.hybrid_search import BiomedicalHybridSearcher
from src.app.utils.logging import logger

def run_search_demo():
    logger.info("--- Initializing Search Framework Demo ---")
    
    # 1. Process sample biomedical abstracts text
    raw_document = (
        "EGFR mutations are highly prevalent in lung cancer variants. "
        "Third-generation kinase inhibitors target tyrosine kinase resistance pathways directly. "
        "Clinical evaluation shows reduction in toxic side effects compared to chemotherapy options."
    )
    
    chunker = MedicalTextChunker(chunk_size=80, chunk_overlap=15)
    embedder = MedicalEmbeddingEngine()
    
    chunks = chunker.chunk_document(doc_id="DOC_001", text=raw_document)
    texts = [c["text"] for c in chunks]
    vectors = embedder.get_embeddings(texts)
    
    # 2. Spin up Search Engine Class
    searcher = BiomedicalHybridSearcher(chunks, vectors)
    
    # 3. Define target query and process its embedding vector
    user_query = "kinase inhibitor resistance mutation"
    query_vector = embedder.get_embeddings([user_query])[0]
    
    # 4. Perform Search Retrieval
    matched_results = searcher.search(user_query, query_vector, top_k=2)
    
    for rank, res in enumerate(matched_results, 1):
        logger.info(f"Rank {rank} | ID: {res['chunk_id']} | Final Combined Score: {res['score']:.4f}")
        logger.info(f"  [Text]: {res['text']}\n")

if __name__ == "__main__":
    run_search_demo()

