 
import pytest
from src.app.retrieval.hybrid_search import BiomedicalHybridSearcher

def test_hybrid_search_scoring_and_sorting():
    # 1. Arrange a mock text collection and dummy 3-dimensional embeddings
    mock_chunks = [
        {"chunk_id": "c1", "text": "The patient has a rare genetic mutation in the EGFR tyrosine kinase domain."},
        {"chunk_id": "c2", "text": "Standard chemotherapy protocols were administered weekly."},
    ]
    # Representing mock 3-dim vectors: c1 has strong correlation, c2 has alternative alignment
    mock_embeddings = [
        [0.9, 0.1, 0.0],
        [0.1, 0.2, 0.8]
    ]
    
    searcher = BiomedicalHybridSearcher(mock_chunks, mock_embeddings)
    
    # 2. Act: Execute a search matching terms heavily correlated with chunk 1
    query_text = "EGFR mutation"
    query_vector = [0.95, 0.05, 0.0]
    
    results = searcher.search(query_text=query_text, query_vector=query_vector, top_k=2)
    
    # 3. Assert: Verify layout structures and sorting integrity
    assert len(results) == 2, "Search should return exactly two results."
    assert results[0]["chunk_id"] == "c1", "Chunk 'c1' should be ranked first due to matching keyword and vectors."
    assert results[0]["score"] > results[1]["score"], "First rank score must be strictly greater than second rank score."
    assert isinstance(results[0]["score"], float), "Calculated matrix metrics must be valid floating point types."
