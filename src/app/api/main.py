 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.app.retrieval.chunking import MedicalTextChunker
from src.app.retrieval.embed import MedicalEmbeddingEngine
from src.app.retrieval.hybrid_search import BiomedicalHybridSearcher
from src.app.llm.answer import MedicalAnswerGenerator
from src.app.utils.logging import logger

app = FastAPI(
    title="Biomedical Hybrid-RAG API Core",
    version="1.0.0",
    description="FastAPI service for search and generation across dense and sparse indexes."
)

# Define clean Pydantic tracking models for input validation
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 2

class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieved_context: List[dict]

# Temporary in-memory document state to power the server instance demo
MOCK_CORPUS = (
    "EGFR mutations are highly prevalent in lung cancer variants. "
    "Third-generation kinase inhibitors target tyrosine kinase resistance pathways directly. "
    "Clinical evaluation shows reduction in toxic side effects compared to chemotherapy options."
)

# Instantiate pipelines on application startup
chunker = MedicalTextChunker(chunk_size=100, chunk_overlap=20)
embedder = MedicalEmbeddingEngine()
generator = MedicalAnswerGenerator()

# Pre-compile the index framework for immediate use
chunks = chunker.chunk_document(doc_id="DOC_START_01", text=MOCK_CORPUS)
texts = [c["text"] for c in chunks]
vectors = embedder.get_embeddings(texts)
searcher = BiomedicalHybridSearcher(chunks, vectors)

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "biomed-rag-hybrid-search"}

@app.post("/v1/query", response_model=RAGResponse)
def query_pipeline(payload: QueryRequest):
    try:
        # 1. Transform query text to raw spatial embeddings vectors
        query_vec = embedder.get_embeddings([payload.query])
        
        # 2. Extract best matching snippets through sparse and dense criteria
        contexts = searcher.search(payload.query, query_vector=query_vec, top_k=payload.top_k)
        
        if not contexts:
            raise HTTPException(status_code=404, detail="No relevant context matches found in matrix data.")
            
        # 3. Ground answer through prompt templates
        answer = generator.generate_answer(payload.query, contexts)
        
        return RAGResponse(
            query=payload.query,
            answer=answer,
            retrieved_context=contexts
        )
    except Exception as e:
        logger.exception("Failure event hit during pipeline orchestration execution.")
        raise HTTPException(status_code=500, detail=str(e))
