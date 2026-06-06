import pytest
from fastapi.testclient import TestClient
from src.app.api.main import app

# Initialize the lightweight FastAPI execution validation client
client = TestClient(app)

def test_read_root_endpoint():
    """Validates that the base API healthcheck endpoint responds with a 200 code."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "biomed-rag-hybrid-search"}

def test_query_pipeline_endpoint_success():
    """Validates that valid POST request payloads return correct text answers and context lists."""
    payload = {
        "query": "What are the mutations targeting third-generation kinase inhibitors?",
        "top_k": 1
    }
    response = client.post("/v1/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Assert output keys exist matching Pydantic validation expectations
    assert "query" in data
    assert "answer" in data
    assert "retrieved_context" in data
    assert len(data["retrieved_context"]) == 1
    assert "Mock LLM Output" in data["answer"]

