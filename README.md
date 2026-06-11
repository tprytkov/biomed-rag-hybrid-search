# Biomedical Hybrid-RAG Search and Synthesis Engine

A local-first biomedical retrieval-augmented generation (RAG) portfolio project that
combines sparse keyword retrieval with dense transformer embeddings and linear score
fusion. Grounding answers in retrieved evidence helps reduce hallucination risk and
makes the retrieval process easier to inspect.

## Live Demo: Streamlit Public Frontend

Explore the [live Streamlit demo](https://biomed-rag-hybrid-search-hmnywnzkx68xjbjkxentmw.streamlit.app/).

The public demo runs in self-contained mode using synthetic, public-safe biomedical
data. It does not require FastAPI, Docker, an OpenAI API, secrets, or private data.

## Project Summary

This repository demonstrates:

- Hybrid biomedical retrieval using sparse TF-IDF scores and dense similarity scores.
- Equal-weight linear score fusion for ranked evidence retrieval.
- Evidence snippets and component scores for transparent result inspection.
- An evidence-grounded answer layer that avoids paid API requirements.
- A FastAPI endpoint for optional backend integration.
- A self-contained Streamlit frontend for local and public portfolio demonstrations.
- Automated tests for retrieval scoring and API behavior.
- Docker configuration for an optional containerized deployment path.

All bundled demonstration content is synthetic and public-safe. The project is not
intended for clinical decision-making.

## Architecture

```text
Synthetic biomedical text
        |
        v
Text chunking
        |
        +--> Sparse TF-IDF keyword retrieval
        |
        +--> Dense transformer embeddings
        |
        v
50/50 linear score fusion
        |
        v
Ranked evidence snippets
        |
        v
Evidence-grounded answer
```

The self-contained Streamlit mode uses deterministic local vector-style similarity so
it can run without downloading a Hugging Face model. The optional FastAPI backend uses
the configured `BAAI/bge-small-en-v1.5` embedding model.

## Local Setup

Python 3.11 with Conda is recommended.

```bash
git clone https://github.com/tprytkov/biomed-rag-hybrid-search.git
cd biomed-rag-hybrid-search

conda create -n biomed-rag python=3.11 -y
conda activate biomed-rag
pip install -r requirements.txt
```

## Run the Streamlit Frontend

Launch the self-contained portfolio demo from the repository root:

```bash
streamlit run frontend/streamlit_app.py
```

The default mode requires no backend service. To use the optional API integration,
start FastAPI and select **FastAPI backend mode** in the Streamlit sidebar.

## Run the FastAPI Backend

```bash
python -m uvicorn src.app.api.main:app --reload
```

Open:

- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health endpoint: [http://localhost:8000/](http://localhost:8000/)

The backend embedding model may download from Hugging Face on first use if it is not
already cached. This download is not required for the default Streamlit demo.

## Streamlit Community Cloud

Use these deployment settings:

| Setting | Value |
|---|---|
| Repository | `tprytkov/biomed-rag-hybrid-search` |
| Branch | `main` |
| Main file path | `frontend/streamlit_app.py` |

No application secrets are required for the self-contained public demo.

## Testing

Run the complete test suite from the activated Conda environment:

```bash
python -m pytest -q
```

## Optional Docker Deployment

Docker is not required for the Streamlit demo. To run the optional containerized
services:

```bash
docker-compose up --build
```

## Repository Layout

```text
frontend/                 Self-contained Streamlit portfolio frontend
src/app/api/              FastAPI application and legacy API client
src/app/retrieval/        Chunking, embedding, and hybrid retrieval logic
src/app/llm/              Evidence-grounded answer layer
src/app/evals/            Evaluation module placeholders
tests/                    Retrieval and API tests
docs/                     Architecture, model, and API documentation
scripts/                  Local ingestion and indexing examples
```

## Safety and Scope

This repository is a software engineering demonstration. It uses synthetic biomedical
examples and must not be treated as medical advice, clinical evidence, or a validated
decision-support system.
