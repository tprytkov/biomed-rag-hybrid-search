\# Biomedical Hybrid-RAG Search \& Synthesis Engine



An asynchronous, production-grade Retrieval-Augmented Generation (RAG) platform specialized for dense clinical and scientific literature. This architecture combines sparse token match matrices (TF-IDF/BM25) with dense transformer embeddings (`bge-small-en-v1.5`) via linear score fusion to eliminate generative hallucinations and optimize search relevance.

## Project Summary

This project demonstrates a biomedical Retrieval-Augmented Generation pipeline using hybrid search. It combines sparse keyword-based retrieval with dense transformer embeddings, fuses retrieval scores, and serves results through a FastAPI backend and Streamlit interface. The repository also includes notebooks, automated tests, Docker configuration, and documentation for API usage, architecture, and model behavior.


\## 🧬 Core Architecture Topology

\* \*\*FastAPI Application Layer\*\*: High-performance, asynchronous REST API orchestration endpoints.

\* \*\*Streamlit UI Portal\*\*: Interactive visual dashboard for real-time query tracking and dual-metric score reporting.

\* \*\*Hybrid Search Engine\*\*: Dual-stream indexing utilizing `scikit-learn` matrix vectorizers and `SentenceTransformers` running on local CPU/GPU hardware.

\* \*\*Automated Test Matrix\*\*: Full endpoint validation and matrix shape verification powered by `pytest`.

\* \*\*Containerized Deployment\*\*: Multi-stage `Dockerfile` and multi-service `docker-compose.yml` blueprints.



\---



\## 🛠️ Local Installation \& Setup



\### Prerequisites

\* Windows 10/11 with \*\*Anaconda Prompt\*\* or \*\*Miniconda\*\* installed.

\* Git command line utility.



\### 1. Clone and Navigate to the Repository

```bash

git clone https://github.com/tprytkov/biomed-rag-hybrid-search.git

cd biomed-rag-hybrid-search

```



\### 2. Configure the Isolated Conda Sandbox Workspace

```bash

\# Create a dedicated environment with Python 3.11

conda create -n biomed-rag python=3.11 -y



\# Activate the sandbox environment

conda activate biomed-rag

```



\### 3. Install System Dependencies Matrix

```bash

pip install -r requirements.txt

```



\---



\## 🚀 Execution Guide



To run the complete framework locally, you must open two separate terminal windows with your `(biomed-rag)` environment activated.



\### Step A: Initialize the FastAPI Service Backend

In your first terminal window, run the application gateway host:

```bash

uvicorn src.app.api.main:app --reload

```

\* \*\*Interactive Swagger UI Endpoint Documentation\*\*: Open your browser and navigate to \[http://localhost:8000/docs](http://localhost:8000/docs) to explore and execute raw backend endpoints.



\### Step B: Launch the Interactive Streamlit Frontend Dashboard

In your second terminal window, run the user graphical user interface layer:

```bash

streamlit run src/app/api/frontend.py

```

\* The portal will open automatically at \[http://localhost:8501](http://localhost:8501).



\---



\## 🧪 Automated Testing Validation

Run the full automated testing suite to verify structural data mapping, matrix alignments, and endpoint health statuses:

```bash

pytest -v

```



\---



\## 🐳 Containerized Deployment (Docker)

To package and run the entire ecosystem (including an integrated Qdrant scalable vector database) inside an isolated container cluster, run:

```bash

docker-compose up --build

```



