\# Biomedical Hybrid-RAG System Architecture



This document provides a comprehensive technical overview of the system data processing, search retrieval logic, and endpoint orchestration layers.



\## 1. System Topology Overview

The architecture is structured as a decoupled, multi-stage pipeline designed to handle dense scientific medical literature without context degradation.



\[Raw Medical Documents] ──> \[Text Chunker Engine] ──> \[Embedding Models Layer]│▼\[User Query] ──> \[Hybrid Search Engine] ◄──────────── \[Vector Storage Matrix]│ (Sparse TF-IDF + Dense Cosine)▼\[LLM Grounding Prompt] ──> \[Context-Aware Answer]

\## 2. Core Functional Components



\### A. Document Ingestion \& Sliding Chunk Engine

\* \*\*File Reference\*\*: `src/app/retrieval/chunking.py`

\* \*\*Mechanism\*\*: Fixed-size sliding character window with an overlapping boundary buffer.

\* \*\*Specifications\*\*: Default window capacity set to `500` characters with a `50` character overlap to preserve semantic context across chunk fragments.



\### B. Dense Vector Embedding Engine

\* \*\*File Reference\*\*: `src/app/retrieval/embed.py`

\* \*\*Model Topology\*\*: `BAAI/bge-small-en-v1.5` transformer architecture.

\* \*\*Vector Dimension\*\*: 384 dimensions.

\* \*\*Execution Boundary\*\*: Runs natively on available GPU accelerators via PyTorch, falling back automatically to CPU threading.



\### C. Hybrid Search Fusion Layer

\* \*\*File Reference\*\*: `src/app/retrieval/hybrid\_search.py`

\* \*\*Sparse Index\*\*: Custom Term Frequency-Inverse Document Frequency (TF-IDF) extraction using `scikit-learn` to index distinct scientific jargon keywords.

\* \*\*Dense Index\*\*: Continuous numeric tracking powered by standard NumPy dot-product cosine metrics.

\* \*\*Score Fusion\*\*: Linear combination weighting:

&#x20; $$\\text{Score}\_{\\text{Hybrid}} = (0.5 \\times \\text{Score}\_{\\text{Sparse}}) + (0.5 \\times \\text{Score}\_{\\text{Dense}})$$



\### D. API Gateway Routing

\* \*\*File Reference\*\*: `src/app/api/main.py`

\* \*\*Framework\*\*: FastAPI core asynchronous runtime layout.

\* \*\*Default Network Boundaries\*\*: Internal hosting binded on port `8000`.



\## 3. Data Processing Sequences

1\. \*\*Scaffolding State\*\*: Input corpora is sliced, vectorized, and compiled into the unified sparse/dense lookup matrix.

2\. \*\*Query Processing\*\*: Inbound JSON POST payloads map to flat 1D arrays (`shape (384,)`).

3\. \*\*Retrieval\*\*: System matches precise terms (Sparse) and conceptual meaning (Dense) to return the `top\_k` matches.

4\. \*\*Generation\*\*: Top matches are stuffed into an isolation instruction prompt template for final generation.

