\# Model Card: Biomedical Hybrid Retrieval \& Embedding Layer



This specification sheet documents the technical properties, performance trade-offs, and operating boundaries of the models used in this platform.



\## 1. Dense Embedding Model Details

\* \*\*Model Architecture\*\*: BAAI/bge-small-en-v1.5 (Bidirectional Encoder Representations from Transformers / BERT base variant)

\* \*\*Parameters\*\*: 33.5 Million

\* \*\*Vector Output Dimension\*\*: 384 dimensions

\* \*\*Sequence Length Context Capacity\*\*: 512 tokens

\* \*\*Primary Developer\*\*: Beijing Academy of Artificial Intelligence (BAAI)

\* \*\*License\*\*: MIT License



\### Intended Use Cases

\* High-throughput retrieval matching across complex scientific and medical texts.

\* Semantic similarity scanning for clinical decision support architectures.

\* Cross-lingual textual alignment tasks involving technical jargon.



\### Performance Limitations

\* Effectiveness decreases for input strings exceeding the 512-token truncation limit.

\* May struggle with non-English localized medical texts unless explicitly mapped via fine-tuning.



\---



\## 2. Sparse Token Retrieval Model Details

\* \*\*Model Engine\*\*: Custom Term Frequency-Inverse Document Frequency (TF-IDF) Matrix Vectorizer

\* \*\*Library Implementation\*\*: `scikit-learn`

\* \*\*Stop-Word Configuration\*\*: Native English standard stop-words filter activated (`stop\_words="english"`)

\* \*\*Vocabulary Matrix Strategy\*\*: Dynamic vocabulary extraction mapped on-the-fly during data corpus ingestion execution phases.



\### Intended Use Cases

\* Exact keyword overlapping lookups for specific scientific entities (e.g., compound acronyms like `EGFR`, drug serial numbers, or genomic mutation codes).



\---



\## 3. Generative Language Model Reference Blueprint

\* \*\*Target Model Layout\*\*: `meta-llama/Llama-3-8B-Instruct`

\* \*\*Parameters\*\*: 8 Billion

\* \*\*Context Window Capacity\*\*: 8,192 tokens

\* \*\*Primary Developer\*\*: Meta AI

\* \*\*Operating Paradigm\*\*: Multi-turn instruct fine-tuned conversational weights.



\### Context-Grounding Strategy

To eliminate hallucination risks typical of generative systems, this model is configured under a strict RAG grounding system:

\* \*\*Temperature Setting\*\*: 0.0 (Deterministic output constraint).

\* \*\*Grounding Instruction Rule\*\*: The model must immediately declare insufficient information if a query cannot be completely answered by the exact text chunks provided by the hybrid search layer.



