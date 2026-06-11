from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.retrieval.hybrid_search import BiomedicalHybridSearcher


APP_TITLE = "Biomedical RAG Hybrid Search Demo"
APP_SUBTITLE = (
    "Local-first biomedical retrieval-augmented search using hybrid evidence retrieval "
    "and synthetic public-safe data."
)
DEMO_MODE = "Self-contained demo mode"
API_MODE = "FastAPI backend mode"

SYNTHETIC_CORPUS = [
    {
        "chunk_id": "SYN-EGFR-001",
        "title": "EGFR resistance overview",
        "topic": "Targeted therapy",
        "text": (
            "Synthetic summary: EGFR T790M is a resistance-associated alteration in "
            "non-small cell lung cancer. Third-generation EGFR tyrosine kinase inhibitors "
            "were designed to retain activity against this resistance mechanism."
        ),
    },
    {
        "chunk_id": "SYN-EGFR-002",
        "title": "EGFR exon 20 insertions",
        "topic": "Biomarkers",
        "text": (
            "Synthetic summary: EGFR exon 20 insertion variants form a distinct biomarker "
            "group and may respond differently from common sensitizing EGFR alterations."
        ),
    },
    {
        "chunk_id": "SYN-ALK-001",
        "title": "ALK-positive lung cancer",
        "topic": "Targeted therapy",
        "text": (
            "Synthetic summary: ALK rearrangements can define a molecular subgroup of "
            "non-small cell lung cancer. ALK inhibitors target signaling driven by the fusion."
        ),
    },
    {
        "chunk_id": "SYN-KRAS-001",
        "title": "KRAS G12C pathway",
        "topic": "Pathway biology",
        "text": (
            "Synthetic summary: KRAS G12C can activate downstream MAPK signaling. "
            "Targeted inhibitors bind the altered KRAS protein and reduce pathway activity."
        ),
    },
    {
        "chunk_id": "SYN-BRCA-001",
        "title": "BRCA and DNA repair",
        "topic": "Genomic medicine",
        "text": (
            "Synthetic summary: BRCA1 and BRCA2 participate in homologous recombination "
            "DNA repair. Loss of repair function can create sensitivity to PARP inhibition."
        ),
    },
    {
        "chunk_id": "SYN-IMMUNO-001",
        "title": "Checkpoint immunotherapy",
        "topic": "Immuno-oncology",
        "text": (
            "Synthetic summary: Immune checkpoint inhibitors target regulatory pathways "
            "such as PD-1 and PD-L1 to support antitumor immune activity in selected settings."
        ),
    },
]

SAMPLE_QUERIES = [
    "What mechanism is associated with EGFR T790M resistance?",
    "How do ALK inhibitors relate to lung cancer?",
    "Which pathway is linked to KRAS G12C?",
    "Why can BRCA loss create sensitivity to PARP inhibition?",
    "What do checkpoint inhibitors target?",
]

EVALUATION_CASES = [
    ("EGFR T790M resistance", "SYN-EGFR-001"),
    ("ALK rearrangement lung cancer", "SYN-ALK-001"),
    ("KRAS G12C MAPK pathway", "SYN-KRAS-001"),
    ("BRCA homologous recombination repair", "SYN-BRCA-001"),
    ("PD-1 checkpoint immunotherapy", "SYN-IMMUNO-001"),
]

st.set_page_config(page_title=APP_TITLE, page_icon="", layout="wide")

st.markdown(
    """
    <style>
        .block-container {max-width: 1380px; padding-top: 2rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {background: #f6f8fb;}
        .hero {
            padding: 1.35rem 1.55rem;
            border: 1px solid #d8e2ec;
            border-radius: 14px;
            background: linear-gradient(135deg, #f7fbff 0%, #eef7f5 100%);
            margin-bottom: 1.25rem;
        }
        .hero h1 {margin: 0 0 0.35rem; color: #14364a; font-size: 2.05rem;}
        .hero p {margin: 0; color: #486577; font-size: 1.02rem;}
        .answer-card {
            padding: 1rem 1.1rem;
            border-left: 4px solid #187b91;
            border-radius: 7px;
            background: #f2f8fa;
            color: #244b5a;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #dfe7ee;
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            background: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def build_demo_index() -> tuple[BiomedicalHybridSearcher, TfidfVectorizer, np.ndarray]:
    texts = [item["text"] for item in SYNTHETIC_CORPUS]
    semantic_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    document_vectors = normalize(semantic_vectorizer.fit_transform(texts)).toarray()
    chunks = [
        {
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "metadata": {"title": item["title"], "topic": item["topic"]},
        }
        for item in SYNTHETIC_CORPUS
    ]
    return (
        BiomedicalHybridSearcher(chunks, document_vectors.tolist()),
        semantic_vectorizer,
        document_vectors,
    )


def demo_search(query: str, top_k: int) -> list[dict[str, Any]]:
    searcher, semantic_vectorizer, _ = build_demo_index()
    query_vector = normalize(semantic_vectorizer.transform([query])).toarray()[0]
    results = searcher.search(query, query_vector.tolist(), top_k=top_k)
    metadata = {item["chunk_id"]: item for item in SYNTHETIC_CORPUS}
    for rank, result in enumerate(results, start=1):
        source = metadata[result["chunk_id"]]
        result.update(rank=rank, title=source["title"], topic=source["topic"])
    return results


def grounded_answer(query: str, results: list[dict[str, Any]]) -> str:
    if not results or results[0]["score"] <= 0:
        return "The synthetic corpus does not contain sufficient evidence to answer this query."
    lead = results[0]
    return (
        f"Based on the highest-ranked synthetic evidence [{lead['chunk_id']}], "
        f"{lead['text'].removeprefix('Synthetic summary: ').strip()}"
    )


def api_request(
    api_url: str, method: str, path: str, payload: dict[str, Any] | None = None
) -> tuple[Any | None, str | None]:
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.request(method, f"{api_url.rstrip('/')}{path}", json=payload)
            response.raise_for_status()
            return response.json(), None
    except httpx.HTTPError as exc:
        return None, str(exc)


def evaluate_demo() -> dict[str, Any]:
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    rows = []
    for query, expected_id in EVALUATION_CASES:
        results = demo_search(query, top_k=3)
        ids = [item["chunk_id"] for item in results]
        rank = ids.index(expected_id) + 1 if expected_id in ids else None
        reciprocal_ranks.append(1 / rank if rank else 0)
        hits_at_1 += int(rank == 1)
        hits_at_3 += int(rank is not None and rank <= 3)
        rows.append(
            {
                "query": query,
                "expected_evidence": expected_id,
                "top_result": ids[0],
                "expected_rank": rank or "not retrieved",
            }
        )
    count = len(EVALUATION_CASES)
    return {
        "hit_at_1": hits_at_1 / count,
        "hit_at_3": hits_at_3 / count,
        "mrr": sum(reciprocal_ranks) / count,
        "queries": count,
        "rows": rows,
    }


st.markdown(
    f"""
    <div class="hero">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo controls")
    mode = st.radio("Application mode", [DEMO_MODE, API_MODE])
    api_url = "http://127.0.0.1:8000"
    if mode == API_MODE:
        api_url = st.text_input("FastAPI URL", value=api_url)
        if st.button("Check API connection", width="stretch"):
            health, error = api_request(api_url, "GET", "/")
            if error:
                st.error("Backend unavailable. Start FastAPI or switch to demo mode.")
            else:
                st.success(f"Connected to {health.get('service', 'FastAPI backend')}")

    st.divider()
    st.markdown("**Self-contained mode**")
    st.caption("Synthetic data · Local TF-IDF · Local vectors · Deterministic answer")
    st.caption("No FastAPI · No Docker · No OpenAI · No paid APIs · No model download")
    st.divider()
    st.caption("Portfolio demonstration only. Not for clinical decision-making.")

if "latest_results" not in st.session_state:
    st.session_state.latest_results = demo_search(SAMPLE_QUERIES[0], top_k=4)
    st.session_state.latest_query = SAMPLE_QUERIES[0]
    st.session_state.latest_answer = grounded_answer(
        SAMPLE_QUERIES[0], st.session_state.latest_results
    )

summary_tab, search_tab, evidence_tab, evaluation_tab, technical_tab = st.tabs(
    [
        "Project summary",
        "Search demo",
        "Retrieved evidence",
        "Evaluation",
        "Technical details",
    ]
)

with summary_tab:
    st.header("Project summary")
    st.write(
        "This project demonstrates an end-to-end biomedical RAG retrieval workflow with "
        "auditable evidence and local-first defaults."
    )
    metrics = st.columns(4)
    metrics[0].metric("Synthetic evidence records", len(SYNTHETIC_CORPUS))
    metrics[1].metric("Retrieval channels", 2)
    metrics[2].metric("Fusion weighting", "50 / 50")
    metrics[3].metric("External API keys", 0)

    left, right = st.columns(2)
    with left:
        st.subheader("What it demonstrates")
        st.markdown(
            """
            - **RAG:** retrieval precedes a source-grounded answer.
            - **Hybrid retrieval:** keyword and vector-style semantic scores run together.
            - **Score fusion:** sparse and dense similarities receive equal weight.
            - **FastAPI:** an optional backend mode uses the existing `/v1/query` endpoint.
            - **Testing:** retrieval and API behavior are covered by pytest.
            - **Local-first design:** demo mode runs without infrastructure or credentials.
            """
        )
    with right:
        st.subheader("Synthetic corpus")
        st.dataframe(
            [
                {
                    "evidence_id": item["chunk_id"],
                    "topic": item["topic"],
                    "title": item["title"],
                }
                for item in SYNTHETIC_CORPUS
            ],
            width="stretch",
            hide_index=True,
        )

with search_tab:
    st.header("Search demo")
    st.write(
        "Compare lexical matching, vector-style semantic similarity, and their fused ranking."
    )
    with st.form("search_form"):
        selected_query = st.selectbox("Example biomedical question", SAMPLE_QUERIES)
        custom_query = st.text_input(
            "Or enter a query",
            placeholder="Example: What pathway is activated by KRAS G12C?",
        )
        top_k = st.slider("Evidence snippets to retrieve", 1, len(SYNTHETIC_CORPUS), 4)
        submitted = st.form_submit_button("Run hybrid search", type="primary")

    if submitted:
        query = custom_query.strip() or selected_query
        if mode == DEMO_MODE:
            results = demo_search(query, top_k)
            answer = grounded_answer(query, results)
            error = None
        else:
            response, error = api_request(
                api_url, "POST", "/v1/query", {"query": query, "top_k": top_k}
            )
            results = response.get("retrieved_context", []) if response else []
            answer = response.get("answer", "") if response else ""

        if error:
            st.error(f"Could not query the FastAPI backend: {error}")
        else:
            st.session_state.latest_query = query
            st.session_state.latest_results = results
            st.session_state.latest_answer = answer

    st.subheader("Evidence-grounded answer")
    st.markdown(
        f'<div class="answer-card">{st.session_state.latest_answer}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Query: {st.session_state.latest_query}")

    if st.session_state.latest_results:
        lead = st.session_state.latest_results[0]
        score_columns = st.columns(3)
        score_columns[0].metric("Top hybrid score", f"{lead['score']:.3f}")
        score_columns[1].metric("Keyword score", f"{lead['sparse_score']:.3f}")
        score_columns[2].metric("Semantic score", f"{lead['dense_score']:.3f}")

with evidence_tab:
    st.header("Retrieved evidence")
    st.write("Inspect ranked snippets and the component scores behind the hybrid result.")
    results = st.session_state.latest_results
    if not results:
        st.info("Run a search to retrieve evidence.")
    else:
        st.dataframe(
            [
                {
                    "rank": item.get("rank", index),
                    "evidence_id": item["chunk_id"],
                    "topic": item.get("topic", "API result"),
                    "hybrid_score": item["score"],
                    "keyword_score": item["sparse_score"],
                    "semantic_score": item["dense_score"],
                }
                for index, item in enumerate(results, start=1)
            ],
            width="stretch",
            hide_index=True,
            column_config={
                "hybrid_score": st.column_config.NumberColumn(format="%.4f"),
                "keyword_score": st.column_config.NumberColumn(format="%.4f"),
                "semantic_score": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        for index, item in enumerate(results, start=1):
            label = item.get("title", item["chunk_id"])
            with st.expander(f"Rank {index}: {label}", expanded=index == 1):
                st.write(item["text"])
                st.caption(f"Evidence ID: {item['chunk_id']}")

with evaluation_tab:
    st.header("Evaluation")
    st.write(
        "A deterministic smoke benchmark checks whether the expected synthetic evidence "
        "appears near the top of the hybrid ranking."
    )
    if mode == DEMO_MODE:
        report = evaluate_demo()
        evaluation_metrics = st.columns(4)
        evaluation_metrics[0].metric("Hit@1", f"{report['hit_at_1']:.0%}")
        evaluation_metrics[1].metric("Hit@3", f"{report['hit_at_3']:.0%}")
        evaluation_metrics[2].metric("Mean reciprocal rank", f"{report['mrr']:.3f}")
        evaluation_metrics[3].metric("Evaluation queries", report["queries"])
        st.dataframe(report["rows"], width="stretch", hide_index=True)
        st.caption(
            "This is a transparent synthetic retrieval benchmark, not a clinical validation study."
        )
    else:
        st.info(
            "The current FastAPI backend does not expose an evaluation endpoint. "
            "Switch to self-contained mode to view the local synthetic benchmark."
        )

with technical_tab:
    st.header("Technical details")
    st.write("The interface keeps retrieval mechanics visible and reproducible.")
    architecture, modes = st.columns(2)
    with architecture:
        st.subheader("Retrieval flow")
        st.code(
            """Biomedical query
    -> sparse TF-IDF keyword score
    -> local character n-gram vector score
    -> 50/50 linear score fusion
    -> ranked evidence snippets
    -> deterministic grounded answer""",
            language="text",
        )
    with modes:
        st.subheader("Operating modes")
        st.markdown(
            """
            | Mode | Retrieval runtime | Requirements |
            |---|---|---|
            | Self-contained demo | In-process hybrid search | Installed Python dependencies |
            | FastAPI backend | Existing `/v1/query` API | FastAPI service on port 8000 |
            """
        )

    st.subheader("Design choices")
    st.markdown(
        """
        - The existing `BiomedicalHybridSearcher` performs sparse scoring, dense cosine similarity, and fusion.
        - Demo vectors use deterministic local character n-grams to avoid downloading an embedding model.
        - Every answer is paired with ranked source snippets and component retrieval scores.
        - All content is synthetic and public-safe; no patient, proprietary, or unpublished research data is used.
        - No OpenAI dependency, secrets, paid APIs, Docker service, or remote vector database is required.
        """
    )
    st.warning(
        "This application demonstrates software architecture and retrieval behavior. "
        "It is not intended for medical advice or clinical use."
    )
