import streamlit as tf
import requests

# 1. Page Configuration Metrics
tf.set_page_config(
    page_title="Biomedical Hybrid-RAG Portal",
    page_icon="🧬",
    layout="wide"
)

tf.title("🧬 Biomedical Hybrid-RAG Discovery Portal")
tf.markdown(
    "Query clinical literature datasets instantly using integrated sparse keyword indices "
    "combined with deep transformer semantic embedding vector matrices."
)

# 2. Sidebar Configuration Framework Controls
tf.sidebar.header("Pipeline Engineering Hub")
api_endpoint = tf.sidebar.text_input("Core API Gateway Gateway URL", value="http://localhost:8000/v1/query")
retrieval_depth = tf.sidebar.slider("Top K Matches Retrieval Depth", min_value=1, max_value=3, value=2)

tf.sidebar.markdown("---")
tf.sidebar.info(
    "**Backend Processing Specifications:**\n"
    "- Dense Embeddings: `bge-small-en-v1.5` (384-dim)\n"
    "- Sparse Retrieval: TF-IDF Matrix Tokenizer\n"
    "- Grounding Layer: `Llama-3-8B-Instruct`"
)

# 3. Main Operational Interface Layout Workspace
user_query = tf.text_input(
    "Enter clinical question or medical biomarker criteria:", 
    value="What target mutations match third-generation kinase inhibitors?"
)

if tf.button("Execute Hybrid Pipeline", type="primary"):
    if not user_query.strip():
        tf.warning("Please input a valid medical search query string before running computation loops.")
    else:
        with tf.spinner("Orchestrating sparse/dense matching and generating answers..."):
            payload = {
                "query": user_query,
                "top_k": retrieval_depth
            }
            
            try:
                # Dispatch POST network requests directly onto the active FastAPI backend layer
                response = requests.post(api_endpoint, json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Output Row A: Render Context-Grounded Answer
                    tf.subheader("💡 Grounded AI Clinical Synthesis")
                    tf.success(data["answer"])
                    
                    # Output Row B: Render Evidence Verification Metrics Charts
                    tf.subheader("🔬 Verifiable Evidence Snippets Retrieved")
                    
                    for index, item in enumerate(data["retrieved_context"], 1):
                        with tf.expander(f"Evidence Rank {index} | ID: {item['chunk_id']} (Combined Match Score: {item['score']:.4f})"):
                            tf.write(f"**Document Abstract Excerpt:** {item['text']}")
                            
                            # Layout metric split matrix values for debugging visual clarity
                            col1, col2 = tf.columns(2)
                            col1.metric("Keyword Match Score (Sparse)", f"{item['sparse_score']:.4f}")
                            col2.metric("Semantic Match Score (Dense)", f"{item['dense_score']:.4f}")
                else:
                    tf.error(f"Backend Server Exception Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                tf.error(
                    "Network Connection Failure: Unable to handshake with the backend core application logic. "
                    "Ensure your FastAPI server application layer is running actively on port 8000 before running the dashboard interface wrapper."
                )
