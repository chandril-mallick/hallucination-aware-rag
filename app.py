import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Hallucination-Aware RAG", layout="wide")

st.title("Hallucination-Aware RAG System")
st.markdown("""
This system retrieves information from documents, generates answers, and **detects hallucinations** using multiple verification techniques.
""")

# Sidebar for controls and upload
with st.sidebar:
    st.header("1. Data Ingestion")
    uploaded_files = st.file_uploader("Upload PDF/TXT Documents", accept_multiple_files=True, type=["pdf", "txt"])
    
    if st.button("Ingest Documents"):
        if uploaded_files:
            files = [("files", (file.name, file, file.type)) for file in uploaded_files]
            with st.spinner("Ingesting and Indexing..."):
                try:
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
        else:
            st.warning("Please upload files first.")

    st.markdown("---")
    st.markdown("### Metrics Legend")
    st.info("**Faithfulness**: How well the answer follows the context.")
    st.info("**Relevance**: How relevant the answer is to the question.")
    st.info("**Attr. Score**: % of sentences with valid citations.")

# Main Chat Interface
st.header("2. Ask a Question")
query = st.text_input("Enter your query about the documents:")

if st.button("Generate Answer"):
    if query:
        with st.spinner("Retrieving, Generating, and Verifying..."):
            try:
                payload = {"text": query}
                response = requests.post(f"{API_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    context = data["context"]
                    hallucination = data["hallucination_analysis"]
                    metrics = data["metrics"]
                    
                    # Display Answer
                    st.subheader("Generated Answer")
                    st.write(answer)
                    
                    # Hallucination Analysis Display
                    st.subheader("Hallucination Detection")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Attribution Check")
                        attr_score = hallucination['attribution_check']['score_attribution']
                        st.progress(attr_score, text=f"Attribution Score: {attr_score:.2f}")
                        
                        # Show flagged sentences
                        results = hallucination['attribution_check']['results']
                        for res in results:
                            if res['status'] == "unsupported":
                                st.markdown(f" **Unsupported**: *{res['sentence']}* ({res['detail']})")
                            else:
                                st.markdown(f" **Supported**: *{res['sentence']}*")

                    with col2:
                        st.markdown("#### Semantic Similarity Check")
                        sim_score = hallucination['semantic_check']['score_similarity']
                        st.progress(sim_score, text=f"Similarity Score: {sim_score:.2f}")
                        
                        details = hallucination['semantic_check']['details']
                        for det in details:
                            color = "green" if det['status'] == "supported" else "red"
                            st.markdown(f":{color}[{det['status'].capitalize()}] ({det['max_similarity']:.2f}): *{det['sentence']}*")

                    # Metrics Panel
                    st.markdown("---")
                    st.subheader(" Evaluation Metrics")
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.2f}")
                    m_col2.metric("Relevance", f"{metrics.get('answer_relevance', 0):.2f}")
                    m_col3.metric("Context Precision", f"{metrics.get('context_precision', 0):.2f}")
                    m_col4.metric("Completeness", f"{metrics.get('answer_completeness', 0):.2f}")

                    # Context Accordion
                    with st.expander(" View Retrieved Context"):
                        for i, doc in enumerate(context):
                            st.markdown(f"**Source {i}** ({doc['source']}, Page {doc['page']})")
                            st.text(doc['content'])
                            st.divider()
                            
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
    else:
        st.warning("Please enter a query.")
