import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# ------------------------------
#  PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="GIS RAG Search Engine",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align: center; color:#00c2ff;'>🌍 GIS Knowledge RAG Search Engine</h1>
    <p style='text-align:center; font-size:18px;'>Ask any GIS question. Answers come from your scraped documents.</p>
""", unsafe_allow_html=True)

# ------------------------------
#  LOAD EMBEDDING MODEL
# ------------------------------
@st.cache_resource
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MODEL = get_model()

# ------------------------------
#  LOAD VECTOR INDEX + DOCUMENTS
# ------------------------------
def load_index():
    try:
        st.session_state.index = faiss.read_index("vector.index")
        st.session_state.documents = pickle.load(open("documents.pkl", "rb"))
        st.session_state.loaded = True
        st.success("Vector Index Loaded Successfully!")
    except Exception as e:
        st.error(f"Failed to load vector index: {e}")

# Load button
if st.button("Load Vector Index"):
    load_index()

st.write("---")

# ------------------------------
#  SEARCH FUNCTION
# ------------------------------
def search_query(query, top_k=3):
    if "loaded" not in st.session_state or not st.session_state.loaded:
        st.error("Please click 'Load Vector Index' first.")
        return []

    index = st.session_state.index
    docs = st.session_state.documents

    q_emb = MODEL.encode([query], convert_to_numpy=True)

    distances, indices = index.search(q_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(docs):
            results.append((docs[idx], dist))
    return results

# ------------------------------
#  USER INPUT
# ------------------------------
query = st.text_input("Enter your GIS question:", "")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        results = search_query(query)
        
        if results:
            st.write("### 🔎 Top Results:")
            for i, (text, score) in enumerate(results, start=1):
                st.markdown(f"#### Result {i} — Score: {score:.4f}")
                st.write(text)
                st.write("---")
        else:
            st.error("No matching documents found.")

# Footer
st.markdown("""
    <p style='text-align:center; font-size:14px; color:gray;'>
    GIS Knowledge RAG Engine — Powered by FAISS + MiniLM Embeddings
    </p>
""", unsafe_allow_html=True)
