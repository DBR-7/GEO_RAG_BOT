import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------- Load Model and Data -----------
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SCRAPED_DIR = "data/scraped"

DOCUMENTS = []
EMBEDDINGS = None
INDEX = None

def load_documents():
    global DOCUMENTS, EMBEDDINGS, INDEX

    texts = []

    for file in os.listdir(SCRAPED_DIR):
        if file.endswith(".txt"):
            path = os.path.join(SCRAPED_DIR, file)
            if os.path.getsize(path) < 100:
                continue
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    # Chunk into paragraphs
    DOCUMENTS = []
    for text in texts:
        for chunk in text.split("\n\n"):
            if len(chunk.strip()) > 60:
                DOCUMENTS.append(chunk.strip())

    # Encode
    EMBEDDINGS = MODEL.encode(DOCUMENTS, convert_to_numpy=True)

    dim = EMBEDDINGS.shape[1]
    INDEX = faiss.IndexFlatL2(dim)
    INDEX.add(EMBEDDINGS)

# ----------- Query Function -----------
def retrieve(query, top_k=3):
    q_embed = MODEL.encode([query], convert_to_numpy=True)
    D, I = INDEX.search(q_embed, top_k)
    results = [(DOCUMENTS[idx], D[0][n]) for n, idx in enumerate(I[0])]
    return results

# ----------- Streamlit UI -----------
st.set_page_config(page_title="GIS RAG Search", layout="wide")
st.title("🌍 GIS Knowledge RAG Search Engine")

st.write("Ask any GIS question. Results come from your scraped documents.")

if st.button("Load Vector Index"):
    load_documents()
    st.success("Documents & FAISS Index Loaded!")

query = st.text_input("Enter your GIS question:")

if st.button("Search"):
    if INDEX is None:
        st.error("Please click 'Load Vector Index' first.")
    else:
        st.write("---")
        results = retrieve(query)
        st.subheader("🔎 Results")
        for text, score in results:
            st.write(f"**Score:** {score}")
            st.write(text)
            st.write("---")
