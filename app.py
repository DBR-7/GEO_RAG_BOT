import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import os

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(page_title="GIS RAG Chat Assistant", page_icon="🌍", layout="wide")

st.markdown("""
<h1 style='text-align:center;color:#00c2ff;'>🌍 GIS RAG Chat Assistant</h1>
<p style='text-align:center;font-size:18px;'>Chat with your GIS knowledge base powered by FAISS + Groq LLM.</p>
""", unsafe_allow_html=True)

# --------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

MODEL = load_model()

# --------------------------------------
# LOAD GROQ API
# --------------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ Missing GROQ_API_KEY in Streamlit Secrets or environment variables.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)


# --------------------------------------
# LOAD VECTOR INDEX
# --------------------------------------
def load_vector_index():
    try:
        st.session_state.index = faiss.read_index("vector.index")
        st.session_state.documents = pickle.load(open("documents.pkl", "rb"))
        st.session_state.loaded = True
        st.success("FAISS Vector Index Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")


if "loaded" not in st.session_state:
    st.session_state.loaded = False

if st.sidebar.button("Load GIS Vector Index"):
    load_vector_index()


# --------------------------------------
# SEARCH FUNCTION
# --------------------------------------
def retrieve_chunks(query, top_k=4):
    if not st.session_state.loaded:
        return []

    index = st.session_state.index
    docs = st.session_state.documents

    query_emb = MODEL.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(docs):
            results.append((docs[idx], float(dist)))

    return results


# --------------------------------------
# GROQ LLM ANSWER
# --------------------------------------
def generate_llm_answer(context, question):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a GIS expert. Answer ONLY using the provided context. "
                        "If the context is insufficient, say so clearly."
                    )
                },
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
                }
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {e}"


# --------------------------------------
# CHAT MEMORY
# --------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --------------------------------------
# CHAT INPUT
# --------------------------------------
prompt = st.chat_input("Ask anything about your GIS documents...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Retrieving GIS knowledge..."):

            # 1. Retrieve chunks
            retrieved = retrieve_chunks(prompt)

            if not retrieved:
                answer = "⚠️ No documents loaded or no relevant chunks found."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                # Build context for LLM
                context_text = ""
                st.markdown("### 📌 Retrieved Chunks")
                for i, (text, score) in enumerate(retrieved, start=1):
                    st.markdown(f"**Chunk {i} — Score: {score:.4f}**")
                    st.write(text)
                    st.write("---")
                    context_text += f"\n\n### Chunk {i}\n{text}"

                # 2. Ask Groq LLM
                st.markdown("### 🤖 Groq LLM Answer")
                answer = generate_llm_answer(context_text, prompt)
                st.markdown(answer)

                # Add to memory
                st.session_state.messages.append({"role": "assistant", "content": answer})
