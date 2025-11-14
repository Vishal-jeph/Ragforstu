import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

st.title("RAG Question Answering System")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# Load precomputed JSON index
def load_index(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  
        return data["texts"], np.array(data["embeddings"])
    except Exception as e:
        st.error(f"Error loading embedding file: {e}")
        return None, None

# SUBJECT OPTIONS
subjects = {
    "Computer Networks": "indexes/computer_networks_index.json",
    "Data Science": "indexes/data_mining_index.json",
}

subject = st.selectbox("Select Subject", options=list(subjects.keys()))

texts, embeddings = load_index(subjects[subject])

query = st.text_input("Ask any question:")

def search(query, texts, embeddings):
    query_emb = model.encode([query])[0]
    similarities = np.dot(embeddings, query_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    top_idx = np.argmax(similarities)
    return similarities[top_idx], texts[top_idx]

if st.button("Search") and query:
    score, result = search(query, texts, embeddings)

    st.subheader("Top Answer")
    st.write(f"**(Similarity Score: {round(float(score), 3)})**")
    st.write(result)
