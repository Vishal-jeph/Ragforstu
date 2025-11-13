import streamlit as st
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ------------------ CONFIG ------------------
INDEX_DIR = "indexes"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()

# ------------------ LOAD AVAILABLE INDEXES ------------------
def load_json_indexes():
    indexes = {}
    for file in os.listdir(INDEX_DIR):
        if file.endswith("_index.json"):
            subject = file.replace("_index.json", "")
            with open(os.path.join(INDEX_DIR, file), "r") as f:
                data = json.load(f)
            indexes[subject] = data
    return indexes

indexes = load_json_indexes()
if not indexes:
    st.error("No JSON indexes found in the 'indexes/' folder.")
    st.stop()

# ------------------ STREAMLIT UI ------------------
st.title("📚 Course RAG App (JSON-based)")
subject = st.selectbox("Select Subject", list(indexes.keys()))
query = st.text_input("Enter your question:")

if st.button("Search") and query:
    index_data = indexes[subject]

    # Load texts and embeddings
    texts = [item["text"] for item in index_data]
    embeddings = np.array([item["embedding"] for item in index_data])

    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity
    cos_scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()

    # Get top 5
    top_indices = np.argsort(cos_scores)[-5:][::-1]

    st.subheader("🔍 Top Relevant Results:")
    for idx in top_indices:
        st.write(f"**Score:** {cos_scores[idx]:.4f}")
        st.write(texts[idx])
        st.markdown("---")

st.caption("Built with ❤️ using Sentence Transformers + Streamlit")
