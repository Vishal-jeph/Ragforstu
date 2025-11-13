import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

st.set_page_config(page_title="Course RAG App", layout="wide")

st.title("📚 Course RAG App")
st.write("Ask questions from your course indexes (Computer Networks, Data Mining, etc.)")

INDEX_DIR = "indexes"
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load indexes ---
def load_indexes():
    indexes = {}
    for file in os.listdir(INDEX_DIR):
        if file.endswith("_index.json"):
            subject = file.replace("_index.json", "")
            with open(os.path.join(INDEX_DIR, file), "r", encoding="utf-8") as f:
                try:
                    indexes[subject] = json.load(f)
                except Exception as e:
                    st.warning(f"⚠️ Error loading {file}: {e}")
    return indexes

indexes = load_indexes()

if not indexes:
    st.error("No index files found in 'indexes'. Please add JSON files like 'computer_networks_index.json'.")
    st.stop()

subject = st.selectbox("Choose a subject", list(indexes.keys()))
query = st.text_input("Ask a question related to this subject:")

if st.button("Search") and query:
    index_data = indexes[subject]

    # Convert dict -> list if needed
    if isinstance(index_data, dict):
        index_data = list(index_data.values())

    # Detect data format
    first_item = index_data[0] if len(index_data) > 0 else None

    if isinstance(first_item, str):
        # List of texts
        texts = index_data
        embeddings = model.encode(texts, convert_to_numpy=True)

    elif isinstance(first_item, list) and len(first_item) == 2:
        # List of [text, embedding]
        texts = [i[0] for i in index_data]
        embeddings = np.array([i[1] for i in index_data])

    elif isinstance(first_item, dict):
        # List of {"text": ..., "embedding": ...}
        texts = [i["text"] for i in index_data]
        embeddings = np.array([i["embedding"] for i in index_data])

    else:
        st.error("❌ Unsupported JSON structure. Please check your index file format.")
        st.stop()

    # Encode query
    query_emb = model.encode(query, convert_to_numpy=True)
    cos_scores = util.cos_sim(query_emb, embeddings)[0]
    top_results = np.argsort(-cos_scores)[:5]

    st.subheader("Top Matching Results:")
    for idx in top_results:
        st.write(f"**Score:** {cos_scores[idx]:.4f}")
        st.write(texts[idx])
        st.markdown("---")
