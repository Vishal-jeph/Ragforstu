import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

st.set_page_config(page_title="Course RAG App", layout="wide")

st.title("📚 Course RAG App")
st.write("Ask questions based on your uploaded course indexes (Computer Networks, Data Mining, etc.)")

INDEX_DIR = "indexes"
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load available index files ---
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
    st.error("No index files found in the 'indexes' folder. Please add files like `computer_networks_index.json` or `data_mining_index.json`.")
    st.stop()

subject = st.selectbox("Choose a subject", list(indexes.keys()))
query = st.text_input("Ask a question related to this subject:")

if st.button("Search") and query:
    index_data = indexes[subject]

    # If the file was saved as a dict instead of list, fix that
    if isinstance(index_data, dict):
        index_data = list(index_data.values())

    # If it’s still a list of strings, convert it to dicts with text only
    if all(isinstance(i, str) for i in index_data):
        texts = index_data
        embeddings = model.encode(texts, convert_to_numpy=True)
    else:
        # Otherwise assume it’s list of dicts with "text" and "embedding"
        texts = [item["text"] for item in index_data]
        embeddings = np.array([item["embedding"] for item in index_data])

    query_emb = model.encode(query, convert_to_numpy=True)

    cos_scores = util.cos_sim(query_emb, embeddings)[0]
    top_results = np.argsort(-cos_scores)[:5]

    st.subheader("Top Matching Results:")
    for idx in top_results:
        st.write(f"**Score:** {cos_scores[idx]:.4f}")
        st.write(texts[idx])
        st.markdown("---")
