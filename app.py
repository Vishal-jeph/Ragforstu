import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

st.set_page_config(page_title="Department RAG Q&A", page_icon="📘", layout="wide")
st.title("🎓 Department Q&A System")

# Path to indexes folder
INDEX_DIR = "indexes"

# Load indexes
indexes = {}
for filename in os.listdir(INDEX_DIR):
    if filename.endswith("_index.json"):
        subject = filename.replace("_index.json", "")
        try:
            with open(os.path.join(INDEX_DIR, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                if "texts" in data and "embeddings" in data:
                    indexes[subject] = data
                else:
                    st.warning(f"⚠️ {filename} has invalid structure — missing 'texts' or 'embeddings'")
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {e}")

if not indexes:
    st.error("No valid index files found in the 'indexes' folder.")
    st.stop()

# Subject selection
subject = st.selectbox("Select Subject:", list(indexes.keys()))

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# User query
query = st.text_input("Ask your question:")

if query:
    index_data = indexes[subject]
    texts = index_data["texts"]
    embeddings = np.array(index_data["embeddings"])

    # Encode query
    query_emb = model.encode(query, convert_to_numpy=True)

    # Compute cosine similarity
    similarities = np.dot(embeddings, query_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    # Get top 3 results
    top_indices = similarities.argsort()[-3:][::-1]

    st.subheader("Top Relevant Answers:")
    for i, idx in enumerate(top_indices):
        st.markdown(f"**Result {i+1} (Similarity: {similarities[idx]:.3f})**")
        st.write(texts[idx])
        st.markdown("---")
