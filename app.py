import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("RAG Question Answer System")

# -----------------------
# Load RAG Index at startup
# -----------------------

INDEX_FILE = "runtime.txt"   # Your file that contains texts + embeddings

@st.cache_resource
def load_index():
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate JSON format
        if "texts" not in data or "embeddings" not in data:
            st.error("❌ Invalid JSON format. Required keys: texts, embeddings")
            st.stop()

        texts = data["texts"]
        embeddings = np.array(data["embeddings"])

        if len(texts) != len(embeddings):
            st.error("❌ Number of texts and embeddings do not match.")
            st.stop()

        return texts, embeddings
    
    except Exception as e:
        st.error(f"Error loading index: {e}")
        st.stop()

texts, stored_embeddings = load_index()

# -----------------------
# Load the embedding model
# -----------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# -----------------------
# Search Function
# -----------------------

def retrieve_top_k(query, k=3):
    query_emb = model.encode([query], convert_to_numpy=True)

    sims = cosine_similarity(query_emb, stored_embeddings)[0]

    top_k_idx = sims.argsort()[::-1][:k]

    results = []
    for idx in top_k_idx:
        results.append({
            "text": texts[idx],
            "score": float(sims[idx])
        })
    return results


# -----------------------
# UI
# -----------------------

query = st.text_input("Ask a question")

if query:
    results = retrieve_top_k(query, k=3)

    st.subheader("Top Relevant Answers:")

    for i, r in enumerate(results, start=1):
        st.markdown(f"### Result {i} (Similarity: {r['score']:.3f})")
        st.write(r["text"])
        st.markdown("---")
