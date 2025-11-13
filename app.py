import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Research Paper & Patent Search", layout="wide")

st.title("🔍 Research Paper & Patent Search System")
st.write("Upload your precomputed embeddings JSON file and ask questions to find the most relevant entries.")

# File uploader
uploaded_file = st.file_uploader("Upload your index file (JSON)", type=["json"])

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

if uploaded_file:
    try:
        index_data = json.load(uploaded_file)

        # ✅ Handle both supported formats
        if isinstance(index_data, dict) and "texts" in index_data and "embeddings" in index_data:
            texts = index_data["texts"]
            embeddings = np.array(index_data["embeddings"])
        elif isinstance(index_data, list) and "text" in index_data[0]:
            texts = [item["text"] for item in index_data]
            embeddings = np.array([item["embedding"] for item in index_data])
        else:
            st.error("❌ Unsupported JSON structure. Please ensure your file has either {'texts': [...], 'embeddings': [...]} or [{'text': ..., 'embedding': [...]}].")
            st.stop()

        # Search box
        query = st.text_input("Enter your query:")
        if query:
            query_emb = model.encode(query, convert_to_numpy=True)
            similarities = cosine_similarity([query_emb], embeddings)[0]
            top_k = min(5, len(similarities))
            top_indices = np.argsort(similarities)[::-1][:top_k]

            st.subheader("Top Results:")
            for i, idx in enumerate(top_indices):
                st.markdown(f"**Result {i+1} (Score: {similarities[idx]:.3f})**")
                st.write(texts[idx])
                st.divider()

    except Exception as e:
        st.error(f"❌ Error reading or processing JSON: {e}")
