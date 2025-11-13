import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Department Research & Paper Search", layout="wide")

st.title("🎓 Department Research & Paper Search System")

# ✅ Load model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ✅ Load all subject indexes automatically
def load_index(subject):
    path = f"indexes/{subject}_index.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "texts" in data and "embeddings" in data:
        texts = data["texts"]
        embeddings = np.array(data["embeddings"])
    elif isinstance(data, list) and "text" in data[0]:
        texts = [item["text"] for item in data]
        embeddings = np.array([item["embedding"] for item in data])
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    return texts, embeddings


# ✅ Sidebar for subject selection
st.sidebar.header("Select Subject")
subjects = ["computer_networks", "data_mining"]
subject = st.sidebar.selectbox("Choose a subject:", subjects)

try:
    texts, embeddings = load_index(subject)
except Exception as e:
    st.error(f"Error loading {subject} index: {e}")
    st.stop()

# ✅ Main interface
query = st.text_input("Ask a question related to the subject:")
if query:
    query_emb = model.encode(query, convert_to_numpy=True)
    similarities = cosine_similarity([query_emb], embeddings)[0]
    top_k = min(5, len(similarities))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    st.subheader("Top Relevant Answers:")
    for i, idx in enumerate(top_indices):
        st.markdown(f"**Result {i+1} (Similarity: {similarities[idx]:.3f})**")
        st.write(texts[idx])
        st.divider()
