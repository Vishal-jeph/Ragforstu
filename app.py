import streamlit as st
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------------
# 2. Load the pre-built index
# -----------------------------
INDEX_DIR = "indexes"
if not os.path.exists(INDEX_DIR):
    st.error("⚠️ No index found. Please run build_index.py first.")
    st.stop()

# Load embeddings and metadata
with open(os.path.join(INDEX_DIR, "embeddings.json"), "r") as f:
    embeddings_data = json.load(f)

with open(os.path.join(INDEX_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

# Convert list back to numpy array
doc_embeddings = np.array(embeddings_data["embeddings"])
docs = metadata["documents"]

# -----------------------------
# 3. Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 4. Streamlit App UI
# -----------------------------
st.title("📚 Department Research Paper & Patent Search")
st.write("Search and summarize research papers or patents from your department database.")

query = st.text_input("🔍 Enter your search query:")
num_results = st.slider("Number of results to display", 1, 10, 5)

# -----------------------------
# 5. Perform Semantic Search
# -----------------------------
if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0].cpu().numpy()

    # Fix: handle small arrays safely
    k = min(num_results, len(cos_scores))
    top_results = np.argpartition(-cos_scores, range(k))[:k]
    top_results = top_results[np.argsort(-cos_scores[top_results])]

    st.subheader("🔎 Top Matching Documents:")
    for idx in top_results:
        st.write(f"**{docs[idx][:150]}...**")
        st.caption(f"Relevance Score: {cos_scores[idx]:.4f}")

    # -----------------------------
    # 6. Summarize / Generate Answer
    # -----------------------------
    if st.button("✨ Generate Summary using Gemini"):
        context = "\n\n".join([docs[idx] for idx in top_results])
        prompt = f"""You are an expert research assistant.
Use the following research abstracts and patent summaries to answer the query:

Query: {query}
Context:
{context}

Provide a concise, well-structured summary with key findings and relevance.
"""

        try:
            gemini_model = genai.GenerativeModel("gemini-1.5-pro")
            response = gemini_model.generate_content(prompt)
            st.markdown("### 🧠 Gemini Summary")
            st.write(response.text)

        except Exception as e:
            st.error(f"❌ Gemini API Error: {e}")
