import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import os
import json

# -----------------------------
# Configuration
# -----------------------------
INDEX_DIR = "indexes"
DATA_DIR = "data"

# -----------------------------
# Initialize embeddings
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Load available subjects
# -----------------------------
def list_subjects():
    return [f.split("_index.json")[0] for f in os.listdir(INDEX_DIR) if f.endswith("_index.json")]

# -----------------------------
# Load FAISS vectorstore for a given subject
# -----------------------------
@st.cache_resource
def load_vectorstore(subject):
    index_path = os.path.join(INDEX_DIR, f"{subject}_index.json")
    if not os.path.exists(index_path):
        st.error(f"Index file not found for subject: {subject}")
        return None

    embeddings = load_embeddings()
    try:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, index_name=subject, allow_dangerous_deserialization=True)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading index for {subject}: {e}")
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Department Research Search", layout="wide")
st.title("🔍 Department Research Paper & Patent Search System")

# Sidebar
st.sidebar.header("Options")
subjects = list_subjects()

if not subjects:
    st.sidebar.warning("No index files found in 'indexes/' folder.")
    st.stop()

selected_subject = st.sidebar.selectbox("Select Subject", subjects)
top_k = st.sidebar.slider("Number of results", 1, 10, 5)

# Load vectorstore
vectorstore = load_vectorstore(selected_subject)
if not vectorstore:
    st.stop()

# -----------------------------
# Search Functionality
# -----------------------------
query = st.text_input("Enter your search query")

if query:
    with st.spinner("Searching..."):
        try:
            results = vectorstore.similarity_search(query, k=top_k)
            if results:
                st.success(f"Found {len(results)} relevant result(s):")
                for i, res in enumerate(results, start=1):
                    st.markdown(f"### {i}. {res.metadata.get('title', 'Untitled')}")
                    st.markdown(f"**Score:** {res.metadata.get('score', 'N/A')}")
                    st.markdown(f"**Content:** {res.page_content[:300]}...")
                    st.markdown("---")
            else:
                st.warning("No results found for this query.")
        except Exception as e:
            st.error(f"Search failed: {e}")
else:
    st.info("Enter a query above to start searching.")

# -----------------------------
# Developer Info
# -----------------------------
st.markdown("---")
st.caption("Developed by Vishal | Powered by FAISS + LangChain + Streamlit")
