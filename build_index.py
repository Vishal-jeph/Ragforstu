import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Check your .env file.")

# Initialize embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Folder where your course content is stored
BASE_PATH = Path("data")

# Folder where we’ll save embeddings
INDEX_PATH = Path("indexes")
INDEX_PATH.mkdir(exist_ok=True)

def clean_texts(texts):
    """Ensure all items are valid strings and remove empties."""
    clean = []
    for t in texts:
        if isinstance(t, (list, dict)):
            t = json.dumps(t)
        if t is None:
            continue
        t = str(t).strip()
        if len(t) > 0:
            clean.append(t)
    return clean

def build_embeddings_for_course(course_name):
    course_path = BASE_PATH / course_name
    print(f"\n📘 Building index for: {course_name}")

    texts = []

    # Read all text files in the course folder
    for file in tqdm(list(course_path.glob("*.txt"))):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append(content)

    texts = clean_texts(texts)
    if not texts:
        print(f"⚠️ No valid text found for {course_name}, skipping.")
        return

    # Create embeddings
    print("🔹 Generating embeddings...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Save embeddings + text
    data = {
        "texts": texts,
        "embeddings": embeddings.tolist()
    }

    with open(INDEX_PATH / f"{course_name}_index.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved index for {course_name} at {INDEX_PATH / f'{course_name}_index.json'}")

def build_all_indexes():
    for folder in BASE_PATH.iterdir():
        if folder.is_dir():
            build_embeddings_for_course(folder.name)

if __name__ == "__main__":
    build_all_indexes()
    print("\n🎉 All indexes built successfully!")
