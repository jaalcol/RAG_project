import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- 1. Read all PDFs in the docs folder ---
def load_pdfs(folder="docs"):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            texts.append(text)
    return texts

documents = load_pdfs("docs")
print(f"Loaded {len(documents)} documents from PDFs.")

# --- 2. Create embeddings ---
model = SentenceTransformer("all-MiniLM-L6-v2")  # fast and small
embeddings = [model.encode(doc) for doc in documents]
embeddings = np.array(embeddings).astype("float32")

# --- 3. Index embeddings with FAISS ---
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("FAISS index created with documents embeddings.")

# --- 4. Simple retrieval function ---
def retrieve(query, top_k=1):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    results = [documents[i] for i in indices[0]]
    return results

# --- 5. Demo query ---
query = "What was the main finding of the study?"
results = retrieve(query)
print("Query:", query)
print("Retrieved text snippet:\n", results[0][:500], "...\n")