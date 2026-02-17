import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------
# 1. Load PDFs from 'docs/' folder
# -------------------------------
def load_pdfs(folder="docs"):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            texts.append(text)
    return texts

documents = load_pdfs("docs")
print(f"Loaded {len(documents)} document(s) from PDFs.")

# -------------------------------
# 2. Split documents into chunks
# -------------------------------
def split_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = []
for doc in documents:
    chunks.extend(split_into_chunks(doc))
print(f"Total chunks created: {len(chunks)}")

# -------------------------------
# 3. Create embeddings
# -------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight
embeddings = [embed_model.encode(chunk) for chunk in chunks]
embeddings = np.array(embeddings).astype("float32")
print("Embeddings created for all chunks.")

# -------------------------------
# 4. Index embeddings with FAISS
# -------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("FAISS index created.")

# -------------------------------
# 5. Retrieval function
# -------------------------------
def retrieve(query, top_k=1):
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

# -------------------------------
# 6. Load FLAN-T5 model for generation
# -------------------------------
model_name = "google/flan-t5-small"  # fast and works on CPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# -------------------------------
# 7. Ask a query
# -------------------------------
query = "What was the main finding of the study?"
retrieved_chunks = retrieve(query, top_k=1)
context = retrieved_chunks[0]

prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
answer = generator(prompt, max_new_tokens=150)

print("\nQuery:", query)
print("Retrieved context snippet:\n", context[:500], "...\n")
print("Answer:\n", answer[0]['generated_text'])
