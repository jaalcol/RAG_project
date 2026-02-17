import os
from PyPDF2 import PdfReader
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
# 2. Chunk documents
# -------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

all_chunks = []
for doc in documents:
    all_chunks.extend(chunk_text(doc))
print(f"Total chunks created: {len(all_chunks)}")

# -------------------------------
# 3. Create embeddings with SentenceTransformer
# -------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(all_chunks, show_progress_bar=True)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

# -------------------------------
# 4. Build FAISS index
# -------------------------------
embedding_dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(chunk_embeddings)
print("FAISS index created.")

# -------------------------------
# 5. Load T5 model for generation
# -------------------------------
model_name = "t5-small"  # you can switch to t5-base or t5-large
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",  # Using text-generation for T5
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU; set 0 if using GPU
)

# -------------------------------
# 6. Query function
# -------------------------------
def query_rag(question, top_k=3, max_tokens=150):
    # 1. Embed the question
    question_embedding = embedder.encode([question]).astype("float32")
    
    # 2. Search in FAISS
    D, I = index.search(question_embedding, top_k)
    retrieved_chunks = [all_chunks[i] for i in I[0]]
    
    # 3. Concatenate retrieved chunks as context
    context = "\n".join(retrieved_chunks)
    
    # 4. Form prompt for T5
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    
    # 5. Generate answer
    outputs = generator(prompt, max_new_tokens=max_tokens, do_sample=False)
    return outputs[0]["generated_text"]

# -------------------------------
# 7. Example usage
# -------------------------------
question = "What was the main finding of the study?"
answer = query_rag(question)
print("\nQuery:", question)
print("Answer:", answer)
