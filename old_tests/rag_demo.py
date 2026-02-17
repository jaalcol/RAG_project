from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --------- Documents ---------
docs = [
    "RAG combines retrieval with generation to improve text generation accuracy.",
    "FAISS is a fast vector store for embeddings.",
    "Transformer models have revolutionized NLP tasks."
]

# --------- Create embeddings ---------
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

# --------- Build FAISS index ---------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --------- Simple retriever function ---------
def retrieve(query, k=2):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [docs[i] for i in indices[0]]

# --------- Load HuggingFace FLAN-T5 model for generation ---------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model_lm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
generator = pipeline("text-generation", model=model_lm, tokenizer=tokenizer)

# --------- RAG query function ---------
def rag_query(query):
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)
    input_text = f"Answer the question using the context:\nContext: {context}\nQuestion: {query}"
    result = generator(input_text, max_length=200)
    return result[0]['generated_text']

# --------- Demo ---------
query = "Explain how RAG helps in NLP"
answer = rag_query(query)
print("Query:", query)
print("Answer:", answer)
