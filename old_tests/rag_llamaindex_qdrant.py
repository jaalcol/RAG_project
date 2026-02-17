import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

print("🔄 RAG SIMPLIFIED - Qdrant + RETRIEVAL ONLY")

# 1. PDFs
documents = SimpleDirectoryReader("docs").load_data()
print(f"✅ {len(documents)} docs")

# 2. CHUNKS
Settings.chunk_size = 200
Settings.chunk_overlap = 20

# 3. Embeddings
Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# 4. Qdrant
client = QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="rag_publication")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. Índice + RETRIEVER
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
retriever = index.as_retriever(similarity_top_k=3)

print("\n🎉 QDRANT RAG RETRIEVER FUNCIONANDO!")
print("📄 Muestra CONTEXTOS relevantes encontrados")
print("=" * 60)

# 6. BUCLE SIMPLE - SOLO RETRIEVAL
while True:
    query = input("\n❓ Pregunta (o 'exit'): ")
    if query.lower() == "exit":
        print("👋 Qdrant retrieval perfecto!")
        break
    
    print("🔍 Buscando en Qdrant...")
    nodes = retriever.retrieve(query)
    
    print(f"\n📊 TOP 3 resultados ({len(nodes)} encontrados):")
    for i, node in enumerate(nodes, 1):
        print(f"\n--- RESULTADO {i} ---")
        print(node.text[:400] + "..." if len(node.text) > 400 else node.text)
        print(f"  📈 Score: {node.score:.3f}")
