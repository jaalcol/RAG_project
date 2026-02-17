from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient

# Tu setup Qdrant (perfecto)
documents = SimpleDirectoryReader("docs").load_data()
Settings.chunk_size = 200
Settings.chunk_overlap = 20
Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient("localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="rag_publication")
index = VectorStoreIndex.from_documents(documents, StorageContext.from_defaults(vector_store=vector_store))
retriever = index.as_retriever(similarity_top_k=3)

# OLLAMA (¡FUNCIONA!)
llm = Ollama(model="llama3.2:1b")

print("🎉 RAG COMPLETO!")
while True:
    query = input("\n❓: ")
    if query == "exit": break
    nodes = retriever.retrieve(query)
    context = "\n".join([n.text for n in nodes])
    response = llm.complete(f"Context: {context}\nQ: {query}\nA:")
    print(f"💡 {response.text}")
