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

# AÑADE ESTO al final de tu script actual
def generate_paper(title="Mi Meta-Análisis", sections=["Intro", "Metodología", "Resultados", "Conclusiones"]):
    """Genera paper completo desde tus PDFs"""
    
    paper = f"# {title}\n\n"
    
    for section in sections:
        print(f"📝 Generando sección: {section}")
        
        # RAG para cada sección
        query = f"¿Qué dice sobre {section.lower()} en los documentos?"
        nodes = retriever.retrieve(query)
        context = "\n".join([n.text for n in nodes])
        
        # Ollama genera sección
        prompt = f"""Escribe sección "{section}" de paper científico.
        
Contexto de PDFs: {context}
        
Estructura:
- 150-200 palabras
- Estilo académico formal
- Lenguaje español
- Cita autores cuando sea posible"""
        
        response = llm.complete(prompt)
        paper += f"## {section}\n{response.text}\n\n"
    
    # Guarda paper
    with open("paper_generado.md", "w") as f:
        f.write(paper)
    
    print("🎉 PAPER GENERADO: paper_generado.md")
    return paper

# USO
print(generate_paper("Meta-Análisis Farmacovigilancia 2026"))
