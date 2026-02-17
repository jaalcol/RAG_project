# -------------------------------
# PDF QA with LangChain and Hugging Face (public models)
# -------------------------------

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------------
# 1. Load PDFs from 'docs/' folder
# -------------------------------
def load_pdfs(folder="docs"):
    texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            loader = PyPDFLoader(path)
            texts.append(loader.load())
    return [item for sublist in texts for item in sublist]  # flatten list

documents = load_pdfs("docs")
print(f"Loaded {len(documents)} document(s) from PDFs.")

# -------------------------------
# 2. Split text into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(docs)}")

# -------------------------------
# 3. Create embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# 4. Create FAISS vector store
# -------------------------------
db = FAISS.from_documents(docs, embeddings)

# -------------------------------
# 5. Setup Hugging Face LLM (public T5)
# -------------------------------
model_name = "google/flan-t5-large"  # public, no token needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Hugging Face pipeline
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------------
# 6. Setup retrieval-based QA
# -------------------------------
retriever = db.as_retriever(search_kwargs={"k": 3})  # top 3 relevant chunks
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # simple chain
)

# -------------------------------
# 7. Ask a question
# -------------------------------
query = "What was the main finding of the study?"
answer = qa.run(query)
print("\nAnswer:")
print(answer)
