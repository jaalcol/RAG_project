
source rag_env310/bin/activate

To run the file
```
python3 rag_llamaindex_qdrant2.py
```

Set up environmetn:
Terminal 1: docker run -p 6333:6333 qdrant/qdrant  (running)
Terminal 2: ollama serve                           (running)
Terminal 3: ollama pull llama3.2:1b              (EXECUTE)
Terminal 4: python3 rag_llamaindex_qdrant5.py     (EXECUTE)



Stop server
```
Ctrl+c
or 
pkill ollama
```

streamlit run web_app.py
http://localhost:8501

Question for the chat
"Should I add real-world evidence (RWE) methodology?"

✅ You Already Have (Strong Foundation)
text
1. ✅ Smart literature search     ← RAG chat + literature_search()
2. ✅ Automated text generation   ← RAG-enhanced sections  
3. ✅ Real-time proofreading      ← proofread_paper()



Comments Fix
Improve The multifaceted relationship between anxiety disorders and insomnia has garnered significant attention in recent years. Insomnia, a widespread condition affecting millions worldwide, can exacerbate anxiety and panic disorders, conversely contributing to their development or perpetuation [1]. As such, addressing one condition alone may not be sufficient to fully manage the interplay between these two, necessitating a more comprehensive treatment approach.