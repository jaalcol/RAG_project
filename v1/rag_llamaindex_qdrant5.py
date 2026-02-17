import os
import re
import pdfplumber
import yaml
from pathlib import Path
from collections import Counter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


class PaperGenerator:
    def __init__(self, config_path="config.yml"):  # ← .yml!
        print("🚀 STARTING WITH:", config_path)
        self.config = self._get_full_defaults()
        
        if os.path.exists(config_path):
            print(f"✅ Found {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
                print("🔍 YAML KEYS:", list(yaml_data.keys()))
                
                personal = yaml_data.get("personal_config", {})
                print("🔍 PERSONAL KEYS:", list(personal.keys()))
                print("🔍 SECTION PDF:", personal.get("section_pdfs"))
                
                self.config.update(personal)
                print("🎉 FINAL section_pdfs:", self.config.get("section_pdfs"))
        else:
            print(f"❌ MISSING: {config_path}")
        
        self.ensure_output_folder()

    def _get_full_defaults(self):
        return {
            "author_name": "Author",
            "title_template": "Advanced {terms} {title_suffix}",
            "title_suffix": "Framework 2026",
            "abstract_template": "Framework focusing on {terms}.",
            "sections": {"order": []},
            "section_pdfs": {},
            "output": {"filename_prefix": "paper", "output_dir": "results"},
            "bad_terms": [],
            "skip_words": [],
            "min_phrase_length": 12,
            "require_vowels": True,
            "llm_model": "llama3.2:1b",
            "chunk_size": 400,
            "chunk_overlap": 50,
            "top_k": 2,
            "references": {"count": 5},
            "journal_pool": ["Journal"],
        }

    def ensure_output_folder(self):
        Path(self.config["output"]["output_dir"]).mkdir(exist_ok=True)

    def get_base_path(self):
        return Path(__file__).resolve().parent

    def clean_text(self, text):
        if not text: return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"[^A-Za-z0-9 .,;:()-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def perfect_term_filter(self, all_text):
        patterns = [
            r'\b[A-Z][a-z]{4,}(?:\s+[A-Z][a-z]{4,}){0,2}\b',
            r'\bPharmaco[A-Za-z]+\b',
            r'\bRisk\s*[A-Za-z]+\b'
        ]
        bad_terms = set(self.config.get("bad_terms", []))
        good_terms = []
        
        for pattern in patterns:
            phrases = re.findall(pattern, all_text, re.IGNORECASE)
            for phrase in phrases:
                phrase_lower = phrase.lower()
                if len(phrase_lower) >= self.config.get("min_phrase_length", 12):
                    if not any(bad in phrase_lower for bad in bad_terms):
                        good_terms.append(phrase.title())
        
        return list(dict.fromkeys(good_terms))[:3]

    def load_template(self):
        base_path = self.get_base_path()
        template_path = base_path / "docs" / "template.pdf"
        if template_path.exists():
            try:
                with pdfplumber.open(template_path) as pdf:
                    text = ""
                    for page in pdf.pages[:1]:
                        text += self.clean_text(page.extract_text() or "")
                print(f"📄 Template loaded: {template_path}")
                return text
            except:
                pass
        print("⚠️ No template.pdf found")
        return ""

    def load_section_pdfs(self):
        """Load PDFs from docs/sections/ matching YAML config"""
        base_path = self.get_base_path()
        sections_folder = base_path / "docs" / "sections"
        section_texts = {}
        
        if not sections_folder.exists():
            print(f"❌ docs/sections/ folder not found")
            return section_texts
        
        for section, pdf_name in self.config.get("section_pdfs", {}).items():
            pdf_path = sections_folder / pdf_name
            if pdf_path.exists():
                try:
                    text = ""
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages:
                            text += self.clean_text(page.extract_text() or "") + "\n"
                    section_texts[section] = text
                    print(f"✅ Section {section}: {pdf_path}")
                except Exception as e:
                    print(f"⚠️ Error {section}: {e}")
            else:
                print(f"❌ Missing section {section}: {pdf_path}")
        
        return section_texts

    def load_data_pdfs(self):
        """Load ALL PDFs from docs/data/ for RAG enrichment"""
        base_path = self.get_base_path()
        data_folder = base_path / "docs" / "data"
        data_paths = []
        
        if data_folder.exists():
            data_paths = [str(p) for p in data_folder.glob("*.pdf")]
            print(f"📚 Found {len(data_paths)} data PDFs for RAG")
        else:
            print("ℹ️ docs/data/ folder not found - no RAG enrichment")
        
        return data_paths

    def extract_terms(self):
        """Extract terms from sections + data PDFs"""
        base_path = self.get_base_path()
        
        all_text = ""
        
        # Sections
        section_texts = self.load_section_pdfs()
        print("🔍 DEBUG SECTIONS:")
        for section, pdf_name in self.config.get("section_pdfs", {}).items():
            pdf_path = (base_path / "docs" / "sections" / pdf_name)
            print(f"  '{section}' → '{pdf_name}' → {pdf_path} → EXISTS: {pdf_path.exists()}")

        for content in section_texts.values():
            all_text += content
        
        # Data PDFs (enrichment)
        data_paths = self.load_data_pdfs()
        base_path = self.get_base_path()
        data_folder = base_path / "docs" / "data"
        if data_folder.exists():
            for pdf_path in data_folder.glob("*.pdf"):
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        for page in pdf.pages[:3]:
                            all_text += self.clean_text(page.extract_text() or "") + "\n"
                except:
                    continue
        
        terms = self.perfect_term_filter(all_text)
        print(f"🔑 Extracted terms: {terms}")
        return terms, section_texts

    def setup_rag(self, data_paths):
        """RAG from docs/data/ ONLY"""
        if not data_paths:
            print("⚠️ No RAG data - LLM only mode")
            return None, Ollama(model=self.config.get("llm_model", "llama3.2:1b"))
        
        try:
            Settings.chunk_size = self.config.get("chunk_size", 400)
            Settings.chunk_overlap = self.config.get("chunk_overlap", 50)
            Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            documents = SimpleDirectoryReader(input_files=data_paths).load_data()
            index = VectorStoreIndex.from_documents(documents)
            retriever = index.as_retriever(similarity_top_k=self.config.get("top_k", 2))
            llm = Ollama(model=self.config.get("llm_model", "llama3.2:1b"))
            print(f"🚀 RAG ready with {len(data_paths)} data PDFs")
            return retriever, llm
        except Exception as e:
            print(f"⚠️ RAG failed: {e}")
            return None, Ollama(model=self.config.get("llm_model", "llama3.2:1b"))

    def generate_paper(self):
        base_path = self.get_base_path()
        
        # 1. Extract terms & load sections
        terms, section_texts = self.extract_terms()
        
        # 2. Generate title/abstract from YAML
        title = self.config["title_template"].format(
            terms=" / ".join(terms),
            title_suffix=self.config["title_suffix"]
        )
        abstract = self.config["abstract_template"].format(terms=", ".join(terms))
        
        # 3. Author from YAML
        author = self.config.get("author_name", "Author")
        co_authors = ", ".join(self.config.get("co_authors", []))
        if co_authors: author += f", {co_authors}"
        
        affiliations = f"{self.config.get('affiliation1', '')}, {self.config.get('affiliation2', '')}".strip(", ")
        if affiliations and self.config.get('city'):
            affiliations += f", {self.config.get('city')}, {self.config.get('country', '')}"

        # 4. RAG from data/ folder only
        data_paths = self.load_data_pdfs()
        retriever, llm = self.setup_rag(data_paths)

        # 5. Build paper
        paper = f"# {title}\n\n"
        paper += f"**{author}**"
        if affiliations: paper += f"\n_{affiliations}_"
        paper += "\n\n"
        
        if self.config.get('email'): paper += f"**Email:** {self.config.get('email')}\n"
        if self.config.get('orcid'): paper += f"**ORCID:** {self.config.get('orcid')}\n\n"
        
        paper += f"## Abstract\n{abstract}\n\n"
        paper += f"**Index Terms**— {', '.join(terms).lower()}\n\n"

        # 6. RAG-ENHANCED sections
        sections_used = 0
        for section in self.config.get("sections", {}).get("order", []):
            original_content = section_texts.get(section, "")
            if original_content.strip():
                
                # RAG query: Enhance this section
                enhancement_prompt = f"""
                ORIGINAL: {original_content[:3000]}

                Rewrite this {section} for Drug Safety journal:
                - Keep ALL technical content and facts  
                - Improve scientific writing style ONLY
                - Smooth transitions ONLY
                - NO SUBSECTION HEADERS like **Background**, **Recent Findings**
                - NO bold subsection titles
                - Continuous flowing narrative like real journal papers
                - 400-600 words
                - Add 1-2 citations [1],[2]

                Keep SAME format as input - NO new subsections.

                ENHANCED VERSION:
                """
                
                # Use RAG context if available
                if retriever:
                    context_nodes = retriever.retrieve(f"pharmacovigilance {section.lower()} methods")
                    context = "\n".join([node.text[:500] for node in context_nodes])
                    full_prompt = f"{enhancement_prompt}\n\nRAG CONTEXT:\n{context}"
                else:
                    full_prompt = enhancement_prompt
                    
                # Generate enhanced content
                enhanced_content = llm.complete(full_prompt).text
                paper += f"## {section}\n\n{enhanced_content}\n\n"
                print(f"✨ RAG-enhanced: {section}")
                sections_used += 1

        # 7. References
        paper += "## REFERENCES\n\n"
        paper += self.generate_references(terms)

        # 8. Footer from YAML
        if self.config.get('acknowledgments'):
            paper += f"\n*Acknowledgments*—{self.config.get('acknowledgments')}\n"
        if self.config.get('funding'):
            paper += f"*Funding*—{self.config.get('funding')}\n"
        if self.config.get('conflicts'):
            paper += f"*Conflicts of Interest*—{self.config.get('conflicts')}\n"

        # 9. Save
        output_dir = base_path / self.config["output"]["output_dir"]
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{self.config['output']['filename_prefix']}_{self.config['title_suffix'].replace(' ', '_').lower()}.md"
        filepath = output_dir / filename
        filepath.write_text(paper, encoding="utf-8")
        print(f"✅ Saved: {filepath}")
        print(f"📊 Summary: {sections_used} sections, {len(data_paths)} RAG PDFs")
        return paper

    def generate_references(self, terms):
        journals = self.config.get("journal_pool", ["Journal"])
        count = self.config.get("references", {}).get("count", 5)
        refs = []
        
        for i in range(count):
            journal = journals[i % len(journals)]
            title = f"{terms[0] if terms else 'Research'}: Automated Framework"
            ref = f"[{i+1}] J. Alhambra et al., \"{title}\", *{journal}*, vol. {24+i}, no. {i+1}, pp. {100+i*15}-{114+i*15}, 2026."
            refs.append(ref)
        return "\n".join(refs)
    
    def chat_about_paper(self):
        """Interactive chat about the paper using internal + RAG context"""
        print("\n💬 === PAPER CHAT MODE ===")
        print("Ask about your paper, get RAG-enhanced answers!")
        print("Type 'quit' to exit\n")
        
        # Internal paper context - FIXED f-string
        terms, section_texts = self.extract_terms()
        title = self.config['title_template'].format(terms=" / ".join(terms), title_suffix=self.config['title_suffix'])
        
        # Build sections summary WITHOUT backslash in f-string
        sections_summary = []
        for sec, content in section_texts.items():
            cleaned = self.clean_text(content)[:800]
            sections_summary.append(f"{sec}:\n{cleaned}...")
        sections_text = "\n".join(sections_summary)
        
        internal_context = f"""
        PAPER INFO:
        - Title: {title}
        - Terms: {terms}
        - Sections: {list(section_texts.keys())}
        - RAG PDFs: {len(self.load_data_pdfs())}
        """
        
        # Setup RAG
        data_paths = self.load_data_pdfs()
        retriever, llm = self.setup_rag(data_paths)
        
        while True:
            user_query = input("\n❓ Your question: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            # Build chat prompt - NO backslashes in f-strings
            chat_prompt = (
                f"You are a pharmacovigilance research assistant analyzing this paper:\n\n"
                f"{internal_context}\n\n"
                f"SECTION CONTENT:\n{sections_text}\n\n"
                f"USER QUESTION: {user_query}\n\n"
                "Answer as a journal peer reviewer with specific, actionable feedback.\n"
                "Use RAG context when relevant. Cite [internal sections] or [RAG-1], [RAG-2]."
            )
            
            # Add RAG context
            if retriever:
                rag_nodes = retriever.retrieve(f"pharmacovigilance {user_query.lower()}")
                rag_context = "\n".join([f"RAG-{i+1}: {node.text[:300]}" for i, node in enumerate(rag_nodes)])
                chat_prompt += f"\n\nEXTERNAL RAG CONTEXT:\n{rag_context}"
            
            # Generate response
            response = llm.complete(chat_prompt).text
            print(f"\n🤖 AI REVIEWER:\n{response}\n")
        
        print("👋 Chat ended")

    def integrate_feedback(self, reviewer_comments):
        """5-STEP LOOP: Fix → Section → Suggest → Confirm → Repeat"""
        data_paths = self.load_data_pdfs()
        _, llm = self.setup_rag(data_paths)
        
        print(f"\n🔧 Processing: {reviewer_comments}")
        
        while True:
            # 1. ASK FOR FIX
            fix_input = input("\n1️⃣ Fix text (or 'done'): ").strip()
            if fix_input.lower() == 'done':
                break
            
            # 2. ASK FOR SECTION (numbered)
            print("\n📂 Sections:")
            terms, section_texts = self.extract_terms()
            sections = list(section_texts.keys()) + ['Abstract', 'References']
            for i, sec in enumerate(sections, 1):
                print(f"{i}. {sec}")
            
            section_num = input(f"2️⃣ Section (1-{len(sections)}): ").strip()
            if not section_num.isdigit() or not (1 <= int(section_num) <= len(sections)):
                print("❌ Invalid section")
                continue
            
            target_section = sections[int(section_num)-1]
            
            # 3. SUGGEST REFINEMENT
            suggestion = llm.complete(f"""
            Original fix: "{fix_input}"
            Paper section: {target_section}
            Suggest improved version for pharmacovigilance paper:
            """).text.strip()
            
            print(f"\n3️⃣ SUGGESTION: {suggestion}")
            
            # 4. CONFIRM SUBSTITUTE
            final_fix = input(f"4️⃣ Use '{suggestion[:50]}...' ? (y/n/edit): ").strip().lower()
            if final_fix == 'n':
                continue
            elif final_fix == 'edit':
                final_fix = input("Edited fix: ").strip()
            else:
                final_fix = suggestion
            
            # 5. APPLY & LOOP
            self._smart_insert(target_section, final_fix)
            print("\n🔄 Ready for next fix...")

    def _smart_insert(self, section_name, fix_text):
        """Pure content only - NO headers, NO prefixes, NO junk"""
        base_path = self.get_base_path()
        output_dir = base_path / self.config["output"]["output_dir"]
        filename = f"{self.config['output']['filename_prefix']}_{self.config['title_suffix'].replace(' ', '_').lower()}.md"
        paper_path = output_dir / filename
        
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper = f.read()
        
        marker = f"## {section_name}"
        pos = paper.find(marker)
        if pos == -1:
            print(f"❌ '{section_name}' not found!")
            return
        
        # ULTRA CLEAN: Remove ALL junk phrases completely
        lines = fix_text.split('\n')
        pure_content = []
        
        junk_phrases = [
            "Here's a rewritten", "Enhanced Version", "Here is the rewritten", 
            "smooth transition", "version of the", "in an enhanced format",
            "In medical research"  # Skip generic starters
        ]
        
        for line in lines:
            line = line.strip()
            if (line and len(line) > 20 and  # Real content
                not any(junk in line.lower() for junk in junk_phrases) and
                not line.startswith("**") and  # No bold headers
                line != "Hello"):
                pure_content.append(line)
        
        # Join ONLY pure content
        clean_text = "\n".join(pure_content).strip()
        
        if not clean_text:
            print("❌ No valid content found!")
            return
        
        # Insert PURE CONTENT after section header
        insert_pos = paper.find('\n\n', pos) + 2
        paper = paper[:insert_pos] + f"\n\n{clean_text}\n\n" + paper[insert_pos:]
        
        paper_path.write_text(paper, encoding='utf-8')
        print(f"✅ Pure content inserted: {len(pure_content)} lines")

    def paper_preview(self):
        """For web display"""
        try:
            base_path = self.get_base_path()
            paper_path = base_path / self.config["output"]["output_dir"] / \
                        f"{self.config['output']['filename_prefix']}_{self.config['title_suffix'].replace(' ', '_').lower()}.md"
            if paper_path.exists():
                return paper_path.read_text(encoding='utf-8')
        except:
            pass
        return "# No paper - click Generate first!"

    def chat_about_paper_question(self, question):
        """Quick chat answer - FIXED VERSION"""
        try:
            # 1. FORCE PAPER CONTEXT (no RAG dependency)
            terms, section_texts = self.extract_terms()
            paper_context = ""
            
            # Build context from ALL available sections
            for section, content in section_texts.items():
                paper_context += f"\n## {section}\n{self.clean_text(content)[:800]}...\n"
            
            # Add paper metadata
            paper_context = f"""
            PAPER: Advanced {terms} Framework 2026 by Javier Alhambra
            
            {paper_context}
            
            AUTHOR AFFILIATION: Pharmacovigilance Department, Valencia Spain
            JOURNAL TARGET: Drug Safety
            """
            
            # 2. Setup LLM with paper context
            data_paths = self.load_data_pdfs()
            _, llm = self.setup_rag(data_paths)
            
            # 3. EXPLICIT PAPER-ONLY PROMPT
            prompt = f"""
            You are reviewing this Drug Safety journal paper by Javier Alhambra (Valencia):
            
            PAPER CONTENT:
            {paper_context}
            
            QUESTION: {question}
            
            Answer ONLY from this paper. Quote sections directly. 
            If no info found: "Not covered in this paper"
            """
            
            response = llm.complete(prompt).text.strip()
            return response if response else "No paper content available"
            
        except Exception as e:
            return f"Paper error: {str(e)}"



if __name__ == "__main__":
    generator = PaperGenerator("config.yaml")
    generator.generate_paper()

    # === NEW: FEEDBACK INTEGRATION ===
    print("\n🔧 REVIEWER FEEDBACK MODE")
    feedback = input("Paste reviewer comment (or Enter to skip): ")
    if feedback:
        generator.integrate_feedback(feedback)

    # Interactive chat option
    chat = input("\n💬 Launch paper chat? (y/n): ").lower()
    if chat == 'y':
        generator.chat_about_paper()  # ← generator not self

