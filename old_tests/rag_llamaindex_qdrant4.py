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

        # 6. ONLY sections from docs/sections/
        sections_used = 0
        for section in self.config.get("sections", {}).get("order", []):
            content = section_texts.get(section, "")
            if content.strip():
                paper += f"## {section}\n\n{content[:2500]}...\n\n"
                print(f"✅ Added: {section}")
                sections_used += 1
            else:
                print(f"⏭️ Skip: {section}")

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


if __name__ == "__main__":
    generator = PaperGenerator("config.yaml")
    generator.generate_paper()
