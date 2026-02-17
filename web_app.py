import streamlit as st
from paper_generator import PaperGenerator
import os
from pathlib import Path
from datetime import datetime

# CONFIG
st.set_page_config(layout="wide", page_title="🧪 PV Paper Assistant 3.0")
generator = PaperGenerator("config.yaml")

# CSS
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧪 Pharmacovigilance Paper Assistant")
st.markdown("**Javier Alhambra** - Valencia Pharma - Drug Safety Ready")

# === 3-PANEL LAYOUT ===
col_left, col_center, col_right = st.columns([1.2, 2, 1])

# LEFT PANEL: YOUR OLD DOCUMENT STRUCTURE + NEW FEATURES
with col_left:
    st.header("📚 **Document Structure**")
    
    # 🔥 YOUR ORIGINAL DOCUMENT STRUCTURE WITH % BAR
    sections = [
        "Title Page", "Abstract", "I. Introduction", "II. Methods", 
        "III. Results", "IV. Discussion", "Ethics", "Acknowledgements"
    ]
    
    # Track completion with checkboxes (PERSISTENT)
    if "section_status" not in st.session_state:
        st.session_state.section_status = {sec: False for sec in sections}
    
    completed = 0
    for i, section in enumerate(sections):
        status = st.checkbox(
            f"{'✅' if st.session_state.section_status[section] else '⬜'} {section}", 
            value=st.session_state.section_status[section],
            key=f"sec_{i}"
        )
        st.session_state.section_status[section] = status
        if status:
            completed += 1
    
    # YOUR ORIGINAL PROGRESS BAR
    progress_value = completed / len(sections)
    st.progress(progress_value)
    st.markdown(f"**Progress: {int(progress_value*100)}%** ({completed}/{len(sections)})")
    
    st.divider()

    # 🔥 REVIEW BUTTON + ANSWER BELOW (REPLACE your Quick Actions section)
    st.subheader("🚀 **Quick Actions**")

    # REVIEW BUTTON
    if st.button("🔍 **Review Paper**", type="primary", use_container_width=True):
        st.session_state.show_review = True
        st.rerun()

    # ANSWER APPEARS BELOW BUTTON
    if st.session_state.get("show_review", False):
        with st.spinner("Generating paper review..."):
            if "review_result" not in st.session_state:
                try:
                    # GENERATE ACTUAL REVIEW
                    data_paths = generator.load_data_pdfs()
                    _, llm = generator.setup_rag(data_paths)
                    
                    paper_path = (generator.get_base_path() / generator.config["output"]["output_dir"] / 
                                f"{generator.config['output']['filename_prefix']}_{generator.config['title_suffix'].replace(' ', '_').lower()}.md")
                    paper_content = paper_path.read_text(encoding='utf-8')[:3000] if paper_path.exists() else "No paper generated"
                    
                    review_prompt = f"""
                    SCIENTIFIC PAPER REVIEW for Drug Safety Journal
                    
                    PAPER CONTENT: {paper_content}
                    
                    Provide structured review:
                    1️⃣ **Summary** (3 lines max)
                    2️⃣ **Writing improvements** (2 specific suggestions)
                    3️⃣ **Scientific review** (key gaps/critique)  
                    4️⃣ **PV Context** (GVP Module/MedDRA references)
                    
                    Professional, concise format.
                    """
                    
                    response = llm.complete(review_prompt).text.strip()
                    st.session_state.review_result = response
                    st.session_state.show_review = False
                    st.rerun()
                except Exception as e:
                    st.session_state.review_result = f"❌ Review error: {str(e)}"
                    st.session_state.show_review = False
                    st.rerun()

    # DISPLAY REVIEW BELOW BUTTON
    if "review_result" in st.session_state and st.session_state.review_result:
        st.markdown("---")
        st.markdown("**📝 PAPER REVIEW RESULTS:**")
        st.markdown(st.session_state.review_result)
        st.markdown("---")
        
        # NEW REVIEW BUTTON
        if st.button("🔄 **New Review**", type="secondary", use_container_width=True):
            st.session_state.review_result = None
            st.session_state.show_review = True
            st.rerun()

    st.divider()

    # === CHAT 1: PAPER RAG ===
    st.subheader("📄 **Paper RAG Chat**")
    st.caption("Ask about YOUR paper + PDFs")
    
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    rag_question = st.text_input("💬 Paper Q:", key="rag_chat", placeholder="Who is the author")
    if st.button("📄 Ask Paper", key="rag_send") and rag_question:
        st.session_state.rag_messages.append({"role": "user", "content": rag_question})
        st.rerun()
    
    # RAG Response
    if st.session_state.rag_messages and st.session_state.rag_messages[-1]["role"] == "user":
        with st.spinner("Searching paper + PDFs..."):
            question = st.session_state.rag_messages[-1]["content"]
            data_paths = generator.load_data_pdfs()
            _, llm = generator.setup_rag(data_paths)
            
            paper_path = (generator.get_base_path() / generator.config["output"]["output_dir"] / 
                         f"{generator.config['output']['filename_prefix']}_{generator.config['title_suffix'].replace(' ', '_').lower()}.md")
            paper_context = paper_path.read_text(encoding='utf-8')[:1500] if paper_path.exists() else ""
            
            rag_prompt = f"""
            PAPER RAG CHAT - Javier Alhambra
            
            PAPER: {paper_context}
            
            QUESTION: {question}
            Answer ONLY from paper content above.
            """
            response = llm.complete(rag_prompt).text.strip()
            st.session_state.rag_messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Show RAG chat (last 2)
    for msg in st.session_state.get("rag_messages", [])[-2:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    st.divider()
    
    # === CHAT 2: GENERAL PV ===
    st.subheader("🤖 **General PV Chat**")
    st.caption("GVP, MedDRA, regulations")
    
    if "general_messages" not in st.session_state:
        st.session_state.general_messages = []
    
    gen_question = st.text_input("💬 General:", key="gen_chat", placeholder="What is trazodone?")
    if st.button("🤖 Ask", key="gen_send") and gen_question:
        st.session_state.general_messages.append({"role": "user", "content": gen_question})
        st.rerun()
    
    # General response
    if st.session_state.general_messages and st.session_state.general_messages[-1]["role"] == "user":
        with st.spinner("Answering..."):
            question = st.session_state.general_messages[-1]["content"]
            _, llm = generator.setup_rag([])  # No RAG
            
            prompt = f"""
            PHARMACOVIGILANCE EXPERT
            
            Q: {question}
            
            Answer concisely as Drug Safety journal reviewer.
            Use: GVP Modules, MedDRA v27.0, EMA guidelines.
            """
            response = llm.complete(prompt).text.strip()
            st.session_state.general_messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Show general chat
    for msg in st.session_state.get("general_messages", [])[-2:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ========================================
# CENTER PANEL: Paper Editor
# ========================================
with col_center:
    st.header("📝 Paper Editor")
    
    # Load paper
    paper_path = (
        generator.get_base_path() / 
        generator.config["output"]["output_dir"] / 
        f"{generator.config['output']['filename_prefix']}_{generator.config['title_suffix'].replace(' ', '_').lower()}.md"
    )
    
    paper_content = ""
    if paper_path.exists():
        paper_content = paper_path.read_text(encoding='utf-8')
    
    # Preview
    st.markdown("**📄 Live Preview**")
    st.markdown("---")
    st.markdown(paper_content or "# No paper yet - click Generate!")
    st.markdown("---")
    
    # Editor
    st.markdown("**✏️ Edit**")
    edited_paper = st.text_area("", value=paper_content, height=300, key="editor")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save", type="primary"):
            paper_path.parent.mkdir(exist_ok=True)
            paper_path.write_text(edited_paper, encoding='utf-8')
            st.success("✅ Saved!")
            st.rerun()
    with col2:
        if st.button("🔄 Generate"):
            generator.generate_paper()
            st.rerun()

# ========================================
# RIGHT PANEL: Tasks + Assignee
# ========================================
with col_right:
    st.header("📋 Review Tasks")
    
    status_options = ["Draft", "In Review", "Reviewed"]
    current_status = st.selectbox("📊 Status", status_options)
    
    assignees = ["Javier Alhambra", "Marta Gómez", "Dr. López PV"]
    current_assignee = st.selectbox("👤 Assignee", assignees)
    
    st.checkbox("🔁 Needs Revision")
    
    st.divider()
    
    # Comments
    if "comments" not in st.session_state:
        st.session_state.comments = []
    
    comment = st.text_area("💬 Comment:", height=80)
    if st.button("Post") and comment.strip():
        st.session_state.comments.append({
            "text": comment.strip(),
            "author": current_assignee,
            "time": datetime.now().strftime("%H:%M")
        })
        st.rerun()
    
    for c in st.session_state.comments[-3:]:
        with st.expander(f"{c['author']} - {c['time']}"):
            st.markdown(c["text"])
