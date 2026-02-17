import streamlit as st
from rag_llamaindex_qdrant5 import PaperGenerator
import os

# 🎨 LOAD CSS
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="🧪 PV Paper Assistant")
generator = PaperGenerator("config.yaml")

st.title("🧪 Pharmacovigilance Paper Assistant")
st.markdown("**Javier Alhambra** - Valencia Pharma - Drug Safety Ready")

# === TWO PANE LAYOUT ===
col_left, col_right = st.columns([2, 1])

# LEFT: PAPER PREVIEW + EDITOR
with col_left:
    st.header("📄 Paper Editor")
    
    try:
        paper_path = (
            generator.get_base_path()
            / generator.config["output"]["output_dir"]
            / f"{generator.config['output']['filename_prefix']}_{generator.config['title_suffix'].replace(' ', '_').lower()}.md"
        )
        
        paper_content = ""
        if paper_path.exists():
            paper_content = paper_path.read_text(encoding='utf-8')
        
        # 1. PERFECT PAPER PREVIEW (TOP)
        st.markdown("**📄 Live Preview**")
        st.markdown("---")
        st.markdown(paper_content)  # Perfect MD rendering
        st.markdown("---")
        
        # 2. YOUR CURRENT WHITE EDITOR (BOTTOM)
        st.markdown("**✏️ Edit Paper**")
        edited_paper = st.text_area(
            "Edit entire paper directly:",
            value=paper_content,
            height=400,  # Smaller since preview above
            key="paper_editor"
        )
        
        col_save, col_regen = st.columns(2)
        with col_save:
            if st.button("💾 Save Paper", type="primary"):
                paper_path.write_text(edited_paper, encoding='utf-8')
                st.success("✅ Paper saved!")
                st.rerun()
        
        with col_regen:
            if st.button("🔄 Regenerate Paper"):
                generator.generate_paper()
                st.rerun()
                
    except Exception as e:
        st.error(f"Error: {e}")


# RIGHT: SUGGESTIONS + QUESTIONS (YOUR EXACT CODE)
with col_right:
    st.header("🤖 Suggestions & Chat")
    
    # Initialize chat
    if "right_messages" not in st.session_state:
        st.session_state.right_messages = []
    
    # QUICK SUGGESTIONS
    st.markdown("**🚀 Quick Actions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✨ Improve Introduction"):
            st.session_state.right_messages.append({"role": "user", "content": "Improve I. INTRODUCTION"})
            st.rerun()
    
    with col2:
        if st.button("🔍 Scientific Review"):
            st.session_state.right_messages.append({"role": "user", "content": "Scientific review entire paper"})
            st.rerun()
    
    with col3:
        if st.button("📊 Add PV Context"):
            st.session_state.right_messages.append({"role": "user", "content": "Add GVP/MedDRA context"})
            st.rerun()
    
    # CHAT INPUT
    question = st.text_input("💬 Ask about paper / suggestions:", key="right_chat")
    if st.button("Send") or question:
        if question:
            st.session_state.right_messages.append({"role": "user", "content": question})
            st.rerun()
    
    # PROCESS RESPONSES
    if st.session_state.right_messages and st.session_state.right_messages[-1]["role"] == "user":
        prompt = st.session_state.right_messages[-1]["content"]
        
        data_paths = generator.load_data_pdfs()
        _, llm = generator.setup_rag(data_paths)
        
        full_prompt = f"""
        DRUG SAFETY PAPER by Javier Alhambra (Valencia)
        USER REQUEST: {prompt}
        
        Respond with:
        - Writing improvements → ONLY improved text (no headers)
        - Reviews → Scientific critique  
        - PV context → GVP/MedDRA/EMA references
        """
        
        response = llm.complete(full_prompt).text.strip()
        st.session_state.right_messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # SHOW SUGGESTIONS/CHAT
    st.markdown("**📝 Recent Suggestions:**")
    for message in st.session_state.right_messages[-4:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                st.markdown("**🔧 Actions:**")
                col_copy, col_apply = st.columns(2)
                with col_copy:
                    st.button("📋 Copy", key=f"copy_{len(st.session_state.right_messages)}")
                with col_apply:
                    if st.button("💾 Insert Suggestion", key=f"apply_{len(st.session_state.right_messages)}"):
                        generator._smart_insert("I. INTRODUCTION", message["content"])
                        st.success("✅ Inserted into paper!")
                        st.rerun()
