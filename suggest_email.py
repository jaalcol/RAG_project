def suggest_email(self, recipient, context="reviewer_comment"):
    """Context-aware email suggestions"""
    author = self.config.get("author_name")
    
    templates = {
        "reviewer_comment": f"""
        Subject: Revisions for "Ontology Framework 2026" - {author}
        
        Dear [Reviewer],
        
        Thank you for your feedback on our pharmacovigilance ontology extraction paper.
        
        [SPECIFIC FIXES BASED ON RAG ANALYSIS]
        
        Revised manuscript attached.
        
        Best regards,
        Javier Alhambra
        """,
        "coauthor_update": f"Quick progress update on Methods section..."
    }
    
    email = llm.complete(f"Polish this email draft for {recipient}: {templates[context]}").text
    print(f"\n📧 EMAIL DRAFT:\n{email}")
