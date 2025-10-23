# utils/doc_loader.py
from docx import Document

def load_resume(docx_path: str) -> str:
    """Extract text from a .docx file."""
    doc = Document(docx_path)
    text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
    return text
