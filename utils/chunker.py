# utils/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size=500, chunk_overlap=100):
    """Split long text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
