# rag_pipeline.py
from utils.doc_loader import load_resume
from utils.chunker import chunk_text
from utils.embedder import build_faiss_index
from utils.retriever import retrieve_and_rerank
from utils.llm_phi3 import generate_answer

def answer_query(resume_path: str, query: str):
    """Main orchestration function for the RAG pipeline."""
    text = load_resume(resume_path)
    chunks = chunk_text(text)
    index, embedder, cached_chunks = build_faiss_index(chunks, resume_path)
    top_chunks = retrieve_and_rerank(query, cached_chunks, index, embedder)
    answer = generate_answer(query, top_chunks)
    return answer
