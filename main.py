# main.py
from rag_pipeline import answer_query

if __name__ == "__main__":
    resume_path = "data/resume.docx"   # path to your resume
    query = "What are Akeel's main MLOps skills?"
    response = answer_query(resume_path, query)
    print("\n=== RAG Answer ===\n")
    print(response)
