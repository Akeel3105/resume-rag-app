# utils/retriever.py
from sentence_transformers import CrossEncoder

def retrieve_and_rerank(query, chunks, index, embedder, top_k=5):
    """Retrieve top chunks and rerank them using a cross-encoder."""
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k * 3)
    retrieved_chunks = [chunks[i] for i in I[0]]

    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(retrieved_chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [r[0] for r in ranked[:top_k]]
    return top_chunks
