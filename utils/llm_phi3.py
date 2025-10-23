# utils/llm_phi3.py
import requests
import json

def generate_answer(query: str, context_chunks):
    """Send prompt to Phi-3 model via LM Studio's REST API."""
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant that answers questions based on the resume below.

Resume Context:
{context}

Question: {query}

Answer clearly and concisely:
"""

    url = "http://localhost:1234/v1/chat/completions"  # LM Studio API endpoint
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "phi-3",  # Adjust if your LM Studio model name differs
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 400
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[Error calling Phi-3 via LM Studio: {e}]"
