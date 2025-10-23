# evaluate_rag.py

from rag_pipeline import answer_query
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm

# ======= CONFIG =======
RESUME_PATH = "data/resume.docx"

# Ground-truth test questions and answers
test_data = [
    {
        "question": "What is the candidate's full name?",
        "expected_answer": "Akeel Mohammad"
    },
    {
        "question": "List main MLOps skills of the candidate.",
        "expected_answer": "CI/CD, Docker, Kubernetes, MLflow, Airflow, and model monitoring"
    },
    {
        "question": "What company or organization was this resume prepared for?",
        "expected_answer": "NSEIT"
    },
    {
        "question": "What is the candidate’s total experience?",
        "expected_answer": "Around 3+ years in MLOps and Data Engineering"
    }
]

# ======= EVALUATION =======
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

results = []

print("Evaluating RAG responses...\n")

for item in tqdm(test_data):
    question = item["question"]
    expected = item["expected_answer"]

    try:
        predicted = answer_query(RESUME_PATH, question)
    except Exception as e:
        predicted = f"[Error: {e}]"

    # Compute semantic similarity
    emb_expected = model.encode(expected, convert_to_tensor=True)
    emb_pred = model.encode(predicted, convert_to_tensor=True)
    similarity = util.cos_sim(emb_expected, emb_pred).item()

    results.append({
        "Question": question,
        "Expected": expected,
        "Predicted": predicted,
        "Similarity": round(similarity, 3)
    })

# ======= RESULTS =======
df = pd.DataFrame(results)
print(df)
print(f"\nAverage Similarity Score: {df['Similarity'].mean():.3f}")

# Optionally save results
df.to_csv("rag_evaluation_results.csv", index=False)
print("\nResults saved to rag_evaluation_results.csv")
