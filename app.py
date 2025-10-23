# app.py
import streamlit as st
from rag_pipeline import answer_query
import pandas as pd
import os
import time
from dotenv import load_dotenv
from langfuse import get_client

# ================== CONFIG ==================
st.set_page_config(page_title="Resume RAG Assistant", page_icon="💼", layout="wide")
st.title("💼 Resume RAG Assistant (Phi-3 + Reranker)")
st.caption("Ask any question about your resume. Context-aware answers powered by LM Studio Phi-3.")

# Load environment variables (expects LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY set)
load_dotenv()

# Initialize Langfuse client (singleton)
lf_client = get_client()

# Fixed resume path
RESUME_PATH = "data/resume.docx"

# ================== DOWNLOAD RESUME BUTTON ==================
if os.path.exists(RESUME_PATH):
    with open(RESUME_PATH, "rb") as f:
        st.download_button(
            label="📄 Download Resume",
            data=f,
            file_name="Akeel_Mohammad_Resume.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
else:
    st.warning(f"Resume not found at {RESUME_PATH}")

# ================== FEEDBACK LOGGER ==================
def log_feedback(feedback, question, answer):
    """Save user feedback locally and send to Langfuse using a span update."""
    feedback_entry = pd.DataFrame([{
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "feedback": feedback
    }])

    feedback_file = "feedback_log.csv"
    if not os.path.exists(feedback_file):
        feedback_entry.to_csv(feedback_file, index=False)
    else:
        feedback_entry.to_csv(feedback_file, mode='a', header=False, index=False)

    st.success(f"✅ Feedback '{feedback}' recorded for this answer.")

    # Log feedback to Langfuse by creating a short span and updating trace metadata
    try:
        # create a short span to attach feedback to a trace in Langfuse
        with lf_client.start_as_current_span(name="resume-feedback", input=question) as fb_span:
            # attach feedback as trace-level metadata / output
            fb_span.update_trace(
                output=answer,
                metadata={"feedback": feedback, "resume": "Akeel_Mohammad"},
            )
    except Exception as e:
        st.warning(f"Langfuse logging failed (feedback): {e}")

# ================== SESSION STATE ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================== CHAT DISPLAY ==================
chat_container = st.container()
with chat_container:
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(
                f"<div style='background-color:#DCF8C6;padding:10px;border-radius:10px;"
                f"margin-bottom:5px;text-align:right;'>🧑‍💻 <b>You:</b> {msg['content']}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='background-color:#F1F0F0;padding:10px;border-radius:10px;"
                f"margin-bottom:5px;text-align:left;'>🤖 <b>Bot:</b> {msg['content']}</div>",
                unsafe_allow_html=True)

            # Feedback buttons
            col1, col2 = st.columns([1, 1])
            if col1.button("👍", key=f"up_{idx}"):
                # the previous message in chat_history is the user question
                prev_q = st.session_state.chat_history[idx-1]["content"] if idx-1 >= 0 else ""
                log_feedback("up", prev_q, msg["content"])
            if col2.button("👎", key=f"down_{idx}"):
                prev_q = st.session_state.chat_history[idx-1]["content"] if idx-1 >= 0 else ""
                log_feedback("down", prev_q, msg["content"])

# ================== CHAT INPUT ==================
st.markdown("---")

with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input(
        "💬 Ask your question here:",
        placeholder="e.g., What are my top MLOps skills?",
        key="query_input"
    )
    submitted = st.form_submit_button("Send")

if submitted and query.strip():
    # Append user query once
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Use Langfuse context managers to create a trace + generation
    answer = "[Error generating answer]"
    try:
        # create a trace/span for this user query
        with lf_client.start_as_current_span(name="resume-query", input=query) as span:
            # Inside this trace, create a generation span for the model output.
            # We call the RAG pipeline while the generation span is open so the SDK can capture timing.
            with span.start_as_current_generation(name="phi3-answer", model="phi-3", input=query) as gen:
                # Generate answer from RAG (this runs LM Studio call inside the generation span)
                try:
                    answer = answer_query(RESUME_PATH, query)
                except Exception as e:
                    answer = f"[Error while processing query: {e}]"

                # update generation span with the model output (this records output & metadata)
                try:
                    gen.update_trace(
                        output=answer,
                        metadata={"resume": "Akeel_Mohammad"}
                    )
                except Exception as e:
                    # non-fatal: warn but continue
                    st.warning(f"Langfuse generation update failed: {e}")

            # Optionally update root span output/metadata too
            try:
                span.update_trace(output=answer, metadata={"resume": "Akeel_Mohammad"})
            except Exception:
                pass

    except Exception as e:
        # If Langfuse client fails, still proceed with the answer
        st.warning(f"Langfuse logging failed (query): {e}")

    # Append model answer to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Refresh chat to display answer immediately
    st.rerun()
