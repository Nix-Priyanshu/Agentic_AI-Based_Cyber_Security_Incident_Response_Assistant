"""
CyberGuardian AI - An Agentic AI Based Cyber Security Incident Response Assistant
Capstone Project - Generative AI + Agentic AI Training

This app lets a user upload a security report/log (PDF or TXT),
runs a simple RAG (Retrieval Augmented Generation) pipeline over it,
and uses an LLM (Gemini or Groq) to analyze the incident and answer
custom questions. A downloadable PDF report is generated at the end.
"""

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import os
import tempfile
from datetime import datetime

import streamlit as st

# LangChain imports (kept simple, as per project scope)
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# LLM + Embeddings providers
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# ReportLab for PDF report generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="CyberGuardian AI",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
# We use session_state to store things across reruns:
# - the vector store / retriever built from the uploaded document
# - the analysis results (so they persist after button clicks)
# - chat-style Q&A history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    # Simple text-based "logo" (emoji) to keep things beginner-friendly
    st.markdown("## 🛡️ CyberGuardian AI")
    st.caption("Agentic AI Cyber Security Incident Response Assistant")

    st.divider()

    # --- LLM Provider Selection ---
    st.subheader("⚙️ LLM Settings")

    llm_provider = st.selectbox(
        "Choose your LLM Provider",
        options=["Gemini", "Groq"],
        help="Select which AI provider you want to use for analysis.",
    )

    # --- API Key Input ---
    # IMPORTANT: The key is never hardcoded. It is only stored in
    # session memory for this browser session and used at runtime.
    api_key = st.text_input(
        f"Enter your {llm_provider} API Key",
        type="password",
        placeholder="Paste your API key here",
    )

    if api_key:
        st.success("API Key received ✅")
    else:
        st.warning("Please enter an API key to use the app.")

    st.divider()

    # --- About Section ---
    st.subheader("ℹ️ About This Project")
    st.write(
        """
        **CyberGuardian AI** is a capstone project built using
        Generative AI + Agentic AI concepts.

        It analyzes uploaded security reports/logs using a simple
        RAG (Retrieval Augmented Generation) pipeline and generates:
        - Incident Summary
        - Attack Type
        - Severity
        - Root Cause
        - Explanation
        - Mitigation Steps
        - Best Practices

        Built with: Python, Streamlit, LangChain, FAISS,
        Google Gemini API, Groq API, and ReportLab.
        """
    )

    st.caption("🎓 Final Capstone Project | Generative AI + Agentic AI Training")
