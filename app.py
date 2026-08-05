#========LOAD MODULES====================
# All imports needed for the whole app are kept here at the top,
# same as a normal single-file student project.

import os
import tempfile
from datetime import datetime

import streamlit as st

# ---- Modern LangChain imports (no deprecated APIs) ----
# NOTE: We do NOT use RetrievalQA / LLMChain / ConversationChain
# (those are the old, deprecated way). Instead we build the RAG
# pipeline using the newer "chain composition" style:
#   create_stuff_documents_chain  -> stuffs retrieved chunks into the prompt
#   create_retrieval_chain        -> connects the retriever to that chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# LLM + Embeddings providers
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# ReportLab for PDF report generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors


#========PAGE CONFIG====================
st.set_page_config(
    page_title="CyberGuardian AI",
    page_icon="🛡️",
    layout="wide",
)


#========SESSION STATE (APP MEMORY)====================
# Streamlit reruns the whole script on every click, so we use
# session_state to "remember" things across reruns.

defaults = {
    "vector_store": None,       # FAISS vector store built from the uploaded doc
    "retriever": None,          # retriever built from the vector store
    "analysis_result": None,    # parsed dict of the incident analysis
    "raw_analysis_text": None,  # raw LLM output (used for the PDF report)
    "qa_history": [],           # list of (question, answer) tuples
    "uploaded_file_name": None, # name of the currently processed file
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


#========SIDEBAR====================
with st.sidebar:

    # ---- Logo + Project Name ----
    st.markdown("## 🛡️ CyberGuardian AI")
    st.caption("Agentic AI Cyber Security Incident Response Assistant")

    st.divider()

    # ---- LLM Provider Selection ----
    st.subheader("⚙️ LLM Settings")

    llm_provider = st.selectbox(
        "Choose your LLM Provider",
        options=["Gemini", "Groq"],
        help="Select which AI provider you want to use for generating answers.",
    )

    # ---- API Key Input(s) ----
    # IMPORTANT: No API key is ever hardcoded. Keys are typed in by
    # the user and only live in this session's memory.
    #
    # NOTE ON EMBEDDINGS: Groq does not provide an embeddings API,
    # so even when "Groq" is selected for chat answers, we still
    # need a Google Gemini API key to create the document embeddings
    # used for search (FAISS). This is explained to the user below.

    if llm_provider == "Gemini":
        gemini_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Paste your Gemini API key here",
            help="Used for both chat answers and document embeddings.",
        )
        groq_key_input = None
        embedding_key = gemini_key_input
        chat_key = gemini_key_input

    else:  # Groq selected
        groq_key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Paste your Groq API key here",
            help="Used for generating chat answers.",
        )
        st.info(
            "ℹ️ Groq doesn't offer an embeddings API. A Google Gemini "
            "key is also needed just to search your document."
        )
        gemini_key_input = st.text_input(
            "Google Gemini API Key (for embeddings)",
            type="password",
            placeholder="Paste your Gemini API key here",
        )
        embedding_key = gemini_key_input
        chat_key = groq_key_input

    if chat_key and embedding_key:
        st.success("API Key(s) received ✅")
    else:
        st.warning("Please enter the required API key(s) to use the app.")

    st.divider()

    # ---- Advanced RAG Settings (extra sidebar feature) ----
    with st.expander("🛠️ Advanced RAG Settings"):
        model_name = st.text_input(
            "Model Name",
            value="gemini-1.5-flash" if llm_provider == "Gemini" else "llama-3.1-8b-instant",
            help="The exact model ID sent to the provider. Change if you know a different model name.",
        )
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=150, step=50)
        top_k = st.slider("Chunks to Retrieve (top-k)", min_value=2, max_value=10, value=4, step=1)

    st.divider()

    # ---- Reset Session (extra sidebar feature) ----
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()

    st.divider()

    # ---- About Section ----
    with st.expander("ℹ️ About This Project"):
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


#========RAG PIPELINE FUNCTIONS====================
# Document -> Chunking -> Embeddings -> FAISS -> Retriever

def load_document(uploaded_file):
    """
    Saves the uploaded file to a temporary location and loads it
    using the correct LangChain loader based on file type
    (PDF or TXT). Returns a list of LangChain Document objects.
    """
    file_suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    if file_suffix.lower() == ".pdf":
        loader = PyPDFLoader(tmp_file_path)
    else:
        loader = TextLoader(tmp_file_path, encoding="utf-8")

    documents = loader.load()
    os.remove(tmp_file_path)
    return documents


def split_documents(documents, chunk_size, chunk_overlap):
    """
    Splits the loaded document(s) into smaller overlapping chunks
    so the retriever can find relevant sections more precisely.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def build_vector_store(chunks, embedding_key):
    """
    Converts text chunks into embeddings and stores them in a
    FAISS vector store for fast similarity search.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=embedding_key,
    )
    return FAISS.from_documents(chunks, embeddings)


def get_retriever(vector_store, top_k):
    """
    Returns a retriever from the FAISS vector store that fetches
    the top-k most relevant chunks for any query.
    """
    return vector_store.as_retriever(search_kwargs={"k": top_k})


#========LLM + MODERN RAG CHAIN FUNCTIONS====================

def get_llm(provider, api_key, model_name):
    """
    Creates and returns a chat LLM object based on the chosen provider.
    """
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )
    else:
        return ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.3,
        )


# System prompt shared by both the full-incident analysis and
# custom Q&A, so answers stay grounded in the uploaded document.
RAG_SYSTEM_PROMPT = (
    "You are CyberGuardian AI, a helpful cyber security incident response "
    "assistant. Answer ONLY using the context provided below, which comes "
    "from the user's uploaded security report/log. If the answer isn't "
    "in the context, say you don't have enough information in the document. "
    "Be clear, concise, and use plain language.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)


def get_rag_answer(llm, retriever, question):
    """
    Runs the modern LangChain RAG pipeline:
      retriever -> create_stuff_documents_chain -> create_retrieval_chain
    and returns the plain text answer.

    This replaces the older/deprecated RetrievalQA approach.
    """
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]


# The fixed question we send for the full incident analysis.
# We ask the model to reply using clear "### Header" markers so
# we can easily split the response into result cards afterwards.
ANALYSIS_QUESTION = """
Analyze the uploaded security report/log and respond STRICTLY in the
following format. Use these exact section headers, each on its own
line starting with '### ', with nothing else outside this format:

### Incident Summary
(A short summary of what happened)

### Attack Type
(The type of attack, e.g. Phishing, DDoS, Malware, Brute Force, etc.)

### Severity
(Low, Medium, High, or Critical - with a one-line reason)

### Root Cause
(What likely caused this incident)

### Explanation
(A simple explanation a non-expert could understand)

### Mitigation
(Immediate steps to contain/fix the issue)

### Best Practices
(Best practices to prevent this in the future)

### Disclaimer
(A short note that this is an AI-generated analysis and should be
reviewed by a qualified security professional before taking action)
"""


def parse_analysis_output(text):
    """
    Splits the LLM's '### Header' formatted response into a
    dictionary so we can display each part as its own result card.
    """
    sections = {}
    parts = text.split("### ")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        header = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        sections[header] = content
    return sections


#========MAIN PAGE: TITLE + DESCRIPTION====================
st.title("🛡️ CyberGuardian AI")
st.write(
    "An **Agentic AI based Cyber Security Incident Response Assistant** "
    "that reads your uploaded security reports/logs and helps you understand "
    "and respond to incidents faster."
)

st.divider()

#========MAIN PAGE: UPLOAD SECTION====================
st.subheader("📂 Upload Security Report / Log File")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file containing the security report or log",
    type=["pdf", "txt"],
)

if uploaded_file is not None:
    st.info(f"File selected: **{uploaded_file.name}**")

analyze_clicked = st.button("🔍 Analyze Incident", type="primary", use_container_width=True)

#========MAIN PAGE: ANALYZE BUTTON LOGIC====================
if analyze_clicked:
    if uploaded_file is None:
        st.error("Please upload a PDF or TXT file first.")
    elif not chat_key or not embedding_key:
        st.error("Please enter the required API key(s) in the sidebar first.")
    else:
        with st.spinner("Reading document and building the search index..."):
            documents = load_document(uploaded_file)
            chunks = split_documents(documents, chunk_size, chunk_overlap)
            vector_store = build_vector_store(chunks, embedding_key)
            retriever = get_retriever(vector_store, top_k)

            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.uploaded_file_name = uploaded_file.name

        with st.spinner("Analyzing the incident with the LLM..."):
            llm = get_llm(llm_provider, chat_key, model_name)
            raw_answer = get_rag_answer(llm, retriever, ANALYSIS_QUESTION)

            st.session_state.raw_analysis_text = raw_answer
            st.session_state.analysis_result = parse_analysis_output(raw_answer)

        st.success("Analysis complete! Scroll down to see the results. ⬇️")


#========MAIN PAGE: RESULTS CARDS====================
if st.session_state.analysis_result:
    st.divider()
    st.subheader("📊 Incident Analysis Results")

    result = st.session_state.analysis_result

    # Icons for each section, purely for visual polish
    section_icons = {
        "Incident Summary": "📝",
        "Attack Type": "🎯",
        "Severity": "🚨",
        "Root Cause": "🔍",
        "Explanation": "💡",
        "Mitigation": "🛠️",
        "Best Practices": "✅",
        "Disclaimer": "⚠️",
    }

    # Display cards in two columns for a clean layout
    section_names = list(section_icons.keys())
    col1, col2 = st.columns(2)

    for i, section in enumerate(section_names):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            with st.container(border=True):
                icon = section_icons.get(section, "📌")
                st.markdown(f"**{icon} {section}**")
                st.write(result.get(section, "_Not available in the analysis._"))
