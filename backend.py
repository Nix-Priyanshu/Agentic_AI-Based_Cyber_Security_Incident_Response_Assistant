import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from rag_engine import (
    load_documents,
    split_documents,
    load_embeddings,
    create_vector_store,
    create_retriever
)

from incident_analyzer import analyze_incident
from report_generator import generate_report

#Step 2 Create Model
def load_llm(provider, api_key):
  if provider == "Gemini":

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

  else:
  
      return ChatGroq(
          model="llama-3.3-70b-versatile",
          api_key=api_key
      )



#Step 3 Knowledge Base
def build_knowledge_base(folder):

    # Load all documents
    documents = load_documents(folder)

    # Split documents
    chunks = split_documents(documents)

    # Embeddings
    embeddings = load_embeddings()

    # Vector DB
    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    # Retriever
    retriever = create_retriever(
        vector_store,
        k=4
    )

    return retriever




#Step 4 Incident Pipeline
def process_incident(
        incident,
        provider,
        api_key,
        kb_folder
):
  # Load LLM
  llm = load_llm(provider, api_key)

  # Build Knowledge Base
  retriever = build_knowledge_base(kb_folder)

  # Retrieve relevant context
  docs = retriever.invoke(incident)

  context = "\n\n".join(
      doc.page_content for doc in docs
  )

  # Analyze incident
  analysis = analyze_incident(
      incident,
      context,
      llm
  )

  # Generate report
  report = generate_report(
      incident,
      analysis,
      llm
  )

  return report
