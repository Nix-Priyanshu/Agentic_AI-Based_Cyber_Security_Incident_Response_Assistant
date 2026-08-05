#Step 1 Imports
import streamlit as st
import os

from backend import process_incident

#Step 2 Page Config
st.set_page_config(
    page_title="Cyber Incident Response Assistant",
    layout="wide"
)

st.title("🛡️ Agentic AI Cyber Security Incident Response Assistant")

st.write("""
Analyze cyber security incidents using
RAG + AI Incident Response Agent.
""")


#Step 3 Sidebar
st.sidebar.title("Configuration")
#Provider Selection
provider = st.sidebar.selectbox(
    "Choose LLM",
    [
        "Gemini",
        "Groq"
    ]
)
#API Key
api_key = st.sidebar.text_input(
    "API Key",
    type="password"
)
#Knowledge Base Upload
uploaded_files = st.sidebar.file_uploader(
    "Upload Knowledge Base",
    type=["pdf","txt"],
    accept_multiple_files=True
)



#Step 4 Save Files
kb_folder = "knowledge_base"

os.makedirs(
    kb_folder,
    exist_ok=True
)
#Save uploaded files.
if uploaded_files:

    for file in uploaded_files:

        with open(
            os.path.join(kb_folder,file.name),
            "wb"
        ) as f:

            f.write(file.getbuffer())

    st.sidebar.success("Knowledge Base Loaded")


#Step 5 Incident Input
incident = st.text_area(
    "Describe the Cyber Security Incident"
)


# ============================
# Step 6 + Step 7 Analyze Incident
# ============================

if st.button("Analyze Incident"):

    # Validation

    if not api_key:
        st.error("Enter API Key")
        st.stop()


    if not uploaded_files:
        st.error("Upload Knowledge Base")
        st.stop()


    if not incident.strip():
        st.error("Enter Incident Description")
        st.stop()



    # Running Agent Pipeline

    with st.spinner("Analyzing Incident..."):

        report = process_incident(
            incident=incident,
            provider=provider,
            api_key=api_key,
            kb_folder=kb_folder
        )


    # Display Result

    st.success("Analysis Completed Successfully!")


    st.markdown(
        "## 🛡️ Incident Response Report"
    )

    st.markdown(report)



    # Download Report

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="incident_response_report.md",
        mime="text/markdown"
    )
