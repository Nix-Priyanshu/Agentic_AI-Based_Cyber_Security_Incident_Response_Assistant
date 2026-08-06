"""
prompt_templates.py

This module defines the prompt templates used by the Agentic AI Cyber Security 
Incident Response Assistant. These templates guide the LLM in performing 
specialized tasks such as incident analysis, report generation, RAG-based 
question answering, and IOC extraction.
"""

SYSTEM_PROMPT = """You are an expert Cyber Security Incident Response Assistant. Your role is to assist security analysts, incident responders, and security operations center (SOC) teams in analyzing, triaging, containing, and recovering from security incidents. You provide precise, actionable, and highly technical guidance following industry standards such as NIST SP 800-61 and the SANS Incident Handler's Handbook. Maintain a professional, analytical, and objective tone at all times."""

INCIDENT_ANALYSIS_PROMPT = """You are analyzing a security incident. Based on the provided Incident Description, perform a comprehensive analysis and output your findings in the exact structure defined below.

Input Incident Description:
{incident_description}

Output Structure:
1. Executive Summary: A high-level overview of the incident, its impact, and current status.
2. Attack Type: Identify the specific category of attack (e.g., Ransomware, Phishing, DDoS, Insider Threat).
3. Severity: Determine the severity level (Low, Medium, High, Critical) with justification.
4. Affected Assets: List the systems, networks, users, or data compromised or targeted.
5. Indicators of Compromise (IOCs): Extract or infer any IOCs (IPs, domains, hashes, etc.).
6. MITRE ATT&CK Mapping: Map the observed adversary behaviors to MITRE ATT&CK tactics and techniques.
7. Containment Strategy: Immediate steps required to stop the incident from spreading.
8. Eradication Strategy: Steps to completely remove the threat from the environment.
9. Recovery Strategy: Steps to restore affected systems and services to normal operations safely.
10. Recommendations: Long-term preventative measures to avoid recurrence."""

REPORT_PROMPT = """Generate a professional, executive-ready Incident Report based on the provided incident analysis and details. The report should be structured formally, suitable for presentation to C-level executives and external stakeholders.

Incident Details and Analysis:
{incident_details}

The report must include:
- Document Control (Date, Author, Version, Classification)
- Executive Summary (Non-technical overview of the event, business impact, and resolution)
- Incident Timeline (Chronological sequence of events from detection to resolution)
- Technical Analysis (Detailed breakdown of the attack vector, tools used, and scope)
- Impact Assessment (Financial, operational, reputational, and regulatory impact)
- Containment, Eradication, and Recovery Actions Taken
- Lessons Learned and Strategic Recommendations (Actionable steps to improve security posture)

Ensure the tone is formal, objective, and authoritative."""

RAG_PROMPT = """You are a security assistant answering questions based strictly on the provided context. 

Context:
{context}

Question:
{question}

Instructions:
- Answer the question ONLY using the retrieved context above.
- Do not use any external knowledge or make assumptions.
- If the answer is not explicitly available in the context, you must say exactly: 'I don't know based on the knowledge base.'"""

IOC_PROMPT = """Analyze the provided text and extract all Indicators of Compromise (IOCs). Organize the extracted IOCs into the following specific categories. If no IOCs are found for a category, write 'None'.

Text to Analyze:
{text}

Extracted IOCs:
- IP Addresses:
- Domains:
- URLs:
- Hashes (MD5, SHA-1, SHA-256):
- Email Addresses:
- File Names:"""