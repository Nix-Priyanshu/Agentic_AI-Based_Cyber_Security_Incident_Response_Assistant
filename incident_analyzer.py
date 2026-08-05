from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def analyze_incident(
    incident,
    context,
    llm
):

    prompt = ChatPromptTemplate.from_template("""
You are an expert Cyber Security Incident Response Analyst.

Your task is to analyze the cyber incident using the provided knowledge base context.

Knowledge Base Context:
{context}

Incident:
{incident}

Perform the following:

1. Identify the type of attack.
2. Explain how the attack works.
3. Determine severity (Low/Medium/High/Critical).
4. List Indicators of Compromise (IOCs).
5. Explain possible business impact.
6. Recommend immediate containment actions.
7. Recommend eradication steps.
8. Recommend recovery actions.
9. Suggest future prevention measures.

Provide a detailed technical analysis.
""")

    parser = StrOutputParser()

    chain = (
        prompt
        | llm
        | parser
    )

    analysis = chain.invoke(
        {
            "incident": incident,
            "context": context
        }
    )

    return analysis
