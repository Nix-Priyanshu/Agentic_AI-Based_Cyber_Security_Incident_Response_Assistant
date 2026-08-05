from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def generate_report(
    incident,
    analysis,
    llm
):

    prompt = ChatPromptTemplate.from_template("""
You are an expert Cyber Security Incident Response Report Writer.

Incident:
{incident}

Technical Analysis:
{analysis}

Create a professional incident response report.

The report must contain:

# Executive Summary

# Incident Overview

# Attack Analysis

# Severity Assessment

# Indicators of Compromise (IOCs)

# Business Impact

# Containment Actions

# Eradication Steps

# Recovery Plan

# Recommendations

# Lessons Learned

Format everything professionally using markdown.
""")

    parser = StrOutputParser()

    chain = (
        prompt
        | llm
        | parser
    )

    report = chain.invoke(
        {
            "incident": incident,
            "analysis": analysis
        }
    )

    return report
