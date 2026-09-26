"""
HR Agent: Deep Resume Analyzer
Gives detailed analysis + can answer questions about a resume.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


ANALYZE_PROMPT = """You are a Senior Resume Analyzer Agent at OmniDesk AI.

You will receive a resume (and optionally a job description).

Produce a DETAILED analysis in markdown with these sections:

## Candidate Snapshot
- Name, current role, years of experience, education

## Skills Breakdown
- **Core strengths** (list)
- **Tools & technologies** (list)
- **Missing/weak areas** (list)

## Strengths
- 3-5 bullet points highlighting the candidate's top strengths

## Weaknesses / Gaps
- 3-5 bullet points listing concerns or gaps

## ATS Readability
- Score out of 100 and a one-line comment

## Interview Focus Areas
- 4-5 specific topics to probe in an interview

## Recommendations
- Should we shortlist? Yes/No/Maybe + one-line reason

Rules:
- Be specific — cite evidence from the resume.
- Use markdown.
- Be honest but professional.
- If a JD is provided, include a **JD Fit** section comparing must-have vs have skills.
"""


QA_PROMPT = """You are a Resume Q&A Assistant at OmniDesk AI.

You have the resume below as context. Answer the user's question about this resume 
concisely and specifically. Cite evidence from the resume. Use markdown.

If the question is unrelated to the resume, politely redirect.
"""


def _extract(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                t = b.get("text", "")
                if t:
                    parts.append(t)
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return str(content)


def analyze_resume(resume_text: str, api_key: str, jd_text: str = "") -> str:
    """Full structured analysis of a single resume."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        google_api_key=api_key,
    )

    body = f"RESUME:\n{resume_text[:8000]}"
    if jd_text.strip():
        body += f"\n\nJOB DESCRIPTION:\n{jd_text[:3000]}"

    response = llm.invoke([
        SystemMessage(content=ANALYZE_PROMPT),
        HumanMessage(content=body),
    ])
    return _extract(response.content)


def ask_resume_question(resume_text: str, question: str, chat_history: list, api_key: str) -> str:
    """Q&A over a single resume."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        google_api_key=api_key,
    )

    messages = [
        SystemMessage(content=QA_PROMPT),
        SystemMessage(content=f"RESUME CONTEXT:\n{resume_text[:8000]}"),
    ]
    for m in chat_history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return _extract(response.content)