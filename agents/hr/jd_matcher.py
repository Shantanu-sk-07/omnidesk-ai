"""
HR Agent 2: JD Matcher
Compares a parsed resume against a job description.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


MATCHER_PROMPT = """You are a Job-Description Matcher Agent.

Given a JOB DESCRIPTION and a CANDIDATE PROFILE, evaluate how well the candidate fits.

Reply in this exact format:

MATCH_SCORE: <number between 0 and 100>
MATCHED_SKILLS: <comma-separated skills found in both>
MISSING_SKILLS: <comma-separated required skills the candidate lacks>
REASONING: <2-3 sentences explaining the score>

Be strict but fair. Score 90+ only for near-perfect matches.
"""


def match_resume_to_jd(jd: str, parsed_resume: dict, api_key: str) -> dict:
    """Returns dict with score, matched, missing, reasoning."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )

    candidate_block = (
        f"Name: {parsed_resume.get('name')}\n"
        f"Role: {parsed_resume.get('current_role')}\n"
        f"Experience: {parsed_resume.get('years_experience')} years\n"
        f"Skills: {', '.join(parsed_resume.get('skills', []))}\n"
        f"Education: {parsed_resume.get('education')}\n"
        f"Summary: {parsed_resume.get('summary')}"
    )

    messages = [
        SystemMessage(content=MATCHER_PROMPT),
        HumanMessage(content=f"JOB DESCRIPTION:\n{jd[:3000]}\n\nCANDIDATE PROFILE:\n{candidate_block}"),
    ]

    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        content = " ".join(parts)

    text = str(content)
    result = {"score": 0, "matched": [], "missing": [], "reasoning": text}

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("MATCH_SCORE:"):
            try:
                result["score"] = int("".join(c for c in line.split(":", 1)[1] if c.isdigit()))
            except Exception:
                result["score"] = 0
        elif line.startswith("MATCHED_SKILLS:"):
            result["matched"] = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
        elif line.startswith("MISSING_SKILLS:"):
            result["missing"] = [s.strip() for s in line.split(":", 1)[1].split(",") if s.strip()]
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result