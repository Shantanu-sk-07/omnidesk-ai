"""
HR Agent 1: Resume Parser
Extracts structured info from raw resume text.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


PARSER_PROMPT = """You are a Resume Parser Agent.
Extract structured information from the resume text below.

Return ONLY valid JSON in this exact format:
{
  "name": "candidate name or Unknown",
  "email": "email or Unknown",
  "years_experience": number,
  "current_role": "role or Unknown",
  "skills": ["skill1", "skill2"],
  "education": "highest degree or Unknown",
  "key_achievements": ["achievement1", "achievement2"],
  "summary": "2-line professional summary"
}

Rules:
- No markdown, no backticks, ONLY the JSON object.
- If info is missing, use "Unknown" or 0.
- Skills: max 15, most relevant.
- Achievements: max 3.
"""


def parse_resume(resume_text: str, api_key: str) -> dict:
    """Parse resume text into structured dict."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )
    messages = [
        SystemMessage(content=PARSER_PROMPT),
        HumanMessage(content=f"RESUME:\n\n{resume_text[:6000]}"),
    ]
    response = llm.invoke(messages)
    content = response.content

    # Handle list-type content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        content = " ".join(parts)

    text = str(content).strip()

    # Clean markdown fences if model ignored instruction
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "name": "Unknown",
            "email": "Unknown",
            "years_experience": 0,
            "current_role": "Unknown",
            "skills": [],
            "education": "Unknown",
            "key_achievements": [],
            "summary": text[:200] if text else "Could not parse resume.",
        }