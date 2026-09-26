"""
OmniDesk AI - Supervisor Router
Decides which team (HR / IT / General) handles each request.
"""

from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


SUPERVISOR_PROMPT = """You are the Supervisor Router for OmniDesk AI.

Your ONLY job: classify the user's request into ONE category.

CATEGORIES:
- HR   → resumes, hiring, candidates, job descriptions, interviews, recruitment
- IT   → VPN, password, wifi, laptop, email, printer, software, tickets, troubleshooting
- MIXED → contains BOTH HR and IT tasks
- GENERAL → everything else (chit-chat, math, general questions, coding help)

EXAMPLES:
"What's 2+2?" → GENERAL
"Screen this resume for Python role" → HR
"My VPN isn't working" → IT
"Hire a DevOps engineer and set up their laptop" → MIXED
"Tell me a joke" → GENERAL

Reply with ONLY one word: HR, IT, MIXED, or GENERAL.
No punctuation, no explanation."""


def classify_intent(user_input: str, api_key: str) -> str:
    """Returns: HR | IT | MIXED | GENERAL"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )
    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=user_input),
    ]
    try:
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
        intent = str(content).strip().upper()
        # Sanity check
        for key in ["MIXED", "HR", "IT", "GENERAL"]:
            if key in intent:
                return key
        return "GENERAL"
    except Exception:
        return "GENERAL"