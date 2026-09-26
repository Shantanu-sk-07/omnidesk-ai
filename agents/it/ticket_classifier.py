"""
IT Agent 1: Ticket Classifier
Categorizes IT tickets into predefined categories.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


CLASSIFIER_PROMPT = """You are an IT Ticket Classifier.

Classify the user's IT issue into ONE category:
- VPN
- PASSWORD
- WIFI
- LAPTOP_PERFORMANCE
- EMAIL
- SOFTWARE_INSTALL
- PRINTER
- HARDWARE
- OTHER

Reply with ONLY the category name. No explanation."""


def classify_ticket(user_input: str, api_key: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )
    response = llm.invoke([
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=user_input),
    ])
    content = response.content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        content = " ".join(parts)
    cat = str(content).strip().upper().replace(" ", "_")
    valid = ["VPN", "PASSWORD", "WIFI", "LAPTOP_PERFORMANCE", "EMAIL",
             "SOFTWARE_INSTALL", "PRINTER", "HARDWARE", "OTHER"]
    for v in valid:
        if v in cat:
            return v
    return "OTHER"