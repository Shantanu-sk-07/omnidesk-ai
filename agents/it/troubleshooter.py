"""
IT Agent 3: Troubleshooter
Generates step-by-step fix using KB articles.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from agents.it.kb_search import search_kb
from agents.it.ticket_classifier import classify_ticket


TROUBLESHOOTER_PROMPT = """You are an IT Support Troubleshooter Agent.

You will receive:
1. The user's IT issue.
2. The category.
3. Relevant KB articles (if any).

Produce a clear response:

**Category**: <category>

**Diagnosis**: <1-2 sentences about likely cause>

**Steps to Fix**:
1. <step>
2. <step>
3. <step>

**Escalation**: <Yes or No> — <when to escalate and to whom>

**Confidence**: <0-100>%

Rules:
- Use KB articles if provided. If no KB match, use your IT knowledge.
- Be specific and actionable.
- Max 6 steps.
- Use markdown.
"""


def troubleshoot(user_input: str, api_key: str) -> str:
    category = classify_ticket(user_input, api_key)
    kb_articles = search_kb(user_input, top_k=2)

    kb_text = ""
    if kb_articles:
        kb_text = "\n\n".join(
            f"[KB: {a['title']}]\n{a['content']}" for a in kb_articles
        )
    else:
        kb_text = "(No KB articles matched — use general IT knowledge.)"

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.2,
        google_api_key=api_key,
    )

    messages = [
        SystemMessage(content=TROUBLESHOOTER_PROMPT),
        HumanMessage(content=(
            f"USER ISSUE: {user_input}\n\n"
            f"CATEGORY: {category}\n\n"
            f"KB ARTICLES:\n{kb_text}"
        )),
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
    return str(content)