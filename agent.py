"""
OmniDesk AI - Agentic Engine (Phase 3)
Supervisor-routed multi-domain agent.
"""

import os
from typing import TypedDict, Annotated, Sequence
from operator import add

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from datetime import datetime

from supervisor import classify_intent
from agents.it.troubleshooter import troubleshoot


# ============ TOOLS ============
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '2 + 2 * 10'."""
    try:
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


web_search = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the internet for current info, news, real-time data."
)

TOOLS = [web_search, calculator, get_current_datetime]


# ============ AGENT STATE ============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]


AGENT_SYSTEM_PROMPT = """You are OmniDesk AI — an autonomous multi-step reasoning agent.

Tools:
- web_search: current info
- calculator: math
- get_current_datetime: date/time

Rules:
1. THINK STEP BY STEP.
2. Use tools when needed. Multiple tools in sequence allowed.
3. VERIFY your answer.
4. Use markdown.
5. Be concise but complete.
"""


def build_agent(api_key: str, temperature: float = 0.3):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def call_model(state: AgentState):
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ============ HELPERS ============
def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text", "")
                if t:
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _extract_final_text_and_trace(final_messages):
    tool_trace = []
    for msg in final_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_trace.append(tc["name"])
    final_text = ""
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage):
            text = _extract_text(msg.content).strip()
            if text:
                final_text = text
                break
    return final_text, tool_trace


# ============ PUBLIC API ============
def run_agent(user_input: str, chat_history: list, api_key: str, temperature: float = 0.3):
    """
    Returns: (final_text, tool_trace, route)
    route ∈ {'GENERAL', 'HR', 'IT', 'MIXED'}
    """
    # --- Route the request ---
    intent = classify_intent(user_input, api_key)

    # --- IT path: use dedicated troubleshooter ---
    if intent == "IT":
        try:
            answer = troubleshoot(user_input, api_key)
            return answer, ["it_troubleshooter"], "IT"
        except Exception as e:
            return f"IT agent error: {e}", [], "IT"

    # --- HR path ---
    if intent == "HR":
        # Simple HR placeholder — full HR pipeline will be added when resumes are uploaded.
        # For now, route through the LLM with an HR-specific prompt.
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.2,
            google_api_key=api_key,
        )
        hr_prompt = """You are an HR Assistant Agent at OmniDesk AI.
Answer HR-related questions: hiring, resumes, job descriptions, interviews, onboarding.
If the user mentions specific resumes, ask them to upload them via the HR tab.
Be professional, concise, and use markdown."""
        msgs = [SystemMessage(content=hr_prompt)]
        for m in chat_history:
            msgs.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=user_input))
        try:
            resp = llm.invoke(msgs)
            return _extract_text(resp.content), ["hr_assistant"], "HR"
        except Exception as e:
            return f"HR agent error: {e}", [], "HR"

    # --- MIXED path: run IT troubleshooter AND general agent ---
    if intent == "MIXED":
        try:
            it_answer = troubleshoot(user_input, api_key)
            # Also run general agent for the non-IT part
            agent = build_agent(api_key, temperature)
            messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
            for m in chat_history:
                messages.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                                else AIMessage(content=m["content"]))
            messages.append(HumanMessage(content=user_input))
            result = agent.invoke({"messages": messages}, config={"recursion_limit": 25})
            general_text, tool_trace = _extract_final_text_and_trace(result["messages"])
            combined = (
                "### HR + IT Combined Response\n\n"
                "**General / HR part:**\n\n"
                f"{general_text}\n\n"
                "**IT Support part:**\n\n"
                f"{it_answer}"
            )
            return combined, ["hr_assistant", "it_troubleshooter"] + tool_trace, "MIXED"
        except Exception as e:
            return f"Mixed agent error: {e}", [], "MIXED"

    # --- GENERAL path: default agentic flow ---
    agent = build_agent(api_key, temperature)
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    for m in chat_history:
        messages.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": messages}, config={"recursion_limit": 25})
    final_text, tool_trace = _extract_final_text_and_trace(result["messages"])
    return final_text or "(No response)", tool_trace, "GENERAL"