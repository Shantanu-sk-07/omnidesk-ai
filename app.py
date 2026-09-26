import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid

from agent import run_agent
from agents.hr.resume_parser import parse_resume
from agents.hr.jd_matcher import match_resume_to_jd
from agents.hr.ranker import rank_candidates
from agents.hr.analyze_resume import analyze_resume, ask_resume_question

load_dotenv()

# ============ PAGE SETUP ============
st.set_page_config(
    page_title="OmniDesk AI",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = os.getenv("GEMINI_API_KEY")

# ============ STATE ============
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "view" not in st.session_state:
    st.session_state.view = "chat"
if "renaming_chat" not in st.session_state:
    st.session_state.renaming_chat = None
if "resume_qa_history" not in st.session_state:
    st.session_state.resume_qa_history = []
if "analyzed_resume_text" not in st.session_state:
    st.session_state.analyzed_resume_text = ""

is_dark = st.session_state.theme_mode == "dark"


# ============ HELPERS ============
def new_chat():
    cid = st.session_state.active_chat_id
    if cid and cid in st.session_state.chats:
        if not st.session_state.chats[cid]["messages"]:
            return cid
    cid = str(uuid.uuid4())
    st.session_state.chats[cid] = {
        "title": "New chat",
        "created": datetime.now().strftime("%b %d, %H:%M"),
        "messages": [],
    }
    st.session_state.active_chat_id = cid
    return cid


def get_active_chat():
    cid = st.session_state.active_chat_id
    if cid is None or cid not in st.session_state.chats:
        cid = new_chat()
    return st.session_state.chats[cid]


def active_chat_is_empty():
    cid = st.session_state.active_chat_id
    if not cid or cid not in st.session_state.chats:
        return True
    return not st.session_state.chats[cid]["messages"]


def send_message(prompt: str, temperature: float):
    if not prompt.strip():
        return
    chat = get_active_chat()
    chat["messages"].append({"role": "user", "content": prompt})
    if chat["title"] == "New chat":
        chat["title"] = prompt[:32] + ("…" if len(prompt) > 32 else "")

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in chat["messages"][:-1]
    ]
    try:
        answer, tool_trace, route = run_agent(
            user_input=prompt,
            chat_history=history,
            api_key=API_KEY,
            temperature=temperature,
        )
        chat["messages"].append({
            "role": "assistant",
            "content": answer,
            "route": route,
            "tools": tool_trace,
        })
    except Exception as e:
        chat["messages"].append({
            "role": "assistant",
            "content": f"**Error:** {e}",
            "route": "ERROR",
            "tools": [],
        })


# ============ THEME TOKENS ============
if is_dark:
    T = dict(
        bg="#0f1117", surface="#16181f", surface2="#1c1f28",
        border="#262936", text="#e5e7eb", text_muted="#94a3b8",
        text_head="#f1f5f9", accent="#818cf8",
        accent_soft="#1e1b4b", accent_soft_text="#a5b4fc",
        user_bubble="#1e3a8a", user_text="#e0e7ff",
        rg_bg="#064e3b", rg_fg="#6ee7b7",
        rh_bg="#78350f", rh_fg="#fcd34d",
        ri_bg="#1e3a8a", ri_fg="#93c5fd",
        rm_bg="#831843", rm_fg="#f9a8d4",
        score="#6ee7b7",
        sidebar_btn_hover="#1e2535",
        card_shadow="rgba(129, 140, 248, 0.15)",
    )
else:
    T = dict(
        bg="#f8fafc", surface="#ffffff", surface2="#f1f5f9",
        border="#e2e8f0", text="#1e293b", text_muted="#64748b",
        text_head="#0f172a", accent="#6366f1",
        accent_soft="#e0e7ff", accent_soft_text="#3730a3",
        user_bubble="#6366f1", user_text="#ffffff",
        rg_bg="#d1fae5", rg_fg="#065f46",
        rh_bg="#fef3c7", rh_fg="#92400e",
        ri_bg="#dbeafe", ri_fg="#1e40af",
        rm_bg="#fce7f3", rm_fg="#9d174d",
        score="#059669",
        sidebar_btn_hover="#e2e8f0",
        card_shadow="rgba(99, 102, 241, 0.10)",
    )


# ============ GLOBAL STYLE ============
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }}
    .stApp, [data-testid="stAppViewContainer"] {{
        background: {T['bg']} !important;
        color: {T['text']} !important;
    }}
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {{
        color: {T['text']} !important;
    }}
    .stCaption, small {{ color: {T['text_muted']} !important; }}

    #MainMenu {{ visibility: hidden !important; }}
    footer {{ visibility: hidden !important; }}

    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    .block-container {{
        padding-top: 1.5rem !important;
        padding-bottom: 8rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }}

    /* ---------- BRANDING HEADER ---------- */
    .brand-wrap {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding-bottom: 16px;
        margin-bottom: 10px;
        border-bottom: 1px solid {T['border']};
    }}
    .brand-icon {{
        width: 46px; height: 46px;
        border-radius: 12px;
        background: linear-gradient(135deg, {T['accent']} 0%, #a78bfa 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-weight: 800;
        font-size: 1.2rem;
        flex-shrink: 0;
        box-shadow: 0 4px 12px {T['card_shadow']};
    }}
    .brand-text {{
        display: flex;
        flex-direction: column;
        line-height: 1.2;
    }}
    .brand-title {{
        font-size: 1.55rem;
        font-weight: 800;
        color: {T['text_head']};
        letter-spacing: -0.02em;
    }}
    .brand-sub {{
        font-size: 0.82rem;
        color: {T['text_muted']};
        margin-top: 2px;
    }}

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {{
        background: {T['surface2']} !important;
        border-right: 1px solid {T['border']} !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.12s ease !important;
        min-height: 34px !important;
        font-size: 0.85rem !important;
    }}

    /* New chat button */
    div[data-testid="stSidebar"] .new-chat-btn + div button,
    div[data-testid="stSidebar"] .new-chat-btn ~ div button {{
        background: {T['accent']} !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 12px 16px !important;
        min-height: 44px !important;
        text-align: center !important;
        box-shadow: 0 2px 8px {T['card_shadow']} !important;
    }}

    /* Chat row buttons */
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
        background: transparent !important;
        color: {T['text']} !important;
        border: 1px solid transparent !important;
        text-align: left !important;
        padding: 6px 10px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
        background: {T['sidebar_btn_hover']} !important;
        border-color: {T['border']} !important;
    }}
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {T['accent_soft']} !important;
        color: {T['accent_soft_text']} !important;
        border: 1px solid {T['accent']} !important;
        text-align: left !important;
        padding: 6px 10px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}

    /* ---------- CHAT MESSAGES ---------- */
    .user-msg {{
        display: flex; justify-content: flex-end; margin: 14px 0;
    }}
    .user-msg .bubble {{
        background: {T['user_bubble']}; color: {T['user_text']};
        padding: 10px 16px; border-radius: 18px 18px 4px 18px;
        max-width: 75%; font-size: 0.94rem; line-height: 1.5;
        word-wrap: break-word; overflow-wrap: break-word;
    }}
    .ai-msg {{
        display: flex; justify-content: flex-start; margin: 14px 0;
    }}
    .ai-msg .bubble {{
        background: {T['surface']}; color: {T['text']};
        border: 1px solid {T['border']};
        padding: 12px 16px; border-radius: 18px 18px 18px 4px;
        max-width: 85%; font-size: 0.94rem; line-height: 1.55;
        word-wrap: break-word; overflow-wrap: break-word;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}

    /* ---------- WELCOME ---------- */
    .welcome-wrap {{
        text-align: center;
        padding: 60px 10px 20px 10px;
    }}
    .welcome-title {{
        font-size: 2rem; font-weight: 700;
        color: {T['text_head']}; letter-spacing: -0.02em;
        margin-bottom: 8px;
    }}
    .welcome-sub {{
        font-size: 0.92rem; color: {T['text_muted']};
    }}

    /* ---------- BADGES ---------- */
    .route-badge {{
        display: inline-block; font-size: 0.68rem; font-weight: 700;
        padding: 3px 10px; border-radius: 6px; margin-bottom: 6px;
        letter-spacing: 0.06em; text-transform: uppercase;
        background: {T['rg_bg']}; color: {T['rg_fg']};
    }}
    .route-hr    {{ background: {T['rh_bg']}; color: {T['rh_fg']}; }}
    .route-it    {{ background: {T['ri_bg']}; color: {T['ri_fg']}; }}
    .route-mixed {{ background: {T['rm_bg']}; color: {T['rm_fg']}; }}

    .tool-badge {{
        display: inline-block; background: {T['accent_soft']};
        color: {T['accent_soft_text']}; font-size: 0.7rem; font-weight: 600;
        padding: 3px 10px; border-radius: 6px; margin: 2px 6px 2px 0;
    }}

    .typing-dots {{
        display: inline-flex; gap: 4px; align-items: center;
        vertical-align: middle;
    }}
    .typing-dots span {{
        width: 6px; height: 6px;
        background: {T['text_muted']}; border-radius: 50%;
        animation: blink 1.4s infinite;
    }}
    .typing-dots span:nth-child(2) {{ animation-delay: 0.2s; }}
    .typing-dots span:nth-child(3) {{ animation-delay: 0.4s; }}
    @keyframes blink {{
        0%, 60%, 100% {{ opacity: 0.3; }}
        30% {{ opacity: 1; }}
    }}

    /* ---------- TOP BAR RADIO (single row) ---------- */
    div[data-testid="stRadio"] > div[role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 8px !important;
        align-items: center !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
        flex: 0 0 auto !important;
        background: {T['surface']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 20px !important;
        padding: 7px 18px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        white-space: nowrap !important;
        cursor: pointer !important;
        color: {T['text']} !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
        background: {T['accent_soft']} !important;
        border-color: {T['accent']} !important;
        color: {T['accent_soft_text']} !important;
    }}
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {{
        display: none !important;
    }}

    /* ---------- CHAT INPUT ---------- */
    [data-testid="stBottom"] {{ background: {T['bg']} !important; }}
    [data-testid="stBottomBlockContainer"] {{
        background: {T['bg']} !important;
        border-top: 1px solid {T['border']} !important;
        padding: 10px 16px 16px 16px !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }}
    .stChatInput {{ max-width: 900px; margin: 0 auto; }}
    .stChatInput > div {{
        background: {T['surface']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 24px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
    }}
    .stChatInput textarea {{
        background: {T['surface']} !important;
        color: {T['text']} !important;
        border: none !important;
        font-size: 0.95rem !important;
        padding: 12px 18px !important;
    }}
    .stChatInput textarea::placeholder {{ color: {T['text_muted']} !important; }}
    .stChatInput button {{ color: {T['accent']} !important; }}

    /* ---------- TEXT AREAS ---------- */
    .stTextArea textarea, .stTextInput input {{
        background: {T['surface']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 10px !important;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {T['accent']} !important;
        box-shadow: 0 0 0 3px {T['accent']}33 !important;
    }}

    [data-testid="stFileUploader"] {{
        background: {T['surface']} !important;
        border: 1px dashed {T['border']} !important;
        border-radius: 12px !important;
    }}

    .candidate-card {{
        border: 1px solid {T['border']}; border-radius: 12px;
        padding: 16px 20px; margin-bottom: 12px;
        background: {T['surface']};
        transition: all 0.15s ease;
    }}
    .candidate-card:hover {{
        border-color: {T['accent']};
        box-shadow: 0 4px 12px {T['card_shadow']};
    }}
    .candidate-name {{ font-size: 1rem; font-weight: 700; color: {T['text_head']}; margin: 0; }}
    .candidate-score {{ font-size: 0.85rem; font-weight: 700; color: {T['score']}; margin: 4px 0 8px 0; }}
    .candidate-label {{
        font-size: 0.68rem; font-weight: 700; color: {T['text_muted']};
        text-transform: uppercase; letter-spacing: 0.08em; margin: 12px 0 4px 0;
    }}
    .candidate-text {{ margin: 0; color: {T['text']}; font-size: 0.9rem; line-height: 1.5; }}

    .stButton > button {{
        background: {T['surface']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {T['accent']} !important;
        color: #ffffff !important;
        border: none !important;
    }}

    .section-label {{
        font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: {T['text_muted']}; margin: 16px 0 8px 0;
    }}

    /* ---------- MOBILE ---------- */
    @media (max-width: 768px) {{
        html, body, [class*="css"] {{ font-size: 13px !important; }}
        .block-container {{
            padding-top: 0.6rem !important;
            padding-bottom: 9rem !important;
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
        }}
        .brand-icon {{ width: 38px; height: 38px; font-size: 1rem; border-radius: 10px; }}
        .brand-title {{ font-size: 1.15rem; }}
        .brand-sub {{ font-size: 0.7rem; }}
        .welcome-title {{ font-size: 1.25rem; }}
        .welcome-sub {{ font-size: 0.78rem; }}
        .user-msg .bubble {{ max-width: 88%; font-size: 0.82rem; padding: 8px 12px; }}
        .ai-msg .bubble {{ max-width: 95%; font-size: 0.82rem; padding: 9px 12px; }}
        .stChatInput textarea {{
            font-size: 0.85rem !important;
            padding: 6px 12px !important;
            min-height: 36px !important;
            max-height: 36px !important;
        }}
        .stChatInput > div {{ border-radius: 20px !important; }}
        .stChatInput button {{
            padding: 4px !important;
            height: 34px !important;
            width: 34px !important;
        }}
        [data-testid="stBottomBlockContainer"] {{
            padding: 6px 10px 14px 10px !important;
        }}
        .route-badge, .tool-badge {{ font-size: 0.62rem; padding: 2px 8px; }}
        .candidate-name {{ font-size: 0.92rem; }}
        .candidate-text {{ font-size: 0.85rem; }}
        /* Topbar pills compact on mobile */
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
            padding: 6px 14px !important;
            font-size: 0.78rem !important;
            border-radius: 18px !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# ============ BRANDING HEADER ============
st.markdown(f"""
<div class="brand-wrap">
    <div class="brand-icon">OD</div>
    <div class="brand-text">
        <span class="brand-title">OmniDesk AI</span>
        <span class="brand-sub">Agentic Assistant &middot; HR, IT &amp; General</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============ SIDEBAR ============
with st.sidebar:
    st.markdown('<div class="new-chat-btn"></div>', unsafe_allow_html=True)
    if st.button("+  New chat", key="new_chat_btn", use_container_width=True):
        if not active_chat_is_empty():
            new_chat()
        st.session_state.view = "chat"
        st.rerun()

    # Auto-clean empty non-active chats
    empty_to_delete = [
        cid for cid, chat in st.session_state.chats.items()
        if not chat["messages"] and cid != st.session_state.active_chat_id
    ]
    for cid in empty_to_delete:
        del st.session_state.chats[cid]

    st.markdown("---")
    st.markdown('<p class="section-label">Chats</p>', unsafe_allow_html=True)

    if not st.session_state.chats:
        st.caption("No chats yet.")
    else:
        sorted_chats = sorted(
            st.session_state.chats.items(),
            key=lambda x: x[1]["created"],
            reverse=True,
        )
        for cid, chat in sorted_chats[:15]:
            is_active = cid == st.session_state.active_chat_id
            col_title, col_ren, col_del = st.columns([5, 1, 1])

            with col_title:
                label = f"{'▶ ' if is_active else ''}{chat['title'][:22]}"
                if st.button(
                    label,
                    key=f"open_{cid}",
                    use_container_width=True,
                    help=chat["title"],
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.active_chat_id = cid
                    st.session_state.view = "chat"
                    st.rerun()

            with col_ren:
                if st.button("Ren", key=f"ren_{cid}", help="Rename chat", use_container_width=True):
                    st.session_state.renaming_chat = cid
                    st.rerun()

            with col_del:
                if st.button("Del", key=f"del_{cid}", help="Delete chat", use_container_width=True):
                    del st.session_state.chats[cid]
                    if st.session_state.active_chat_id == cid:
                        st.session_state.active_chat_id = None
                    st.rerun()

            if st.session_state.renaming_chat == cid:
                new_name = st.text_input(
                    "New name",
                    value=chat["title"],
                    key=f"rename_input_{cid}",
                    label_visibility="collapsed",
                    placeholder="Chat name...",
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Save", key=f"save_{cid}", use_container_width=True, type="primary"):
                        chat["title"] = new_name.strip() or "Untitled"
                        st.session_state.renaming_chat = None
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"cancel_{cid}", use_container_width=True):
                        st.session_state.renaming_chat = None
                        st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-label">Settings</p>', unsafe_allow_html=True)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)
    show_tools = st.checkbox("Show tool usage", value=True)


# ============ TOP BAR — single row radio ============
current_index = 0
if st.session_state.view == "hr":
    current_index = 1

topbar_choice = st.radio(
    "Topbar",
    options=["Chat", "HR", "Theme"],
    index=current_index,
    horizontal=True,
    label_visibility="collapsed",
    key=f"topbar_radio_{st.session_state.theme_mode}",
)

if topbar_choice == "Theme":
    st.session_state.theme_mode = "light" if is_dark else "dark"
    st.rerun()
else:
    new_view = topbar_choice.lower()
    if st.session_state.view != new_view:
        st.session_state.view = new_view
        st.rerun()

st.markdown("---")


# ============================================================
# VIEW — CHAT
# ============================================================
if st.session_state.view == "chat":
    active = get_active_chat()
    is_empty = not active["messages"]

    if is_empty:
        st.markdown(f"""
<div class="welcome-wrap">
    <div class="welcome-title">What can I help you with?</div>
    <div class="welcome-sub">Ask anything — resume screening, IT troubleshooting, or general questions.</div>
</div>
""", unsafe_allow_html=True)

    for msg in active["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg"><div class="bubble">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            badge_html = ""
            if msg.get("route"):
                route = msg["route"]
                css = f"route-{route.lower()}" if route.lower() in ("hr", "it", "mixed") else ""
                badge_html += f'<span class="route-badge {css}">{route}</span> '
            if show_tools and msg.get("tools"):
                unique = list(dict.fromkeys(msg["tools"]))
                for t in unique:
                    badge_html += f'<span class="tool-badge">{t}</span>'
            bubble_content = f"{badge_html}<br>{msg['content']}" if badge_html else msg["content"]
            st.markdown(
                f'<div class="ai-msg"><div class="bubble">{bubble_content}</div></div>',
                unsafe_allow_html=True,
            )

    prompt = st.chat_input("Message OmniDesk AI...")
    if prompt:
        st.markdown(
            f'<div class="user-msg"><div class="bubble">{prompt}</div></div>',
            unsafe_allow_html=True,
        )
        thinking = st.empty()
        thinking.markdown(
            f'<div class="ai-msg"><div class="bubble">'
            f'<div class="typing-dots"><span></span><span></span><span></span></div>'
            f'<span style="margin-left:8px; color:{T["text_muted"]}; font-size:0.85rem;">Thinking…</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        send_message(prompt, temperature)
        thinking.empty()
        st.rerun()


# ============================================================
# VIEW — HR SCREENING
# ============================================================
else:
    st.markdown("### HR Screening & Resume Analyzer")
    st.caption("Rank candidates, deep-analyze a resume, or chat about it.")

    hr_mode = st.radio(
        "Choose mode",
        options=["Screen & Rank", "Analyze Resume", "Resume Q&A"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ---------- Screen & Rank ----------
    if hr_mode == "Screen & Rank":
        st.markdown("#### Screen & Rank Candidates")
        jd_text = st.text_area(
            "Job Description", height=180,
            placeholder="Paste the job description...",
        )
        input_mode = st.radio(
            "Resume input", ["Paste text", "Upload files"],
            horizontal=True, label_visibility="collapsed",
            key="screen_input_mode",
        )
        resumes_list = []
        if input_mode == "Paste text":
            resumes_raw = st.text_area(
                "Resumes (separate with ---)", height=240,
                placeholder="Resume 1...\n\n---\n\nResume 2...",
            )
            if resumes_raw.strip():
                resumes_list = [r.strip() for r in resumes_raw.split("---") if r.strip()]
        else:
            uploaded_files = st.file_uploader(
                "Upload resumes", type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                from agents.hr.file_reader import extract_text_from_upload
                for f in uploaded_files:
                    text = extract_text_from_upload(f)
                    if text.strip() and not text.startswith("(Error"):
                        resumes_list.append(text)
                    else:
                        st.warning(f"Could not read {f.name}")

        if st.button("Screen & Rank", type="primary", use_container_width=True):
            if not jd_text.strip() or not resumes_list:
                st.warning("Please provide a JD and at least one resume.")
            else:
                st.info(f"Screening {len(resumes_list)} resume(s)...")
                progress = st.progress(0)
                candidates = []
                for idx, resume_text in enumerate(resumes_list):
                    try:
                        parsed = parse_resume(resume_text, API_KEY)
                        match = match_resume_to_jd(jd_text, parsed, API_KEY)
                        candidates.append({
                            "name": parsed.get("name", f"Candidate {idx+1}"),
                            "role": parsed.get("current_role", "Unknown"),
                            "experience": parsed.get("years_experience", 0),
                            "skills": parsed.get("skills", []),
                            "score": match.get("score", 0),
                            "reasoning": match.get("reasoning", ""),
                        })
                    except Exception as e:
                        st.warning(f"Could not process resume {idx+1}: {e}")
                    progress.progress((idx + 1) / len(resumes_list))

                ranked = rank_candidates(candidates)
                st.success(f"Ranked {len(ranked)} candidate(s).")

                for i, c in enumerate(ranked, start=1):
                    skills_html = "".join(
                        f'<span class="tool-badge">{s}</span>' for s in c["skills"][:8]
                    )
                    st.markdown(f"""
<div class="candidate-card">
    <p class="candidate-name">#{i} — {c['name']}</p>
    <p class="candidate-score">Match Score: {c['score']}/100</p>
    <p class="candidate-text"><strong>Role:</strong> {c['role']} &nbsp;|&nbsp; <strong>Experience:</strong> {c['experience']} years</p>
    <p class="candidate-label">Skills</p>
    <p>{skills_html}</p>
    <p class="candidate-label">Reasoning</p>
    <p class="candidate-text">{c['reasoning']}</p>
</div>
""", unsafe_allow_html=True)

    # ---------- Analyze Resume ----------
    elif hr_mode == "Analyze Resume":
        st.markdown("#### Deep Resume Analysis")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            single_input_mode = st.radio(
                "Resume input", ["Paste text", "Upload file"],
                horizontal=True, label_visibility="collapsed",
                key="single_resume_mode",
            )
            single_resume = ""
            if single_input_mode == "Paste text":
                single_resume = st.text_area(
                    "Resume", height=300,
                    placeholder="Paste the resume...",
                    label_visibility="collapsed",
                )
            else:
                single_file = st.file_uploader(
                    "Upload resume", type=["pdf", "docx", "txt"],
                    key="single_resume_upload",
                )
                if single_file:
                    from agents.hr.file_reader import extract_text_from_upload
                    single_resume = extract_text_from_upload(single_file)
                    if single_resume.strip() and not single_resume.startswith("(Error"):
                        st.success(f"Loaded: {single_file.name}")
                    else:
                        st.error("Could not read file")

        with col2:
            single_jd = st.text_area(
                "Job Description (optional)", height=300,
                placeholder="Paste JD (optional)...",
                label_visibility="collapsed",
            )

        if st.button("Analyze Resume", type="primary", use_container_width=True):
            if not single_resume.strip():
                st.warning("Please provide a resume.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        report = analyze_resume(single_resume, API_KEY, single_jd)
                        st.markdown(report)
                        st.session_state.analyzed_resume_text = single_resume
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    # ---------- Resume Q&A ----------
    else:
        st.markdown("#### Ask Questions About a Resume")
        qa_input_mode = st.radio(
            "Resume input", ["Paste text", "Upload file"],
            horizontal=True, label_visibility="collapsed",
            key="qa_resume_mode",
        )
        resume_input = ""
        if qa_input_mode == "Paste text":
            resume_input = st.text_area(
                "Resume",
                value=st.session_state.analyzed_resume_text,
                height=180,
                placeholder="Paste the resume...",
            )
        else:
            qa_file = st.file_uploader(
                "Upload resume", type=["pdf", "docx", "txt"],
                key="qa_resume_upload",
            )
            if qa_file:
                from agents.hr.file_reader import extract_text_from_upload
                resume_input = extract_text_from_upload(qa_file)
                if resume_input.strip() and not resume_input.startswith("(Error"):
                    st.success(f"Loaded: {qa_file.name}")

        if st.button("Load Resume", type="primary"):
            if resume_input.strip():
                st.session_state.analyzed_resume_text = resume_input
                st.session_state.resume_qa_history = []
                st.success("Resume loaded. Ask questions below.")

        if st.session_state.analyzed_resume_text:
            st.markdown("---")
            for qa in st.session_state.resume_qa_history:
                st.markdown(
                    f'<div class="user-msg"><div class="bubble">{qa["q"]}</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="ai-msg"><div class="bubble">{qa["a"]}</div></div>',
                    unsafe_allow_html=True,
                )

            question = st.chat_input("Ask about this resume...", key="resume_qa_input")
            if question:
                st.session_state.resume_qa_history.append({"q": question, "a": "..."})
                with st.spinner("Analyzing..."):
                    try:
                        hist = []
                        for qa in st.session_state.resume_qa_history[:-1]:
                            hist.append({"role": "user", "content": qa["q"]})
                            hist.append({"role": "assistant", "content": qa["a"]})
                        ans = ask_resume_question(
                            st.session_state.analyzed_resume_text,
                            question,
                            hist,
                            API_KEY,
                        )
                        st.session_state.resume_qa_history[-1]["a"] = ans
                        st.rerun()
                    except Exception as e:
                        st.session_state.resume_qa_history[-1]["a"] = f"Error: {e}"
                        st.rerun()