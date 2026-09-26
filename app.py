import streamlit as st
import os
from dotenv import load_dotenv

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

# ============ STATE ============
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_qa_history" not in st.session_state:
    st.session_state.resume_qa_history = []
if "analyzed_resume_text" not in st.session_state:
    st.session_state.analyzed_resume_text = ""

is_dark = st.session_state.theme_mode == "dark"

# ============ THEME TOKENS ============
if is_dark:
    T = dict(bg="#0f1117", surface="#16181f", surface2="#1c1f28",
             border="#262936", text="#e5e7eb", text_muted="#94a3b8",
             text_head="#f1f5f9", accent="#818cf8",
             accent_soft="#1e1b4b", accent_soft_text="#a5b4fc",
             rg_bg="#064e3b", rg_fg="#6ee7b7",
             rh_bg="#78350f", rh_fg="#fcd34d",
             ri_bg="#1e3a8a", ri_fg="#93c5fd",
             rm_bg="#831843", rm_fg="#f9a8d4",
             score="#6ee7b7")
else:
    T = dict(bg="#ffffff", surface="#ffffff", surface2="#fafafa",
             border="#eaeaea", text="#111827", text_muted="#64748b",
             text_head="#0f172a", accent="#4f46e5",
             accent_soft="#eef2ff", accent_soft_text="#4338ca",
             rg_bg="#ecfdf5", rg_fg="#065f46",
             rh_bg="#fef3c7", rh_fg="#92400e",
             ri_bg="#dbeafe", ri_fg="#1e40af",
             rm_bg="#fce7f3", rm_fg="#9d174d",
             score="#059669")

# ============ GLOBAL STYLE ============
st.markdown(f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }}

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: {T['bg']} !important;
        color: {T['text']} !important;
    }}

    [data-testid="stSidebar"] {{
        background: {T['surface2']} !important;
        border-right: 1px solid {T['border']} !important;
    }}

    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {{
        color: {T['text']} !important;
    }}
    .stCaption, small {{ color: {T['text_muted']} !important; }}

    .block-container {{
        padding-top: 2rem; max-width: 1200px; padding-bottom: 6rem;
    }}

    /* ---------- PAGE TITLE ---------- */
    h1 {{
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: {T['text_head']} !important;
    }}

    /* ---------- TABS — BIGGER & CLEARER ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px !important;
        border-bottom: 2px solid {T['border']} !important;
        margin-bottom: 24px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 52px !important;
        padding: 0 22px !important;
        background: transparent !important;
        color: {T['text_muted']} !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
        border-radius: 8px 8px 0 0 !important;
        letter-spacing: 0.01em;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {T['text']} !important;
        background: {T['surface2']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {T['accent']} !important;
        background: {T['surface2']} !important;
        border-bottom: 3px solid {T['accent']} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
    .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

    /* ---------- SUBHEADERS ---------- */
    h3 {{
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        margin-top: 8px !important;
        margin-bottom: 6px !important;
        color: {T['text_head']} !important;
    }}

    /* ---------- RADIO GROUPS (mode selectors) ---------- */
    .stRadio > div {{
        gap: 8px !important;
        flex-wrap: wrap;
    }}
    .stRadio > div > label {{
        background: {T['surface']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 10px !important;
        padding: 10px 18px !important;
        cursor: pointer;
        transition: all 0.15s ease;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }}
    .stRadio > div > label:hover {{
        border-color: {T['accent']} !important;
    }}
    .stRadio > div > label[data-checked="true"],
    .stRadio > div > label:has(input:checked) {{
        background: {T['accent_soft']} !important;
        border-color: {T['accent']} !important;
        color: {T['accent_soft_text']} !important;
    }}

    /* ---------- CHAT INPUT (PINNED TO BOTTOM) ---------- */
    [data-testid="stBottomBlockContainer"] {{
        background: {T['bg']} !important;
        border-top: 1px solid {T['border']} !important;
        padding: 12px 0 18px 0 !important;
    }}
    [data-testid="stBottom"] > div {{ background: transparent !important; }}

    .stChatInput {{ max-width: 1200px; margin: 0 auto; }}
    .stChatInput > div {{
        background: {T['surface']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15) !important;
    }}
    .stChatInput textarea {{
        background: {T['surface']} !important;
        color: {T['text']} !important;
        border: none !important;
        font-size: 0.95rem !important;
        padding: 12px 16px !important;
    }}
    .stChatInput textarea::placeholder {{ color: {T['text_muted']} !important; }}
    .stChatInput button {{ color: {T['accent']} !important; }}

    /* ---------- CHAT MESSAGES ---------- */
    .stChatMessage {{
        background: {T['surface']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 12px !important;
        margin-bottom: 10px;
    }}

    /* ---------- BADGES ---------- */
    .route-badge {{
        display: inline-block; font-size: 0.7rem; font-weight: 700;
        padding: 4px 11px; border-radius: 6px; margin-bottom: 8px;
        letter-spacing: 0.06em; text-transform: uppercase;
        background: {T['rg_bg']}; color: {T['rg_fg']};
    }}
    .route-hr    {{ background: {T['rh_bg']}; color: {T['rh_fg']}; }}
    .route-it    {{ background: {T['ri_bg']}; color: {T['ri_fg']}; }}
    .route-mixed {{ background: {T['rm_bg']}; color: {T['rm_fg']}; }}

    .tool-badge {{
        display: inline-block; background: {T['accent_soft']};
        color: {T['accent_soft_text']}; font-size: 0.72rem; font-weight: 600;
        padding: 4px 11px; border-radius: 6px; margin: 2px 6px 2px 0;
    }}

    /* ---------- TEXT AREAS ---------- */
    .stTextArea textarea {{
        background: {T['surface']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 10px !important;
        font-size: 0.9rem !important;
    }}
    .stTextArea textarea:focus {{
        border-color: {T['accent']} !important;
        box-shadow: 0 0 0 3px {T['accent']}33 !important;
    }}

    /* ---------- FILE UPLOADER ---------- */
    [data-testid="stFileUploader"] {{
        background: {T['surface']} !important;
        border: 1px dashed {T['border']} !important;
        border-radius: 12px !important;
        padding: 8px !important;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {T['accent']} !important;
    }}

    /* ---------- CANDIDATE CARD ---------- */
    .candidate-card {{
        border: 1px solid {T['border']};
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        background: {T['surface']};
        transition: border-color 0.15s ease;
    }}
    .candidate-card:hover {{ border-color: {T['accent']}; }}
    .candidate-name {{
        font-size: 1.1rem; font-weight: 700;
        color: {T['text_head']}; margin: 0;
    }}
    .candidate-score {{
        font-size: 0.9rem; font-weight: 700;
        color: {T['score']}; margin: 6px 0 10px 0;
    }}
    .candidate-label {{
        font-size: 0.7rem; font-weight: 700; color: {T['text_muted']};
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 14px 0 6px 0;
    }}
    .candidate-text {{
        margin: 0; color: {T['text']};
        font-size: 0.92rem; line-height: 1.55;
    }}

    /* ---------- BUTTONS ---------- */
    .stButton > button {{
        background: {T['surface']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {T['accent']} !important;
        color: {T['accent']} !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {T['accent']} !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 20px !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        opacity: 0.92 !important;
        color: #ffffff !important;
    }}

    /* Section labels */
    .section-label {{
        font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: {T['text_muted']}; margin: 18px 0 10px 0;
    }}
</style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.title("OmniDesk AI")
st.caption("Unified Agentic Assistant — HR, IT & General")

API_KEY = os.getenv("GEMINI_API_KEY")

# ============ SIDEBAR ============
with st.sidebar:
    st.subheader("Appearance")
    new_mode = st.radio(
        "Theme",
        options=["dark", "light"],
        index=0 if is_dark else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    if new_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = new_mode
        st.rerun()

    st.divider()
    st.subheader("Controls")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.3, 0.1)
    show_tools = st.checkbox("Show tool usage", value=True)

    st.divider()
    st.subheader("Capabilities")
    st.markdown("""
- Web Search
- Calculator
- Date & Time
- Supervisor Router
- HR Screening
- Resume Analyzer
- Resume Q&A
- IT Support
""")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============ CHAT INPUT (TOP-LEVEL → pinned to bottom) ============
user_prompt = st.chat_input(
    "Ask anything — try 'my VPN isn't working' or 'what is 234*567?'"
)

# ============ TABS ============
tab_chat, tab_hr = st.tabs([":material/chat: Chat", ":material/group: HR Screening"])

# ============================================================
# TAB 1 — CHAT
# ============================================================
with tab_chat:
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.spinner("Thinking..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                answer, tool_trace, route = run_agent(
                    user_input=user_prompt,
                    chat_history=history,
                    api_key=API_KEY,
                    temperature=temperature,
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "route": route,
                    "tools": tool_trace,
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Error:** {e}",
                    "route": "ERROR",
                    "tools": [],
                })

    for msg in st.session_state.messages:
        avatar = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("route"):
                route = msg["route"]
                css = f"route-{route.lower()}" if route.lower() in ("hr", "it", "mixed") else ""
                st.markdown(
                    f'<span class="route-badge {css}">{route}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if show_tools and msg.get("tools"):
                unique = list(dict.fromkeys(msg["tools"]))
                badges = "".join(f'<span class="tool-badge">{t}</span>' for t in unique)
                st.markdown(badges, unsafe_allow_html=True)

# ============================================================
# TAB 2 — HR SCREENING
# ============================================================
with tab_hr:
    st.markdown("### HR Screening & Resume Analyzer")
    st.caption("Three tools in one place — rank candidates, deep-analyze a resume, or chat about it.")

    hr_mode = st.radio(
        "Choose mode",
        options=[
            "Screen & Rank",
            "Analyze Resume",
            "Resume Q&A",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ------------------------------------------------------
    # MODE 1 — SCREEN & RANK
    # ------------------------------------------------------
    if hr_mode == "Screen & Rank":
        st.markdown("#### Screen & Rank Candidates")
        st.caption("Compare many resumes against one job description. Get a ranked shortlist.")

        st.markdown('<p class="section-label">Job Description</p>', unsafe_allow_html=True)
        jd_text = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the full job description here...",
            label_visibility="collapsed",
        )

        st.markdown('<p class="section-label">Resume Input Method</p>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Resume input mode",
            ["Paste text", "Upload files"],
            horizontal=True,
            label_visibility="collapsed",
            key="screen_input_mode",
        )

        resumes_list = []

        if input_mode == "Paste text":
            resumes_raw = st.text_area(
                "Resumes (separate each with ---)",
                height=280,
                placeholder="Resume 1...\n\n---\n\nResume 2...\n\n---\n\nResume 3...",
            )
            if resumes_raw.strip():
                resumes_list = [r.strip() for r in resumes_raw.split("---") if r.strip()]
        else:
            uploaded_files = st.file_uploader(
                "Upload resumes (PDF, DOCX, or TXT)",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                st.caption(f"{len(uploaded_files)} file(s) selected")
                from agents.hr.file_reader import extract_text_from_upload
                for f in uploaded_files:
                    text = extract_text_from_upload(f)
                    if text.strip() and not text.startswith("(Error"):
                        resumes_list.append(text)
                    else:
                        st.warning(f"Could not read {f.name}: {text[:100]}")

        if st.button("Screen & Rank Candidates", type="primary", use_container_width=True):
            if not jd_text.strip():
                st.warning("Please enter a job description.")
            elif not resumes_list:
                st.warning("Please provide at least one resume.")
            else:
                st.info(f"Screening {len(resumes_list)} resume(s). This may take 30–60 seconds.")

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

    # ------------------------------------------------------
    # MODE 2 — ANALYZE RESUME
    # ------------------------------------------------------
    elif hr_mode == "Analyze Resume":
        st.markdown("#### Deep Resume Analysis")
        st.caption("Get a detailed report: strengths, weaknesses, ATS score, interview focus, and recommendations.")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<p class="section-label">Resume</p>', unsafe_allow_html=True)
            single_input_mode = st.radio(
                "Resume input",
                ["Paste text", "Upload file"],
                horizontal=True,
                label_visibility="collapsed",
                key="single_resume_mode",
            )

            single_resume = ""
            if single_input_mode == "Paste text":
                single_resume = st.text_area(
                    "Resume text",
                    height=340,
                    placeholder="Paste the resume here...",
                    label_visibility="collapsed",
                )
            else:
                single_file = st.file_uploader(
                    "Upload resume (PDF, DOCX, or TXT)",
                    type=["pdf", "docx", "txt"],
                    key="single_resume_upload",
                )
                if single_file:
                    from agents.hr.file_reader import extract_text_from_upload
                    single_resume = extract_text_from_upload(single_file)
                    if single_resume.strip() and not single_resume.startswith("(Error"):
                        st.success(f"Loaded: {single_file.name}")
                        with st.expander("Preview extracted text"):
                            st.text(single_resume[:2000] + ("..." if len(single_resume) > 2000 else ""))
                    else:
                        st.error(f"Could not read file: {single_resume[:120]}")

        with col2:
            st.markdown('<p class="section-label">Job Description (optional)</p>', unsafe_allow_html=True)
            single_jd = st.text_area(
                "JD",
                height=340,
                placeholder="Paste the JD here (optional)...",
                label_visibility="collapsed",
            )

        if st.button("Analyze Resume", type="primary", use_container_width=True):
            if not single_resume.strip():
                st.warning("Please paste or upload a resume.")
            else:
                with st.spinner("Analyzing resume..."):
                    try:
                        report = analyze_resume(single_resume, API_KEY, single_jd)
                        st.markdown(report)
                        st.session_state.analyzed_resume_text = single_resume
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    # ------------------------------------------------------
    # MODE 3 — RESUME Q&A
    # ------------------------------------------------------
    else:
        st.markdown("#### Ask Questions About a Resume")
        st.caption("Load a resume once, then ask unlimited questions — perfect for interview prep.")

        qa_input_mode = st.radio(
            "Resume input",
            ["Paste text", "Upload file"],
            horizontal=True,
            label_visibility="collapsed",
            key="qa_resume_mode",
        )

        resume_input = ""
        if qa_input_mode == "Paste text":
            resume_input = st.text_area(
                "Resume (paste once)",
                value=st.session_state.analyzed_resume_text,
                height=200,
                placeholder="Paste the resume here...",
            )
        else:
            qa_file = st.file_uploader(
                "Upload resume (PDF, DOCX, or TXT)",
                type=["pdf", "docx", "txt"],
                key="qa_resume_upload",
            )
            if qa_file:
                from agents.hr.file_reader import extract_text_from_upload
                resume_input = extract_text_from_upload(qa_file)
                if resume_input.strip() and not resume_input.startswith("(Error"):
                    st.success(f"Loaded: {qa_file.name}")
                else:
                    st.error(f"Could not read file: {resume_input[:120]}")

        if st.button("Load Resume", type="primary"):
            if resume_input.strip():
                st.session_state.analyzed_resume_text = resume_input
                st.session_state.resume_qa_history = []
                st.success("Resume loaded. Ask your questions below.")
            else:
                st.warning("Please paste or upload a resume first.")

        if st.session_state.analyzed_resume_text:
            st.markdown("---")
            st.markdown("##### Conversation")

            for qa in st.session_state.resume_qa_history:
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(qa["q"])
                with st.chat_message("assistant", avatar=":material/smart_toy:"):
                    st.markdown(qa["a"])

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