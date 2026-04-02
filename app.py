import streamlit as st
import sys
import os
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import create_loan_agent, create_general_agent, get_response
from vector_store import build_vector_store_from_files, get_retriever

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ══ GLOBAL ══ */
.stApp {
    font-family: 'Inter', sans-serif;
    background: #ffffff !important;
}
#MainMenu {visibility:hidden;}
.stDeployButton {display:none;}
header[data-testid="stHeader"] {background:#ffffff!important;}
.main .block-container {padding-top:1rem;max-width:1100px;}
.main {background:#ffffff!important;}

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1628 0%, #0f1b33 50%, #0c1628 100%) !important;
    border-right: 1px solid rgba(100,150,255,0.08);
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}
.sidebar-brand {
    display:flex;align-items:center;gap:12px;
    padding:1.2rem 0.5rem 1.5rem;
    border-bottom:1px solid rgba(100,150,255,0.1);
    margin-bottom:1.2rem;
}
.sidebar-brand .brand-icon {font-size:1.6rem;}
.sidebar-brand .brand-text {font-size:1.05rem;font-weight:700;color:#e0e4f0;line-height:1.3;}
.sidebar-section-header {
    display:flex;align-items:center;gap:8px;
    font-size:0.72rem;font-weight:700;color:#5ba0e0;
    text-transform:uppercase;letter-spacing:1.5px;
    margin-bottom:0.8rem;margin-top:0.2rem;
}
section[data-testid="stSidebar"] .stButton > button {
    background:linear-gradient(135deg,#d4a84e,#e8c86a)!important;
    color:#2a1f0a!important;border:none!important;border-radius:8px!important;
    font-weight:700!important;font-size:0.85rem!important;
    padding:0.6rem 1.2rem!important;
    box-shadow:0 4px 15px rgba(201,164,78,0.25)!important;
    transition:all 0.3s ease!important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background:linear-gradient(135deg,#e0b85a,#f0d478)!important;
    box-shadow:0 6px 20px rgba(201,164,78,0.4)!important;
    transform:translateY(-1px)!important;
}
section[data-testid="stSidebar"] .stButton > button:disabled {
    background:rgba(100,110,140,0.3)!important;
    color:rgba(180,180,200,0.5)!important;box-shadow:none!important;
}
.file-item {
    background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
    border-radius:8px;padding:0.55rem 0.8rem;margin-bottom:0.4rem;
    font-size:0.8rem;color:#a0b8d8;display:flex;align-items:center;
}
.file-item .icon {margin-right:0.5rem;}

/* Sample question buttons styled as cards */
section[data-testid="stSidebar"] button[kind="secondary"] {
    background:rgba(255,255,255,0.05)!important;
    border:1px solid rgba(255,255,255,0.08)!important;
    border-radius:10px!important;
    color:#9ab0cc!important;
    font-size:0.78rem!important;
    font-weight:400!important;
    text-align:left!important;
    padding:0.65rem 0.8rem!important;
    box-shadow:none!important;
    transition:all 0.25s ease!important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background:rgba(100,160,255,0.08)!important;
    border-color:rgba(100,160,255,0.18)!important;
    color:#c0d4f0!important;
    transform:none!important;
}
.sidebar-divider {
    height:1px;
    background:linear-gradient(90deg,transparent,rgba(100,150,255,0.12),transparent);
    margin:1rem 0;
}
section[data-testid="stSidebar"] .stFileUploader label {
    color:#8898b8!important;font-size:0.8rem!important;
}
section[data-testid="stSidebar"] .stFileUploader > div > div {
    background:rgba(255,255,255,0.03)!important;
    border:2px dashed rgba(100,160,255,0.2)!important;
    border-radius:12px!important;
}
section[data-testid="stSidebar"] .stFileUploader small,
section[data-testid="stSidebar"] .stFileUploader span {
    color:#8898b8!important;
}

/* ══ HEADER (Blue) ══ */
.main-header {
    position:relative;
    background:linear-gradient(135deg,#0c1628 0%,#122040 30%,#1a2d5a 60%,#0c1628 100%);
    padding:2.5rem 2rem 2rem;border-radius:16px;margin-bottom:1.2rem;
    text-align:center;box-shadow:0 4px 24px rgba(12,22,40,0.35);
    overflow:hidden;border:1px solid rgba(100,150,255,0.06);
}
.main-header::before {
    content:'';position:absolute;top:0;left:0;right:0;bottom:0;
    background-image:
        radial-gradient(1px 1px at 15% 25%,rgba(255,255,255,0.4) 0%,transparent 100%),
        radial-gradient(1px 1px at 45% 15%,rgba(255,255,255,0.3) 0%,transparent 100%),
        radial-gradient(1px 1px at 75% 35%,rgba(255,255,255,0.35) 0%,transparent 100%),
        radial-gradient(1px 1px at 25% 65%,rgba(255,255,255,0.25) 0%,transparent 100%),
        radial-gradient(1px 1px at 85% 70%,rgba(255,255,255,0.3) 0%,transparent 100%),
        radial-gradient(1.5px 1.5px at 55% 50%,rgba(255,255,255,0.2) 0%,transparent 100%),
        radial-gradient(1px 1px at 35% 80%,rgba(255,255,255,0.3) 0%,transparent 100%),
        radial-gradient(1px 1px at 90% 20%,rgba(255,255,255,0.3) 0%,transparent 100%);
    pointer-events:none;
}
.main-header h1 {
    position:relative;z-index:1;color:#e8ecf8;font-size:2rem;font-weight:800;
    margin:0;letter-spacing:-0.5px;
}
.main-header p {
    position:relative;z-index:1;color:#8a96b8;font-size:0.92rem;margin-top:0.5rem;
}

/* ══ MODE CARDS ══ */
.mode-card {
    border-radius:14px;padding:1.2rem 1.3rem;height:100%;
    transition:all 0.3s ease;margin-bottom:0.5rem;
}
.mode-card:hover {transform:translateY(-2px);}
.mode-card .card-header {display:flex;align-items:center;gap:8px;margin-bottom:0.6rem;}
.mode-card .card-title {font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;}
.mode-card .card-desc {font-size:0.84rem;line-height:1.55;}

/* General Guidance = Brown/Amber */
.mode-card.gold {
    background:linear-gradient(135deg,#2a1f0e 0%,#3d2a14 100%);
    border:1px solid rgba(212,168,78,0.2);
    box-shadow:0 4px 20px rgba(42,31,14,0.3);
}
.mode-card.gold:hover {
    border-color:rgba(212,168,78,0.35);
    box-shadow:0 8px 30px rgba(42,31,14,0.4);
}
.mode-card.gold .card-title,.mode-card.gold .card-icon {color:#d4a84e;}
.mode-card.gold .card-desc {color:#c8b088;}

/* Document-Based Analysis = Blue */
.mode-card.blue {
    background:linear-gradient(135deg,#0e182d 0%,#122040 100%);
    border:1px solid rgba(80,140,230,0.15);
    box-shadow:0 4px 20px rgba(14,24,45,0.3);
}
.mode-card.blue:hover {
    border-color:rgba(80,140,230,0.3);
    box-shadow:0 8px 30px rgba(14,24,45,0.4);
}
.mode-card.blue .card-title,.mode-card.blue .card-icon {color:#5ba0e0;}
.mode-card.blue .card-desc {color:#94a3c0;}

/* ══ STATUS CARDS ══ */
.status-card {
    background:linear-gradient(135deg,#111d35 0%,#0e1628 100%);
    border:1px solid rgba(100,150,255,0.1);border-radius:14px;
    padding:1.2rem 1.3rem;margin-top:1rem;margin-bottom:1.2rem;color:#c8d6e5;
    box-shadow:0 4px 20px rgba(0,0,0,0.15);
}
.status-card.success {
    border-color:rgba(0,210,140,0.25);
    background:linear-gradient(135deg,#0d2e24 0%,#0e1628 100%);
}
.status-card.warning {
    border-color:rgba(212,168,83,0.2);
    background:linear-gradient(135deg,#2a1f0e 0%,#1e1a0e 100%);
}
.status-card h4 {
    margin:0 0 0.4rem;color:#e0e4f0;font-size:0.8rem;font-weight:700;
    text-transform:uppercase;letter-spacing:1px;
}
.status-card p {margin:0;font-size:0.88rem;line-height:1.5;color:#94a3c0;}

/* ══ CHAT ══ */
.stChatMessage {
    border-radius:14px!important;padding:1rem 1.2rem!important;
    margin-bottom:0.8rem!important;
}
[data-testid="stChatMessageUser"] {
    background:#f0f3f8!important;
    border:1px solid #dde3ef!important;
}
[data-testid="stChatMessageUser"] p,
[data-testid="stChatMessageUser"] li,
[data-testid="stChatMessageUser"] span {
    color:#1a2540!important;
}
[data-testid="stChatMessageAssistant"] {
    background:#f8f9fc!important;
    border:1px solid #e4e9f2!important;
    box-shadow:0 2px 10px rgba(0,0,0,0.05)!important;
}
[data-testid="stChatMessageAssistant"] p,
[data-testid="stChatMessageAssistant"] li,
[data-testid="stChatMessageAssistant"] span {
    color:#2a3450!important;
}
/* ══ CHAT INPUT (Search bar) ══ */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] > div > div > div,
.stChatInput, 
.stChatInput > div,
.stChatInputContainer {
    background-color: #ffffff !important;
    background: #ffffff !important;
}

[data-testid="stChatInput"] {
    border: 1px solid #d0d8e8 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    padding: 2px !important;
}

[data-testid="stChatInput"] textarea {
    color: #1a2540 !important;
    background-color: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8898b0 !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #d4a84e, #e8c86a) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #2a1f0a !important;
}

/* ══ MISC ══ */
::-webkit-scrollbar {width:6px;}
::-webkit-scrollbar-track {background:#f0f0f0;}
::-webkit-scrollbar-thumb {background:rgba(100,120,160,0.25);border-radius:3px;}
.main .stMarkdown p {color:#3a4560;}
hr {border-color:#e4e9f2!important;}
[data-testid="stHorizontalBlock"] {gap:1rem;}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "general_agent" not in st.session_state:
    st.session_state.general_agent = create_general_agent()
if "doc_agent" not in st.session_state:
    st.session_state.doc_agent = None
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []
if "sample_q" not in st.session_state:
    st.session_state.sample_q = None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">🏠</span>
        <span class="brand-text">AI Loan Approval<br>System</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-header"><span class="section-icon">📁</span> Document Upload</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload loan-related documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload bank policies, loan applications, credit reports, income proofs, etc.",
        label_visibility="collapsed"
    )

    if uploaded_files:
        for f in uploaded_files:
            ext_icon = "📕" if f.name.endswith(".pdf") else "📄"
            st.markdown(f'<div class="file-item"><span class="icon">{ext_icon}</span>{f.name}</div>', unsafe_allow_html=True)

    if st.button("⚡ Process Documents", use_container_width=True, disabled=not uploaded_files):
        with st.spinner("Ingesting documents & building vector index..."):
            vector_store, file_names = build_vector_store_from_files(uploaded_files)
            if vector_store:
                retriever = get_retriever(vector_store)
                st.session_state.doc_agent = create_loan_agent(retriever)
                st.session_state.docs_processed = True
                st.session_state.processed_file_names = file_names
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success(f"✅ {len(file_names)} document(s) processed! Switched to Document-Based Analysis mode.")
            else:
                st.error("❌ Could not extract content from uploaded files.")

    if st.session_state.docs_processed:
        st.markdown('<div class="sidebar-section-header"><span class="section-icon">✅</span> Active Documents</div>', unsafe_allow_html=True)
        for name in st.session_state.processed_file_names:
            ext_icon = "📕" if name.endswith(".pdf") else "📄"
            st.markdown(f'<div class="file-item"><span class="icon">{ext_icon}</span>{name}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-header"><span class="section-icon">⚡</span> Sample Questions</div>', unsafe_allow_html=True)

    samples = [
        "What is the minimum credit score required?",
        "What are the interest rates?",
        "How long is the loan approval process?",
    ]
    for i, q in enumerate(samples):
        if st.button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.sample_q = q
            st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat & Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.doc_agent = None
        st.session_state.docs_processed = False
        st.session_state.processed_file_names = []
        st.session_state.general_agent = create_general_agent()
        st.rerun()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 AI Loan Approval System</h1>
    <p>Ask loan questions instantly • Upload documents for personalized evaluation</p>
</div>
""", unsafe_allow_html=True)

# ── Mode Cards ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("""<div class="mode-card gold">
        <div class="card-header"><span class="card-icon">💛</span><span class="card-title">General Guidance Mode</span></div>
        <div class="card-desc">Get quick answers to general loan questions using banking knowledge.</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="mode-card blue">
        <div class="card-header"><span class="card-icon">📄</span><span class="card-title">Document-Based Analysis</span></div>
        <div class="card-desc">For a personalized evaluation, upload your loan documents (PDF or TXT) via the sidebar and switch modes after processing.</div>
    </div>""", unsafe_allow_html=True)

# ── Mode Status ──────────────────────────────────────────────────────────────
if not st.session_state.docs_processed:
    st.markdown("""
    <div class="status-card warning">
        <h4>💬 General Guidance Mode</h4>
        <p>You can start chatting right away! I'll answer your loan and banking questions using general knowledge.<br><br>
        <strong>Want a personalized evaluation?</strong> Upload your loan-related documents via the sidebar and click <strong>Process Documents</strong>.</p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="status-card success">
        <h4>📄 Document-Based Analysis Mode</h4>
        <p>{len(st.session_state.processed_file_names)} document(s) loaded and indexed. Responses are based on your uploaded documents.</p>
    </div>""", unsafe_allow_html=True)

# ── Chat Messages ────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Handle sample question click ─────────────────────────────────────────────
prompt = None
if st.session_state.sample_q:
    prompt = st.session_state.sample_q
    st.session_state.sample_q = None
else:
    prompt = st.chat_input("Ask about loan eligibility, policies, or documents...")

# ── Chat Input ───────────────────────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.docs_processed and st.session_state.doc_agent:
            mode_label = "📄 **Document-Based Analysis**"
            spinner_text = "Analyzing documents and evaluating..."
            active_agent = st.session_state.doc_agent
        else:
            mode_label = "📘 **General Guidance**"
            spinner_text = "Thinking..."
            active_agent = st.session_state.general_agent

        with st.spinner(spinner_text):
            response = get_response(active_agent, prompt, st.session_state.chat_history)
            
            # --- Rich UI Injection for Credit Score ---
            if "credit score" in prompt.lower() and "620" in response:
                ui_html = """
<div style="background:#ffffff;border:1px solid #e4e9f2;border-radius:12px;padding:1.2rem;margin-top:1rem;box-shadow:0 2px 10px rgba(0,0,0,0.02)">
    <div style="display:flex;justify-content:space-between;margin-bottom:8px;font-weight:700;font-size:0.85rem">
        <span style="color:#e04a4a;flex:1;text-align:center">Poor</span>
        <span style="color:#f59e0b;flex:1;text-align:center">Fair</span>
        <span style="color:#d4a84e;flex:1;text-align:center">Good</span>
        <span style="color:#6c7a9c;flex:1;text-align:center">Very Good</span>
        <span style="color:#6c7a9c;flex:1;text-align:center">Excellent</span>
    </div>
    <div style="display:flex;height:12px;border-radius:6px;overflow:hidden;margin-bottom:8px;position:relative">
        <div style="background:#e04a4a;flex:1"></div>
        <div style="background:#f59e0b;flex:1"></div>
        <div style="background:#eab308;flex:1;position:relative">
            <div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:6px;height:6px;background:#1a2540;border-radius:50%"></div>
        </div>
        <div style="background:#5ba0e0;flex:1"></div>
        <div style="background:#2a9d8f;flex:1"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6c7a9c;align-items:flex-start;">
        <span style="flex:1;text-align:center">300-579</span>
        <span style="flex:1;text-align:center">580-660</span>
        <span style="flex:1;text-align:center;position:relative">
            <span style="background:#d4a84e;color:#fff;padding:2px 8px;border-radius:6px;font-weight:700;font-size:0.85rem;display:inline-block">620</span>
        </span>
        <span style="flex:1;text-align:center">670-739</span>
        <span style="flex:1;text-align:center">800-850</span>
    </div>
</div>
"""
                response += "\n\n" + ui_html
            
            full_response = f"{mode_label}\n\n{response}"
            display_response = full_response.replace("$", "\\$")
            st.markdown(display_response, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response))
