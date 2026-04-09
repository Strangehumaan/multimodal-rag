import os
import shutil
import streamlit as st
from main import ingest, query

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

  :root {
    --bg: #f6f8fc;
    --surface: #ffffff;
    --surface-soft: #f8fafc;
    --border: #d9e2f0;
    --border-soft: #e6ecf5;
    --text: #0f172a;
    --text-dim: #475569;
    --primary: #2563eb;
    --primary-soft: rgba(37, 99, 235, 0.12);
    --success: #16a34a;
    --warn: #be185d;
    --btn-bg: #2563eb;
    --btn-bg-hover: #1d4ed8;
    --btn-bg-active: #1e40af;
    --btn-text: #ffffff;
    --btn-disabled-bg: #e2e8f0;
    --btn-disabled-text: #94a3b8;
  }

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
  }

  .stApp {
    background:
      radial-gradient(1200px 420px at 15% -10%, rgba(59, 130, 246, 0.10), transparent 55%),
      radial-gradient(900px 300px at 85% -20%, rgba(236, 72, 153, 0.07), transparent 60%),
      var(--bg);
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid var(--border);
    padding-top: 1.2rem;
  }

  section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--text-dim);
  }

  .block-container {
    max-width: 980px;
    padding: 2.2rem 2.4rem 2.6rem;
  }

  .hero {
    margin-bottom: 1.2rem;
    padding: 1.35rem 1.45rem;
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
  }

  .hero-kicker {
    display: inline-block;
    margin-bottom: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--primary);
  }

  .hero-title {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
  }

  .hero-sub {
    margin: 0.5rem 0 0;
    color: var(--text-dim);
    font-size: 0.92rem;
    line-height: 1.55;
  }

  .section-title {
    margin: 0 0 1rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid var(--border-soft);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
  }

  .answer-box {
    margin: 1.25rem 0;
    padding: 1.35rem 1.2rem;
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 12px;
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    font-size: 0.96rem;
    line-height: 1.75;
    color: var(--text);
    white-space: pre-wrap;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
  }

  .source-card {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 0.55rem;
    padding: 0.78rem 1rem;
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    background: #ffffff;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }

  .source-card:hover {
    border-color: #93b4ea;
    transform: translateY(-1px);
  }

  .source-badge {
    flex-shrink: 0;
    padding: 0.24rem 0.55rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: 1px solid transparent;
  }

  .badge-text  { color: #15803d; background: rgba(34, 197, 94, 0.12); border-color: rgba(34, 197, 94, 0.22); }
  .badge-image { color: #1d4ed8; background: rgba(59, 130, 246, 0.12); border-color: rgba(59, 130, 246, 0.22); }
  .badge-audio { color: #a21caf; background: rgba(217, 70, 239, 0.12); border-color: rgba(217, 70, 239, 0.22); }

  .source-filename {
    flex: 1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.77rem;
    color: #1e293b;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .source-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.69rem;
    color: var(--text-dim);
  }

  .file-pill {
    display: inline-block;
    margin: 0.15rem 0.2rem;
    padding: 0.2rem 0.62rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: #f8fafc;
    color: #334155;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
  }

  .stButton > button {
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    background: var(--btn-bg) !important;
    color: var(--btn-text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: none;
    min-height: 42px !important;
    padding: 0.58rem 1.15rem !important;
    transition: transform 0.12s ease, box-shadow 0.16s ease, background-color 0.16s ease, border-color 0.16s ease;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.22);
  }

  .stButton > button:hover {
    background: var(--btn-bg-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 18px rgba(37, 99, 235, 0.28);
  }

  .stButton > button:active {
    background: var(--btn-bg-active) !important;
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
  }

  .stButton > button:focus-visible {
    outline: none !important;
    border-color: #93c5fd !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.22), 0 8px 18px rgba(37, 99, 235, 0.25) !important;
  }

  .stButton > button:disabled {
    background: var(--btn-disabled-bg) !important;
    color: var(--btn-disabled-text) !important;
    border-color: #cbd5e1 !important;
    transform: none !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
  }

  section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
  }

  .block-container .stButton > button {
    width: auto;
    min-width: 140px;
  }

  div[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #93a6c4;
    border-radius: 12px;
    background: rgba(248, 250, 252, 0.95);
  }

  div[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-dim) !important;
  }

  .stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: #ffffff !important;
    color: var(--text) !important;
    font-size: 0.94rem !important;
    padding: 0.7rem 0.9rem !important;
  }

  .stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.35) !important;
  }

  div[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid var(--border);
  }

  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
UPLOAD_DIRS = {
    "pdf":  "data/documents",
    "docx": "data/documents",
    "jpg":  "data/images",
    "jpeg": "data/images",
    "png":  "data/images",
    "gif":  "data/images",
    "webp": "data/images",
    "mp3":  "data/audio",
    "mp4":  "data/audio",
    "wav":  "data/audio",
    "m4a":  "data/audio",
    "webm": "data/audio",
    "ogg":  "data/audio",
}

BADGE_CLASS = {
    "text":  "badge-text",
    "image": "badge-image",
    "audio": "badge-audio",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to the correct data/ subdirectory."""
    extension    = uploaded_file.name.rsplit(".", 1)[-1].lower()
    target_dir   = UPLOAD_DIRS.get(extension, "data/documents")
    os.makedirs(target_dir, exist_ok=True)
    target_path  = os.path.join(target_dir, uploaded_file.name)

    with open(target_path, "wb") as output_file:
        shutil.copyfileobj(uploaded_file, output_file)

    return target_path


def render_source_card(source: dict) -> None:
    """Render a single source citation card."""
    badge_class = BADGE_CLASS.get(source["modality"], "badge-text")
    st.markdown(f"""
    <div class="source-card">
      <span class="source-badge {badge_class}">{source['modality']}</span>
      <span class="source-filename">{source['file']}</span>
      <span class="source-score">score {source['score']:.2f}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar — Section 1: Load Data ────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">01 — Load Data</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Drop files here",
        accept_multiple_files=True,
        type=list(UPLOAD_DIRS.keys()),
        label_visibility="collapsed",
    )

    if uploaded_files:
        pills_html = "".join(
            f'<span class="file-pill">{f.name}</span>' for f in uploaded_files
        )
        st.markdown(pills_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    ingest_clicked = st.button("Ingest Files", disabled=not uploaded_files)

    if ingest_clicked and uploaded_files:
        saved_paths = []
        progress_bar = st.progress(0, text="Saving files...")

        for idx, uploaded_file in enumerate(uploaded_files):
            path = save_uploaded_file(uploaded_file)
            saved_paths.append(path)
            progress_bar.progress(
                (idx + 1) / len(uploaded_files),
                text=f"Saved {uploaded_file.name}",
            )

        progress_bar.progress(100, text="Ingesting into vector DB...")

        try:
            ingest()
            progress_bar.empty()
            st.success(f"✓ {len(saved_paths)} file(s) ingested")
        except Exception as error:
            progress_bar.empty()
            st.error(f"Ingestion failed: {error}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">02 — Search</div>', unsafe_allow_html=True)
    st.caption("Ask a question in the main panel →")


# ── Main Panel — Section 2: Search ────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="hero-kicker">Multimodal RAG Assistant</span>
  <h1 class="hero-title">Search Across Text, Images, and Audio</h1>
  <p class="hero-sub">Upload your files from the sidebar, ingest once, and ask natural-language questions to retrieve grounded answers with source citations.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Ask Your Documents</div>', unsafe_allow_html=True)

user_question = st.text_input(
    label="Question",
    placeholder="What did the CEO say about layoffs?",
    label_visibility="collapsed",
)

search_clicked = st.button("Search", disabled=not user_question)

if search_clicked and user_question:
    with st.spinner("Retrieving and generating answer..."):
        try:
            result = query(user_question)

            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )

            if result["sources"]:
                st.markdown(
                    '<div class="section-title" style="margin-top:2rem;">Sources</div>',
                    unsafe_allow_html=True,
                )
                for source in result["sources"]:
                    render_source_card(source)

        except Exception as error:
            st.error(f"Query failed: {error}")
