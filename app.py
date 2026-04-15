import os
import shutil
import streamlit as st
from main import ingest_with_progress, query

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0e0e0e;
    color: #e8e8e8;
  }

  section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
    padding-top: 2rem;
  }

  .block-container { padding: 2rem 3rem; max-width: 900px; }

  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2a2a;
  }

  .answer-box {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #e8e8e8;
    border-radius: 4px;
    padding: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.75;
    white-space: pre-wrap;
    margin: 1.5rem 0;
  }

  .source-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
  }

  .source-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.2rem 0.55rem;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .badge-text  { background: #1e3a2e; color: #4ade80; }
  .badge-image { background: #1e2a3a; color: #60a5fa; }
  .badge-audio { background: #2e1e3a; color: #c084fc; }

  .source-filename {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #aaa;
    flex: 1;
  }

  .source-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #555;
  }

  .log-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #888;
    padding: 0.15rem 0;
    border-bottom: 1px solid #1a1a1a;
  }

  .log-line.done { color: #4ade80; }

  .file-pill {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 0.15rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    margin: 0.2rem;
  }

  .stButton > button {
    background: #e8e8e8 !important;
    color: #0e0e0e !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.4rem !important;
  }

  .stTextInput > div > div > input {
    background: #141414 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# -- Constants -----------------------------------------------------------------
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

MODALITY_ICON = {
    "text":  "📄",
    "image": "🖼️",
    "audio": "🎵",
}


# -- Helpers -------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to the correct data/ subdirectory."""
    extension  = uploaded_file.name.rsplit(".", 1)[-1].lower()
    target_dir = UPLOAD_DIRS.get(extension, "data/documents")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, uploaded_file.name)
    with open(target_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return target_path


def render_source_card(source: dict) -> None:
    """Render a single source citation card."""
    badge_class = BADGE_CLASS.get(source["modality"], "badge-text")
    icon        = MODALITY_ICON.get(source["modality"], "📄")
    st.markdown(f"""
    <div class="source-card">
      <span class="source-badge {badge_class}">{source['modality']}</span>
      <span class="source-filename">{icon} {source['file']}</span>
      <span class="source-score">score: {source['score']:.2f}</span>
    </div>
    """, unsafe_allow_html=True)


# -- Sidebar: Section 1 — Load Data -------------------------------------------
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

        # Save all uploaded files to disk first
        for uploaded_file in uploaded_files:
            save_uploaded_file(uploaded_file)

        # Progress bar + live log area
        progress_bar = st.progress(0.0, text="Starting ingestion...")
        log_area     = st.empty()
        log_lines    = []

        try:
            for update in ingest_with_progress():
                status  = update[0]
                message = update[1]

                if status == "warning":
                    progress_bar.empty()
                    st.warning(message)
                    break

                fraction = update[2]
                progress_bar.progress(fraction, text=message)

                # Append to live log — show last 8 lines to avoid overflow
                css_class = "log-line done" if status == "done" else "log-line"
                log_lines.append(f'<div class="{css_class}">{message}</div>')
                log_area.markdown(
                    "".join(log_lines[-8:]),
                    unsafe_allow_html=True,
                )

            progress_bar.empty()
            log_area.empty()
            st.success(f"✓ Ingestion complete")

        except Exception as error:
            progress_bar.empty()
            st.error(f"Ingestion failed: {error}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">02 — Search</div>', unsafe_allow_html=True)
    st.caption("Ask a question in the main panel →")


# -- Main Panel: Section 2 — Search -------------------------------------------
st.markdown('<div class="section-title">Ask Your Documents</div>', unsafe_allow_html=True)

user_question = st.text_input(
    label="Question",
    placeholder="What does this document say about...?",
    label_visibility="collapsed",
)

search_clicked = st.button("Search", disabled=not user_question)

if search_clicked and user_question:
    with st.spinner("Retrieving relevant chunks and generating answer..."):
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
