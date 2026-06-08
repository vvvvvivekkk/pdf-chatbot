# -*- coding: utf-8 -*-
import os
import re
import gc
import time
import json
import shutil
import warnings
import chromadb
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_and_chunk_pdf,
    load_and_chunk_multiple_pdfs,
    get_embeddings,
    build_vectorstore,
    load_vectorstore,
    build_rag_chain,
    ask,
    summarize_pdf,
    summary_to_pdf_bytes,
    stored_name,   # import helper to normalize filenames
    CHROMA_DIR,
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG PDF ChatBot",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, .stApp, p, h1, h2, h3, h4, h5, h6,
textarea, button, label, input, select,
div[data-testid="stMarkdownContainer"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background: radial-gradient(circle at 50% 50%, #0d111d 0%, #07090e 100%) !important;
    color: #e2e8f0 !important;
}
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #6366f1 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    background: rgba(9, 11, 20, 0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 1rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stChatMessage"]:hover { border-color: rgba(99,102,241,0.3) !important; }
.status-pill {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem;
    border: 1px solid transparent;
}
.status-pill.connected    { background:rgba(52,211,153,0.1);  color:#34d399; border-color:rgba(52,211,153,0.2);  }
.status-pill.disconnected { background:rgba(248,113,113,0.1); color:#f87171; border-color:rgba(248,113,113,0.2); }
.status-pill.active       { background:rgba(129,140,248,0.15);color:#a5b4fc; border-color:rgba(129,140,248,0.25);}
.pdf-tag {
    display: inline-flex; align-items: center;
    background: rgba(99,102,241,0.1); border: 0.5px solid rgba(99,102,241,0.25);
    border-radius: 8px; padding: 3px 10px; font-size: 12px; color: #a5b4fc; margin: 2px 3px 2px 0;
}
.source-badge {
    display: inline-block; background: rgba(52,211,153,0.08);
    border: 0.5px solid rgba(52,211,153,0.2); border-radius: 6px;
    padding: 2px 8px; font-size: 11px; color: #34d399; margin: 2px 2px 0 0;
}
.summary-box {
    background: rgba(99,102,241,0.05); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px; padding: 1.2rem 1.4rem; margin: 1rem 0;
}
[data-testid="stChatInput"] { border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 14px !important; }
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; width: 100%; transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(79,70,229,0.4) !important; }
[data-testid="stFileUploader"] { border: 1px dashed rgba(255,255,255,0.1) !important; border-radius: 12px !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

for key, default in {
    "chain":             None,
    "chat_history":      [],
    "pdf_name":          None,
    "embeddings":        None,
    "loaded_from_cache": False,
    "loaded_pdfs":       [],
    "filter_pdf":        "All PDFs",
    "multi_pdf_mode":    False,
    "vectorstore":       None,
    "summary":           None,
    "active_chroma_dir": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────

os.makedirs(CHROMA_DIR, exist_ok=True)
METADATA_PATH    = os.path.join(CHROMA_DIR, "cache_metadata.json")
MULTI_PDF_CHROMA = os.path.join(CHROMA_DIR, "multi_pdf_store")

def load_cache_metadata() -> dict:
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_cache_metadata(safe_name: str, original_name: str):
    meta = load_cache_metadata()
    meta[safe_name] = original_name
    try:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save cache metadata: {e}")

def safe_folder_name(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", filename).lower()

def safe_delete_chroma(path: str):
    if not os.path.exists(path):
        return
    try:
        st.session_state.chain             = None
        st.session_state.vectorstore       = None
        st.session_state.active_chroma_dir = None
        try:
            client = chromadb.PersistentClient(path=path)
            for col in client.list_collections():
                client.delete_collection(col.name)
            del client
        except Exception:
            pass
    except Exception:
        pass
    gc.collect()
    time.sleep(1.0)
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            break
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
            gc.collect()

def get_vectorstore(persist_dir: str):
    """Returns live vectorstore — reloads from disk if lost on Streamlit rerun."""
    if (st.session_state.vectorstore is not None and
            st.session_state.active_chroma_dir == persist_dir):
        return st.session_state.vectorstore
    print(f"[INFO] Reloading vectorstore from {persist_dir}")
    vs = load_vectorstore(st.session_state.embeddings, persist_directory=persist_dir)
    st.session_state.vectorstore       = vs
    st.session_state.active_chroma_dir = persist_dir
    return vs

def get_active_chroma_dir() -> str:
    if st.session_state.multi_pdf_mode:
        return MULTI_PDF_CHROMA
    elif st.session_state.pdf_name:
        return os.path.join(CHROMA_DIR, safe_folder_name(st.session_state.pdf_name))
    return None

def render_sources(sources):
    if not sources:
        return
    badges = []
    for s in sources:
        if isinstance(s, dict):
            page     = s.get("page", "?")
            file     = s.get("file", "unknown")
            page_num = (page + 1) if isinstance(page, int) else page
            badges.append(f'<span class="source-badge">📖 {file} p.{page_num}</span>')
        else:
            page_num = (s + 1) if isinstance(s, int) else s
            badges.append(f'<span class="source-badge">📖 p.{page_num}</span>')
    st.markdown(" ".join(badges), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 RAG Bot Settings")
    st.markdown("Configure your document and model parameters.")
    st.divider()

    # API Key
    st.subheader("🔑 API Configuration")
    env_key = os.getenv("MISTRAL_API_KEY")
    if env_key:
        st.markdown('<div class="status-pill connected">🟢 Mistral Connected (.env)</div>', unsafe_allow_html=True)
        active_api_key = env_key
    else:
        user_key = st.text_input("Enter Mistral API Key", type="password", placeholder="Paste your key here...")
        if user_key:
            st.markdown('<div class="status-pill connected">🟢 API Key Provided</div>', unsafe_allow_html=True)
            active_api_key = user_key
        else:
            st.markdown('<div class="status-pill disconnected">🔴 Missing Mistral Key</div>', unsafe_allow_html=True)
            active_api_key = None

    st.divider()

    # File Upload
    st.subheader("📂 Upload Document(s)")
    st.caption("Upload one or multiple PDFs.")
    uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if not active_api_key:
            st.error("⚠️ Please configure a Mistral API key first.")
        else:
            current_names = sorted([f.name for f in uploaded_files])
            is_new        = current_names != sorted(st.session_state.loaded_pdfs)

            if is_new:
                st.session_state.chat_history      = []
                st.session_state.chain             = None
                st.session_state.vectorstore       = None
                st.session_state.active_chroma_dir = None
                st.session_state.loaded_pdfs       = []
                st.session_state.pdf_name          = None
                st.session_state.filter_pdf        = "All PDFs"
                st.session_state.multi_pdf_mode    = len(uploaded_files) > 1
                st.session_state.summary           = None

                pdf_paths = []
                for uf in uploaded_files:
                    path = f"uploaded_{uf.name}"
                    with open(path, "wb") as f:
                        f.write(uf.read())
                    pdf_paths.append(path)

                if st.session_state.embeddings is None:
                    with st.spinner("Loading embedding model..."):
                        st.session_state.embeddings = get_embeddings()

                if len(pdf_paths) == 1:
                    safe_name  = safe_folder_name(uploaded_files[0].name)
                    pdf_chroma = os.path.join(CHROMA_DIR, safe_name)
                    is_cached  = os.path.exists(pdf_chroma)

                    if is_cached:
                        with st.spinner("Loading cached DB..."):
                            try:
                                vs = load_vectorstore(st.session_state.embeddings, persist_directory=pdf_chroma)
                                st.session_state.chain             = build_rag_chain(vs, mistral_api_key=active_api_key)
                                st.session_state.vectorstore       = vs
                                st.session_state.active_chroma_dir = pdf_chroma
                                st.session_state.loaded_from_cache = True
                            except Exception as e:
                                st.warning(f"Cache load failed ({e}). Re-embedding...")
                                shutil.rmtree(pdf_chroma, ignore_errors=True)
                                is_cached = False

                    if not is_cached:
                        with st.spinner("Processing PDF..."):
                            try:
                                chunks = load_and_chunk_pdf(pdf_paths[0])
                                vs     = build_vectorstore(chunks, st.session_state.embeddings, persist_directory=pdf_chroma)
                                st.session_state.chain             = build_rag_chain(vs, mistral_api_key=active_api_key)
                                st.session_state.vectorstore       = vs
                                st.session_state.active_chroma_dir = pdf_chroma
                                st.session_state.loaded_from_cache = False
                                save_cache_metadata(safe_name, uploaded_files[0].name)
                            except ValueError as e:
                                st.error(f"📄 PDF Error: {e}")
                            except Exception as e:
                                st.error(f"❌ Unexpected error: {e}")

                    st.session_state.pdf_name = uploaded_files[0].name

                else:
                    with st.spinner(f"Processing {len(pdf_paths)} PDFs..."):
                        try:
                            safe_delete_chroma(MULTI_PDF_CHROMA)
                            all_chunks, failed = load_and_chunk_multiple_pdfs(pdf_paths)
                            for fname, err in failed:
                                st.warning(f"⚠️ Skipped **{fname}**: {err}")
                            vs = build_vectorstore(all_chunks, st.session_state.embeddings, persist_directory=MULTI_PDF_CHROMA)
                            st.session_state.chain             = build_rag_chain(vs, mistral_api_key=active_api_key)
                            st.session_state.vectorstore       = vs
                            st.session_state.active_chroma_dir = MULTI_PDF_CHROMA
                            st.session_state.loaded_from_cache = False
                        except ValueError as e:
                            st.error(f"📄 Error: {e}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {e}")

                if st.session_state.chain:
                    st.session_state.loaded_pdfs = current_names

            if st.session_state.chain:
                if st.session_state.multi_pdf_mode:
                    st.success(f"✅ {len(st.session_state.loaded_pdfs)} PDFs ready!")
                    st.markdown("**Loaded documents:**")
                    for name in st.session_state.loaded_pdfs:
                        st.markdown(f'<span class="pdf-tag">📄 {name}</span>', unsafe_allow_html=True)
                else:
                    tag = "⚡ Instant Cache" if st.session_state.loaded_from_cache else "✅ Newly Embedded"
                    st.success(f"{tag} — **{st.session_state.loaded_pdfs[0]}** ready!")

    if st.session_state.multi_pdf_mode and len(st.session_state.loaded_pdfs) > 1:
        st.divider()
        st.subheader("🔎 Filter by Document")
        options  = ["All PDFs"] + st.session_state.loaded_pdfs
        selected = st.selectbox("Answer questions from:", options, index=0)
        if selected != st.session_state.filter_pdf:
            st.session_state.filter_pdf   = selected
            st.session_state.chat_history = []
            st.session_state.summary      = None
            if os.path.exists(MULTI_PDF_CHROMA):
                with st.spinner("Applying filter..."):
                    vs            = load_vectorstore(st.session_state.embeddings, persist_directory=MULTI_PDF_CHROMA)
                    filter_source = None if selected == "All PDFs" else selected
                    st.session_state.chain             = build_rag_chain(vs, mistral_api_key=active_api_key, filter_source=filter_source)
                    st.session_state.vectorstore       = vs
                    st.session_state.active_chroma_dir = MULTI_PDF_CHROMA

    cache_meta = load_cache_metadata()
    if cache_meta:
        st.divider()
        st.subheader("🗄️ Cached Documents")
        st.caption("These load instantly next time:")
        for _, orig in cache_meta.items():
            st.markdown(f"• {orig}")

    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ── Main Chat Area ────────────────────────────────────────────────────────────

st.title("📄 RAG PDF ChatBot")

if st.session_state.loaded_pdfs and st.session_state.chain:
    if st.session_state.multi_pdf_mode:
        flt = st.session_state.filter_pdf
        st.markdown(f'<div class="status-pill active">📄 {len(st.session_state.loaded_pdfs)} PDFs &nbsp;·&nbsp; 🔎 {flt}</div>', unsafe_allow_html=True)
    else:
        tag = "⚡ Cache" if st.session_state.loaded_from_cache else "✅ Embedded"
        st.markdown(f'<div class="status-pill active">📄 {st.session_state.pdf_name} &nbsp;·&nbsp; {tag}</div>', unsafe_allow_html=True)
else:
    st.caption("Upload a PDF from the sidebar to begin chatting.")

# ── Summarization ─────────────────────────────────────────────────────────────

if st.session_state.chain and st.session_state.loaded_pdfs:
    st.divider()

    if st.session_state.multi_pdf_mode:
        flt        = st.session_state.filter_pdf
        btn_label  = f"📝 Summarize — {flt}"
        sum_target = None if flt == "All PDFs" else flt
    else:
        btn_label  = f"📝 Summarize — {st.session_state.pdf_name}"
        sum_target = st.session_state.pdf_name  # plain name — stored_name() applied inside summarize_pdf

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Quick Summary** — get an overview of your document(s) instantly.")
    with col2:
        do_summarize = st.button(btn_label)

    if do_summarize:
        st.session_state.summary = None

    if do_summarize or st.session_state.summary:
        if st.session_state.summary is None:
            with st.spinner("Generating summary... (may take ~20 seconds)"):
                try:
                    active_dir = get_active_chroma_dir()
                    if not active_dir or not os.path.exists(active_dir):
                        st.error("❌ No document database found. Please re-upload your PDF.")
                    else:
                        vs = get_vectorstore(active_dir)
                        summary, pages = summarize_pdf(
                            vs,
                            mistral_api_key=active_api_key,
                            filename=sum_target,  # normalized inside summarize_pdf
                        )
                        st.session_state.summary = {"text": summary, "pages": pages}
                except Exception as e:
                    st.error(f"❌ Summary error: {e}")
                    st.session_state.summary = None

        if st.session_state.summary:
            with st.container():
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.summary["text"])
                pages     = st.session_state.summary["pages"]
                page_nums = [p + 1 if isinstance(p, int) else p for p in pages]
                st.caption(f"📖 Based on pages: {page_nums}")
                st.markdown('</div>', unsafe_allow_html=True)
                pdf_bytes = summary_to_pdf_bytes(
                    st.session_state.summary["text"],
                    pdf_name=st.session_state.pdf_name or "documents",
                )
                st.download_button(
                    label="⬇️ Download Summary as PDF",
                    data=pdf_bytes,
                    file_name=f"summary_{st.session_state.pdf_name or 'documents'}.pdf",
                    mime="application/pdf",
                )

    st.divider()

# ── Chat History ──────────────────────────────────────────────────────────────

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# ── Chat Input ────────────────────────────────────────────────────────────────

if st.session_state.chain is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    placeholder = (
        f"Ask about {st.session_state.filter_pdf.lower()}..."
        if st.session_state.filter_pdf != "All PDFs"
        else "Ask a question about your PDF(s)..."
    )
    user_input = st.chat_input(placeholder)

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        lc_history = []
        for m in st.session_state.chat_history[:-1]:
            role = "human" if m["role"] == "user" else "ai"
            lc_history.append((role, m["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = ask(st.session_state.chain, user_input, chat_history=lc_history)
                    st.markdown(answer)
                    render_sources(sources)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": answer, "sources": sources,
                    })
                except Exception as e:
                    err_msg = f"❌ Error getting answer: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": err_msg, "sources": []
                    })