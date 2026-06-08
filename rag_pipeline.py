# -*- coding: utf-8 -*-
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR    = "chroma_db"
MISTRAL_MODEL = "mistral-small-latest"

# ── Helper: normalize filename to match stored metadata ──────────────────────

def stored_name(filename: str) -> str:
    """
    ChromaDB stores source_file as 'uploaded_<filename>'.
    This ensures any filename passed always matches what's stored.
    """
    if filename and not filename.startswith("uploaded_"):
        return f"uploaded_{filename}"
    return filename

# ── Step 1: Load & chunk PDF ──────────────────────────────────────────────────

def load_and_chunk_pdf(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    print(f"  [OK] Loaded {len(pages)} pages")

    # Tag every chunk with the full path basename (includes uploaded_ prefix)
    filename = os.path.basename(pdf_path)
    for page in pages:
        page.metadata["source_file"] = filename

    empty_indices = [i for i, p in enumerate(pages) if not p.page_content.strip()]
    if empty_indices:
        print(f"  [WARN] {len(empty_indices)} empty page(s) — trying pdfplumber...")
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for i in empty_indices:
                    if i < len(pdf.pages):
                        text = pdf.pages[i].extract_text() or ""
                        if text.strip():
                            pages[i].page_content = text
        except ImportError:
            print("  [WARN] pdfplumber not installed.")
        except Exception as e:
            print(f"  [WARN] pdfplumber error: {e}")

    all_text = " ".join(p.page_content for p in pages).strip()
    if not all_text:
        raise ValueError(
            "This PDF has no extractable text. It may be a scanned image-only PDF.\n"
            "Please use a PDF with selectable text, or run OCR first."
        )

    all_chunks = []
    for chunk_size, chunk_overlap in [(1200, 200), (400, 80)]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        all_chunks.extend([c for c in chunks if c.page_content.strip()])

    seen, unique = set(), []
    for c in all_chunks:
        key = c.page_content.strip()[:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    print(f"  [OK] {len(unique)} unique chunks")
    if not unique:
        raise ValueError("PDF loaded but produced no usable chunks.")
    return unique


def load_and_chunk_multiple_pdfs(pdf_paths: list):
    all_chunks, failed = [], []
    for path in pdf_paths:
        try:
            chunks = load_and_chunk_pdf(path)
            all_chunks.extend(chunks)
        except ValueError as e:
            print(f"  [SKIP] {os.path.basename(path)}: {e}")
            failed.append((os.path.basename(path), str(e)))
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(path)}: {e}")
            failed.append((os.path.basename(path), str(e)))

    if not all_chunks:
        raise ValueError("No usable text found in any of the uploaded PDFs.")
    print(f"\n  [OK] Total chunks across all PDFs: {len(all_chunks)}")
    return all_chunks, failed

# ── Step 2: Embeddings & ChromaDB ─────────────────────────────────────────────

def get_embeddings():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("  [OK] Embedding model ready")
    return embeddings


def build_vectorstore(chunks, embeddings, persist_directory=CHROMA_DIR):
    print(f"Embedding {len(chunks)} chunks → ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"  [OK] Vectorstore saved to '{persist_directory}'")
    return vectorstore


def load_vectorstore(embeddings, persist_directory=CHROMA_DIR):
    print(f"Loading cached vectorstore from '{persist_directory}'...")
    vs    = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    count = vs._collection.count()
    if count == 0:
        raise ValueError(f"Cached vectorstore at '{persist_directory}' is empty.")
    print(f"  [OK] Loaded {count} vectors from cache")
    return vs

# ── Step 3: Conversational RAG chain ─────────────────────────────────────────

CONTEXTUALIZE_PROMPT = (
    "Given the chat history and the latest user question, which may reference "
    "earlier context, rewrite it as a fully standalone question. "
    "Do NOT answer — just reformulate if needed, otherwise return as-is."
)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about uploaded documents.\n\n"
    "Guidelines:\n"
    "- If the user sends a greeting or pleasantry, respond conversationally.\n"
    "- For document questions: use the context chunks provided below to answer.\n"
    "- If answers span multiple documents, mention which document each part came from.\n"
    "- The context may contain the answer in different wording — read carefully.\n"
    "- If you can partially answer, do so and mention what's missing.\n"
    "- Only say \"I couldn't find that in the document\" if the topic is genuinely absent.\n"
    "- Be concise, clear, and never fabricate information.\n\n"
    "Context:\n{context}"
)


def build_rag_chain(vectorstore, mistral_api_key=None, filter_source=None):
    search_kwargs = {"k": 8}
    if filter_source:
        # Normalize to stored name (uploaded_ prefix)
        search_kwargs["filter"] = {"source_file": stored_name(filter_source)}

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY is missing. Add it to your .env file.")

    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=key,
        temperature=0.3,
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    qa_chain  = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    print("  [OK] RAG chain ready")
    return rag_chain


def ask(chain, question: str, chat_history=None):
    lc_history = []
    for role, content in (chat_history or []):
        if role == "human":
            lc_history.append(HumanMessage(content=content))
        elif role == "ai":
            lc_history.append(AIMessage(content=content))

    result = chain.invoke({"input": question, "chat_history": lc_history})
    answer = result.get("answer", "No answer returned.")

    sources = []
    if "context" in result:
        seen = set()
        for doc in result["context"]:
            page = doc.metadata.get("page", "?")
            file = doc.metadata.get("source_file", "unknown")
            key  = (file, page)
            if key not in seen:
                seen.add(key)
                sources.append({"file": file, "page": page})

    return answer, sources


# ── Summarization ─────────────────────────────────────────────────────────────

SUMMARY_PROMPT = """You are a document analyst. Analyze the provided document chunks and generate a comprehensive structured summary.

Your summary MUST follow this exact format:

## 📋 Overview
[2-3 sentences describing what this document is about]

## 🎯 Main Topics
[Bullet points of the key subjects covered]

## 💡 Key Points
[The most important facts, findings, or arguments — bullet points]

## 📊 Important Details
[Specific data, numbers, dates, names, or technical details mentioned]

## 🔚 Conclusion
[What the document concludes or recommends, if applicable]

Document content:
{context}

Generate the structured summary now:"""


def summarize_pdf(vectorstore, mistral_api_key=None, filename=None):
    """
    Generate a structured summary.
    filename: the display name (with or without uploaded_ prefix) — normalized internally.
    """
    key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY is missing.")

    docs, metas = [], []

    try:
        if filename:
            # Normalize to match stored metadata (uploaded_ prefix)
            sname = stored_name(filename)
            print(f"  [INFO] Summarizing with filter: source_file = '{sname}'")

            # Try filtered fetch using LangChain's vectorstore.get()
            try:
                results = vectorstore.get(
                    where={"source_file": sname},
                    include=["documents", "metadatas"],
                )
                docs  = results.get("documents", []) or []
                metas = results.get("metadatas", []) or []
            except Exception as e:
                print(f"  [WARN] Filtered get failed: {e} — falling back to manual filter")
                results = vectorstore.get(include=["documents", "metadatas"])
                all_docs  = results.get("documents", []) or []
                all_metas = results.get("metadatas", []) or []
                # Manual filter — match stored name
                paired_all = [
                    (d, m) for d, m in zip(all_docs, all_metas)
                    if m and m.get("source_file") == sname and d and d.strip()
                ]
                docs  = [p[0] for p in paired_all]
                metas = [p[1] for p in paired_all]

            # If still empty — fetch everything (no filter)
            if not docs:
                print(f"  [WARN] Filter returned 0 docs — fetching all without filter")
                results = vectorstore.get(include=["documents", "metadatas"])
                docs  = results.get("documents", []) or []
                metas = results.get("metadatas", []) or []
        else:
            # No filter — get all
            results = vectorstore.get(include=["documents", "metadatas"])
            docs  = results.get("documents", []) or []
            metas = results.get("metadatas", []) or []

    except Exception as e:
        raise ValueError(f"Could not fetch content from vectorstore: {e}")

    # Filter empty docs
    paired = [(d, m) for d, m in zip(docs, metas) if d and d.strip()]

    if not paired:
        raise ValueError(
            "No content found. Please re-upload the PDF and try again."
        )

    # Sort by page for logical reading order
    paired.sort(key=lambda x: x[1].get("page", 0) if x[1] else 0)

    # Sample evenly — up to 20 chunks spread across the document
    if len(paired) > 20:
        step   = max(1, len(paired) // 20)
        paired = paired[::step][:20]

    print(f"  [OK] Summarizing using {len(paired)} chunks")

    # Build context
    context_parts = []
    for text, meta in paired:
        page     = meta.get("page", "?") if meta else "?"
        source   = meta.get("source_file", "") if meta else ""
        page_num = (page + 1) if isinstance(page, int) else page
        label    = f"[{source} — Page {page_num}]" if source else f"[Page {page_num}]"
        context_parts.append(f"{label}\n{text.strip()}")

    context = "\n\n---\n\n".join(context_parts)

    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=key,
        temperature=0.3,
    )

    response = llm.invoke(SUMMARY_PROMPT.format(context=context))
    summary  = response.content if hasattr(response, "content") else str(response)

    pages_covered = sorted(
        {m.get("page", "?") for _, m in paired if m},
        key=lambda x: (isinstance(x, str), x)
    )

    return summary, pages_covered

def summary_to_pdf_bytes(summary_text: str, pdf_name: str = "document") -> bytes:
    """Convert summary markdown text to PDF bytes for download."""
    from fpdf import FPDF
    import re

    def clean(text: str) -> str:
        """Convert text to latin-1 safe string."""
        text = text.replace("•", "-").replace("\u2022", "-")
        text = text.replace("\u2013", "-").replace("\u2014", "--")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2026", "...").replace("\u00a0", " ")
        # Strip emojis and any remaining non-latin1 chars
        text = re.sub(r'[^\x00-\xFF]', '', text)
        text = text.encode("latin-1", errors="ignore").decode("latin-1")
        return text.strip()

    pdf = FPDF()
    pdf.set_margins(left=15, top=15, right=15)  # consistent margins
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Page width for multi_cell = total - left - right margins
    pw = pdf.w - 30  # 210 - 15 - 15 = 180mm usable width

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(80, 80, 200)
    pdf.multi_cell(pw, 10, clean(f"Summary: {pdf_name}"))
    pdf.ln(2)

    # Divider
    pdf.set_draw_color(80, 80, 200)
    pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
    pdf.ln(5)

    # Process each line
    for line in summary_text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(2)
            continue

        # Section header: ## Title
        if line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(80, 80, 200)
            pdf.ln(2)
            pdf.multi_cell(pw, 8, clean(line[3:]))
            pdf.ln(1)

        # Bullet point: - item or • item
        elif line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(40, 40, 40)
            # Indent bullet with left margin shift — use set_x safely
            pdf.set_x(20)
            pdf.multi_cell(pw - 5, 6, clean(f"- {line[2:]}"))

        elif line.startswith("\u2022 "):
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(20)
            pdf.multi_cell(pw - 5, 6, clean(f"- {line[2:]}"))

        # Bold line: **text**
        elif line.startswith("**") and line.endswith("**"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(pw, 7, clean(line[2:-2]))

        # Normal text
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(40, 40, 40)
            c = clean(line)
            if c:
                pdf.multi_cell(pw, 7, c)

    return bytes(pdf.output())

# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

    chunks      = load_and_chunk_pdf(pdf_path)
    embeddings  = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    chain       = build_rag_chain(vectorstore)

    print("\n" + "=" * 50)
    history = []
    while True:
        q = input("\nAsk a question (or 'summarize' / 'quit'): ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q.lower() == "summarize":
            print("\nGenerating summary...")
            summary, pages = summarize_pdf(vectorstore)
            print(f"\n{summary}")
            print(f"\nPages sampled: {pages}")
            continue
        answer, sources = ask(chain, q, history)
        print(f"\nAnswer: {answer}")
        print(f"Sources: {sources}")
        history.append(("human", q))
        history.append(("ai", answer))