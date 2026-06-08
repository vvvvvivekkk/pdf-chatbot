# 📄 PDF ChatBot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that lets users upload one or multiple PDFs and chat with them using Mistral AI. Built with LangChain, ChromaDB, HuggingFace Embeddings, and Streamlit.

## 🚀 Features

* 📂 Multi-PDF Support — upload and chat across multiple documents simultaneously
* 🔎 Per-document filtering — restrict answers to a specific PDF
* 🧠 Conversational memory — follow-up questions work naturally across the chat
* 📝 One-click summarization — structured 5-section summary with PDF download
* ⚡ Smart caching — re-uploaded PDFs load instantly from saved vectorstore
* 🎯 Source attribution — every answer shows which file and page it came from
* 🌙 Premium dark UI — clean Streamlit interface with custom styling

## 🛠️ Tech Stack

| Layer          | Tool                              |
| -------------- | --------------------------------- |
| LLM            | Mistral AI (mistral-small-latest) |
| Embeddings     | HuggingFace all-MiniLM-L6-v2      |
| Vector DB      | ChromaDB                          |
| Framework      | LangChain                         |
| PDF Processing | PyPDF + pdfplumber                |
| UI             | Streamlit                         |
| Export         | fpdf2                             |

## 📁 Project Structure

```text
pdf-chatbot/
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── .env
├── .gitignore
├── chroma_db/
└── uploaded_*.pdf
```

## ⚙️ Setup

```bash
git clone https://github.com/vvvvvivekkk/pdf-chatbot.git
cd pdf-chatbot

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key_here
```

Run:

```bash
streamlit run app.py
```

## 🧠 How It Works

PDF Upload
↓
Chunking
↓
HuggingFace Embeddings
↓
ChromaDB Vector Store
↓
History-Aware Retrieval
↓
Mistral AI
↓
Answer + Sources

## 📄 License

MIT License

## 👨‍💻 Author

**Vivek**

GitHub: https://github.com/vvvvvivekkk
LinkedIn: https://linkedin.com/in/your-linkedin-profile

---

If this project helped you, consider giving it a ⭐ on GitHub.
