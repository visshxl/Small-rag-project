# 🤖 RAG PDF Chatbot

> Ask questions over **any PDF document** using Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Google AI Studio](https://img.shields.io/badge/Google%20AI%20Studio-Gemini-yellow)
![Chroma](https://img.shields.io/badge/Chroma-Vector%20DB-orange)
![OpenAI SDK](https://img.shields.io/badge/OpenAI%20SDK-compatible-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🧠 What is RAG? (Simple Explanation)

Imagine you have a 100-page PDF. You can't paste the whole thing into ChatGPT. RAG solves this:

```
Your PDF
   │
   ▼
Split into small chunks (500 characters each)
   │
   ▼
Convert each chunk into a number vector (semantic embedding)
   │
   ▼
Store all vectors in Chroma (a persistent local vector database)
   │
   ▼
You ask a question
   │
   ▼
Find the 4 most similar chunks using vector search
   │
   ▼
Send those chunks + your question to an LLM
   │
   ▼
LLM generates an answer based ONLY on those chunks
```

The LLM never sees the full PDF — it only sees the most relevant pieces.
This is why it's fast, accurate, and doesn't hallucinate (make things up).

---

## ⚠️ Honest Limitations (Read This First!)

> This section is important. RAG is powerful but it has real limitations.
> If you test this chatbot and it says "I don't know", that's not always a bug —
> it's often just how RAG works. Here's what to expect:

### ❌ What this chatbot CANNOT do

| Limitation                                    | Why it happens                                      | Example                                                 |
| --------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| **Cannot describe images**                    | RAG only reads text, not pixels                     | "What does the diagram show?" → "I don't know"          |
| **Cannot read scanned PDFs well**             | Scanned PDFs are photos, not text                   | A photographed paper document → garbled output          |
| **Cannot answer from memory**                 | It only knows what's in YOUR PDF                    | "Who is Elon Musk?" → "Not in this document"            |
| **Cannot answer cross-document questions**    | Only one PDF loaded at a time                       | "Compare this with that other file" → won't work        |
| **May miss answers on large PDFs**            | Retrieves only top 4 chunks out of hundreds         | Answer might be in chunk 50 but chunk 12 gets retrieved |
| **Stylized/decorative fonts confuse OCR**     | Tesseract reads design PDFs poorly                  | Presentation PDFs with gradient text → garbled          |
| **Cannot summarize very long PDFs perfectly** | "Summarize everything" needs all chunks, not just 4 | Better to ask specific questions                        |

### ✅ What this chatbot IS good at

- PDFs with real embedded text (research papers, reports, books, resumes)
- Specific factual questions ("Who won?", "What was the conclusion?", "How many X?")
- Finding information buried deep in a long document
- Extracting key points from a section

### 💡 Tips for best results

- Ask **specific questions** instead of vague ones
  - ❌ "Tell me everything"
  - ✅ "What were the main findings in section 3?"
- Use **text-heavy PDFs** — reports, papers, ebooks, resumes work best
- If an answer seems wrong, try **rephrasing the question**
- PDFs exported from Word/Google Docs work better than design tools like Canva

---

## 🏗️ Architecture

```
PDF File (text-based)
   │
   ▼
PyPDFLoader → extracts text per page
   │
   ▼
RecursiveCharacterTextSplitter → 500 char chunks, 50 char overlap
   │
   ▼
Google AI Studio Embeddings (gemini-embedding-001)
   │
   ▼
Chroma Vector Store (persistent local vector database)
   │
   ▼
User asks a question
   │
   ├── Chroma retrieves top 4 relevant chunks
   │
   └── Google AI Studio (Gemini) generates answer through the OpenAI SDK
            │
            ▼
       Answer + Source Pages shown in UI
```

---

## ⚙️ Tech Stack

| Component           | Tool                           | Why chosen                                     |
| ------------------- | ------------------------------ | ---------------------------------------------- |
| PDF text extraction | PyPDFLoader                    | Reads embedded fonts cleanly, no OCR noise     |
| Text splitting      | RecursiveCharacterTextSplitter | Preserves sentence context at chunk boundaries |
| Embeddings          | Google `gemini-embedding-001`  | Managed semantic embeddings for retrieval      |
| Vector database     | Chroma                         | Persistent local vector storage and retrieval  |
| LLM                 | Google Gemini via OpenAI SDK   | Hosted answer generation                       |
| Backend             | Flask                          | Lightweight Python web server                  |
| Frontend            | Custom HTML/CSS/JS             | Animated dark UI, no framework needed          |

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Aaricacoding/rag-pdf-chatbot.git
cd rag-pdf-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a Google AI Studio API key

1. Go to **https://aistudio.google.com/apikey**
2. Create a Gemini API key
3. Copy your key

### 4. Create your `.env` file

```
GOOGLE_API_KEY=your_google_ai_studio_key_here
GOOGLE_LLM_MODEL=gemini-3.8-flash
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
CHROMA_DIR=./chroma_db
CHROMA_COLLECTION_NAME=pdf_documents
GOOGLE_API_TIMEOUT_SECONDS=30
GOOGLE_API_MAX_RETRIES=0
```

> ⚠️ Never commit this file to GitHub. It's already in `.gitignore`.

The Chroma directory is persistent. In Docker, mount `/data/chroma` to a
durable host or Azure managed-disk directory so the index survives container
recreation.

For local Windows development, you can place Chroma outside OneDrive:

```env
CHROMA_DIR=C:/rag-data/chroma
```

The application prints `[TIMING]` lines for query embedding, Chroma retrieval,
Gemini generation, and total answer time.

### 5. Run

```bash
python app.py
```

Open `http://127.0.0.1:8081` in your browser, unless you set `PORT`.

### Docker deployment

Build and run the container with a persistent Chroma directory:

```bash
docker build -t rag-pdf-chatbot .
docker run -p 8081:8081 --env-file .env \
  -v /srv/rag-data/chroma:/data/chroma \
  rag-pdf-chatbot
```

The container listens on `0.0.0.0:8081`. Allow TCP port `8081` in the
Azure VM Network Security Group when hosting it on an Azure VM.

---

## 📖 How to Use

1. **Upload** a text-based PDF (research paper, report, resume, ebook)
2. Click **Index PDF** — wait for ✅ (takes 5–15 seconds for embedding)
3. **Ask a specific question** about the document
4. Get an answer with **source page numbers** cited

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py              # Flask server — API endpoints + serves HTML
├── rag_engine.py       # RAG pipeline — PDF loading, Chroma indexing, QA chain
├── requirements.txt    # Python dependencies
├── .gitignore          # Keeps .env and cache out of git
├── README.md           # This file
└── static/
    └── index.html      # Animated frontend UI
```

---

## 🔧 Customization

| What to change             | Where               | How                                                    |
| -------------------------- | ------------------- | ------------------------------------------------------ |
| Number of chunks retrieved | `rag_engine.py`     | Change `k=4` in `as_retriever()`                       |
| Chunk size                 | `rag_engine.py`     | Adjust `chunk_size=500`                                |
| LLM model                  | `.env`              | Set `GOOGLE_LLM_MODEL` to an available Gemini model   |
| Embedding model             | `.env`              | Set `GOOGLE_EMBEDDING_MODEL` to an available model    |
| UI colors/theme            | `static/index.html` | Edit CSS variables at the top of the file              |

---

## 🏷️ Skills Demonstrated

- **RAG architecture** — end-to-end implementation from scratch
- **Vector databases** — Chroma persistence and semantic similarity search
- **LangChain 0.3** — PDF loaders, chunking, Chroma retrievers, and LCEL integration
- **Google AI Studio** — Gemini embeddings and answer generation
- **OpenAI SDK** — API client used with Google’s compatibility endpoint
- **Flask** — REST API backend with file upload handling
- **Frontend** — animated UI without any JS framework

---

## 📄 License

MIT License — free to use, modify, and distribute.
