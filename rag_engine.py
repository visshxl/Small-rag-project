# rag_engine.py — RAG pipeline using the OpenAI SDK with Google AI Studio

import os
import time
from typing import List

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter


GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_LLM_MODEL = "gemini-3.8-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_CHROMA_DIR = "./chroma_db"
DEFAULT_CHROMA_COLLECTION = "pdf_documents"


def get_google_client() -> OpenAI:
    """Create an OpenAI SDK client configured for Google AI Studio."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Add your Google AI Studio key to .env."
        )

    timeout_seconds = float(os.environ.get("GOOGLE_API_TIMEOUT_SECONDS", "30"))
    max_retries = int(os.environ.get("GOOGLE_API_MAX_RETRIES", "0"))

    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("GOOGLE_BASE_URL", GOOGLE_OPENAI_BASE_URL),
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def _chroma_settings():
    return (
        os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR),
        os.environ.get("CHROMA_COLLECTION_NAME", DEFAULT_CHROMA_COLLECTION),
    )


def _collection_exists(chroma_dir: str, collection_name: str) -> bool:
    import chromadb

    if not os.path.isdir(chroma_dir):
        return False

    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection_names = {
        collection.name if hasattr(collection, "name") else collection
        for collection in chroma_client.list_collections()
    }
    return collection_name in collection_names


class GoogleAIStudioEmbeddings(Embeddings):
    """LangChain Embeddings adapter backed by Google AI Studio via OpenAI SDK."""

    def __init__(self, client: OpenAI, model: str, batch_size: int = 100):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        embedding_started = time.perf_counter()

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            indexed_items = list(enumerate(response.data))

            def response_index(pair):
                index = getattr(pair[1], "index", None)
                return index if isinstance(index, int) else pair[0]

            items = sorted(indexed_items, key=response_index)
            vectors.extend(item.embedding for _, item in items)

        print(
            f"[TIMING] Document embeddings: "
            f"{time.perf_counter() - embedding_started:.2f}s "
            f"for {len(texts)} chunks"
        )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        embedding_started = time.perf_counter()
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        print(
            f"[TIMING] Query embedding: "
            f"{time.perf_counter() - embedding_started:.2f}s"
        )
        return response.data[0].embedding


def load_and_index_pdf(pdf_path: str) -> Chroma:
    """Extract, split, embed, and persist a PDF in Chroma."""
    print("[INFO] Loading PDF with PyPDFLoader...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    total_text = sum(len(doc.page_content) for doc in documents)
    print(f"[INFO] Extracted {total_text} characters from {len(documents)} pages")

    if total_text < 100:
        raise ValueError(
            "This PDF has almost no extractable text. "
            "It may be a scanned document. OCR support coming soon."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks")

    client = get_google_client()
    embeddings = GoogleAIStudioEmbeddings(
        client=client,
        model=os.environ.get("GOOGLE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )

    chroma_dir, collection_name = _chroma_settings()

    # This learning app supports one active PDF at a time. Remove the old
    # collection before creating the replacement so old documents are not
    # accidentally retrieved with the newly uploaded PDF.
    os.makedirs(chroma_dir, exist_ok=True)
    if _collection_exists(chroma_dir, collection_name):
        import chromadb

        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        chroma_client.delete_collection(collection_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=chroma_dir,
    )
    print(f"[INFO] Chroma collection ready at {chroma_dir}")
    return vector_store


def load_persisted_vector_store():
    """Load the active Chroma collection after an app/container restart."""
    chroma_dir, collection_name = _chroma_settings()
    if not _collection_exists(chroma_dir, collection_name):
        return None

    client = get_google_client()
    embeddings = GoogleAIStudioEmbeddings(
        client=client,
        model=os.environ.get("GOOGLE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    return Chroma(
        collection_name=collection_name,
        persist_directory=chroma_dir,
        embedding_function=embeddings,
    )


def build_qa_chain(vector_store: Chroma):
    """Build a retrieval-and-answer chain using Google AI Studio."""
    client = get_google_client()
    llm_model = os.environ.get("GOOGLE_LLM_MODEL", DEFAULT_LLM_MODEL)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "You are a helpful assistant analyzing a document.\n"
        "Answer the question using ONLY the information in the provided context.\n"
        "If the answer is not in the context, say: "
        "'This information is not in the document.'\n"
        "Be clear, concise, and natural."
    )

    def answer_question(question: str):
        request_started = time.perf_counter()

        retrieval_started = time.perf_counter()
        documents = retriever.invoke(question)
        print(
            f"[TIMING] Chroma retrieval: "
            f"{time.perf_counter() - retrieval_started:.2f}s "
            f"for {len(documents)} chunks"
        )

        context = "\n\n---\n\n".join(
            f"[Page {int(doc.metadata.get('page', 0)) + 1}]\n{doc.page_content}"
            for doc in documents
        )

        generation_started = time.perf_counter()
        response = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        print(
            f"[TIMING] Gemini generation: "
            f"{time.perf_counter() - generation_started:.2f}s"
        )

        answer = response.choices[0].message.content or ""
        pages = sorted(
            {
                int(doc.metadata["page"]) + 1
                for doc in documents
                if "page" in doc.metadata
            }
        )

        print(
            f"[TIMING] Total answer time: "
            f"{time.perf_counter() - request_started:.2f}s"
        )

        return {"answer": answer.strip(), "pages": pages}

    return RunnableLambda(answer_question), retriever
