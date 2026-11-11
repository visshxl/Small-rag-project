# app.py — Flask server with Google AI Studio RAG backend

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv  # loads .env file into os.environ automatically
import os
import tempfile
from rag_engine import (
    build_qa_chain,
    load_and_index_pdf,
    load_persisted_vector_store,
)

load_dotenv()  # loads Google AI Studio credentials before the RAG engine is used

app = Flask(__name__, static_folder="static")

qa_chain = None


def restore_persisted_index():
    """Restore the active Chroma collection when the app starts."""
    global qa_chain

    try:
        vector_store = load_persisted_vector_store()
        if vector_store is not None:
            qa_chain, _ = build_qa_chain(vector_store)
            print("[INFO] Restored persisted Chroma collection")
    except Exception as e:
        # The app can still start and allow a fresh upload if restoration fails.
        print(f"[WARN] Could not restore persisted Chroma collection: {e}")


restore_persisted_index()


@app.route("/")
def index():
    """Serve the main HTML UI."""
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """POST /upload — indexes the uploaded PDF and prepares QA chain."""
    global qa_chain

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "message": "Empty filename"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "message": "Only PDF files accepted"}), 400

    tmp_path = None
    try:
        # Save to temp file so LangChain's PyPDFLoader can read it from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Build persistent Chroma index using Google AI Studio embeddings
        vector_store = load_and_index_pdf(tmp_path)

        # Build the Google AI Studio QA chain
        qa_chain, _ = build_qa_chain(vector_store)

        return jsonify({"success": True, "message": "PDF indexed successfully!"})

    except ValueError as e:
        # Catches missing Google AI Studio credentials with a helpful message
        return jsonify({"success": False, "message": str(e)}), 400

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/status", methods=["GET"])
def index_status():
    """Return whether a PDF index is currently available."""
    return jsonify({"success": True, "indexed": qa_chain is not None})


@app.route("/ask", methods=["POST"])
def ask_question():
    """POST /ask — runs RAG pipeline and returns answer."""
    global qa_chain

    if qa_chain is None:
        return jsonify({"success": False, "message": "Please upload a PDF first"}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"success": False, "message": "No question provided"}), 400

    if not isinstance(data["question"], str):
        return jsonify({"success": False, "message": "Question must be text"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"success": False, "message": "Question cannot be empty"}), 400

    try:
        result = qa_chain.invoke(question)
        return jsonify({
            "success": True,
            "answer": result["answer"],
            "pages": result["pages"],
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    port = int(os.environ.get("PORT", 8081))
    print(f"\n🚀 RAG PDF Chatbot running at: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port)
