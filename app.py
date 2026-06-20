import os
from flask import Flask, request, render_template

from utils.pdf_loader import load_pdf, load_txt
from utils.text_splitter import split_text
from utils.embeddings import get_embeddings, get_query_embedding
from utils.rag import VectorStore, generate_answer

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create vector store (FAISS memory)
vector_store = VectorStore()


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# UPLOAD DOCUMENT
# =========================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # STEP 1: Extract text
    if file.filename.endswith(".pdf"):
        text = load_pdf(file_path)
    else:
        text = load_txt(file_path)

    # STEP 2: Split text into chunks
    chunks = split_text(text)

    # STEP 3: Convert chunks into embeddings
    embeddings = get_embeddings(chunks)

    # STEP 4: Store in FAISS vector DB
    vector_store.build_index(embeddings, chunks)

    return "✅ Document uploaded and indexed successfully!"


# =========================
# ASK QUESTION
# =========================
@app.route("/ask", methods=["POST"])
def ask():

    question = request.form["question"]

    query_vector = get_query_embedding(question)

    chunks = vector_store.search(query_vector)

    # context = "\n".join(chunks)
    context = "\n\n".join([
    chunk.strip() for chunk in chunks
    if len(chunk.strip()) > 20
])

    answer = generate_answer(question, context)

    return answer


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
