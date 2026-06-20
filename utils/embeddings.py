from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2") #sentence-transformers/all-mpnet-base-v2


# =========================
# EMBEDDINGS FOR CHUNKS
# =========================
def get_embeddings(text_chunks):
    return model.encode(text_chunks)


# =========================
# EMBEDDING FOR QUESTION
# =========================
def get_query_embedding(text):
    return model.encode([text])[0]