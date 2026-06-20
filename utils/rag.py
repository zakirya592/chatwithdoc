import faiss
import numpy as np
from transformers import pipeline

# ✅ WORKING PIPELINE FOR YOUR VERSION
generator = pipeline(
    "text-generation",
    model="google/flan-t5-small"
)


class VectorStore:

    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, embeddings, chunks):

        dimension = len(embeddings[0])

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(
            np.array(embeddings).astype("float32")
        )

        self.chunks = chunks

    def search(self, query_vector, top_k=3):

        D, I = self.index.search(
            np.array([query_vector]).astype("float32"),
            top_k
        )

        return [self.chunks[i] for i in I[0] if i < len(self.chunks)]


def generate_answer(question, context):

    context = context[:1500]

    prompt = f"""
Answer ONLY from context.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        temperature=0.2
    )

    return result[0]["generated_text"]
