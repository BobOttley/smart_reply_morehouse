import os
import pickle
import numpy as np
from PyPDF2 import PdfReader
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> list:
    """Generate embedding for a block of text using OpenAI."""
    try:
        response = openai.embeddings.create(
            model=EMBED_MODEL,
            input=text.replace("\n", " ")
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return np.zeros(1536).tolist()

def chunk_text(text: str, max_length: int = 1000) -> list:
    """Split text into chunks no longer than max_length characters, by paragraph."""
    paragraphs = text.split("\n")
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_length:
            current += "\n" + para
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para
    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 30]

def process_pdf_and_append_to_kb(file_path, metadata_path="embeddings/metadata_morehouse.pkl") -> int:
    """Read a PDF, embed chunks, and append to More House metadata."""
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks = chunk_text(text)

        embeddings = [embed_text(chunk) for chunk in chunks]
        metadata = [{"content": chunk, "url": ""} for chunk in chunks]

        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                kb = pickle.load(f)
        else:
            kb = {"embeddings": [], "messages": []}

        kb["embeddings"].extend(embeddings)
        kb["messages"].extend(metadata)

        with open(metadata_path, "wb") as f:
            pickle.dump(kb, f)

        print(f"✅ {len(chunks)} chunks added to More House KB.")
        return len(chunks)

    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return 0
