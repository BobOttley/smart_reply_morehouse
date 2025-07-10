import os
import json
import pickle
import numpy as np
import pdfplumber
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 3000  # safer chunk size under token limit

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

def chunk_text(text: str, max_length: int = CHUNK_SIZE) -> list:
    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_length:
            # Split large paragraph into smaller parts
            for i in range(0, len(para), max_length):
                part = para[i:i+max_length]
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.append(part.strip())
        else:
            if len(current) + len(para) + 1 <= max_length:
                current += " " + para
            else:
                if current:
                    chunks.append(current.strip())
                current = para
    if current:
        chunks.append(current.strip())
    return chunks

def process_pdf_and_append_to_kb(file_path, metadata_path="embeddings/metadata.pkl") -> int:
    """Extract text from one PDF, embed chunks, and append to KB."""
    try:
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'

        chunks = chunk_text(text)
        embeddings = [embed_text(chunk) for chunk in chunks]

        filename = os.path.basename(file_path)
        new_metadata = [
            {
                "content": chunk,
                "source": filename,
                "url": f"/uploaded_pdfs/{filename}"
            }
            for chunk in chunks
        ]

        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                kb = pickle.load(f)
        else:
            kb = {"embeddings": [], "messages": []}

        kb["embeddings"].extend(embeddings)
        kb["messages"].extend(new_metadata)

        with open(metadata_path, "wb") as f:
            pickle.dump(kb, f)

        print(f"✅ {len(chunks)} chunks added to KB from {filename}.")
        return len(chunks)

    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return 0

def build_kb_from_scraped_content(content_path="scraped/content.jsonl", metadata_path="embeddings/metadata.pkl") -> int:
    """Process all scraped content (HTML + PDF text), embed chunks, and build full KB."""
    if not os.path.exists(content_path):
        print(f"❌ Scraped content file not found: {content_path}")
        return 0

    with open(content_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    all_chunks = []
    all_embeddings = []
    all_metadata = []

    for item in data:
        text = item.get("content", "")
        url = item.get("url", "")
        if not text.strip():
            continue
        chunks = chunk_text(text)
        embeddings = [embed_text(chunk) for chunk in chunks]

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        all_metadata.extend([
            {
                "content": chunk,
                "source": url,
                "url": url
            }
            for chunk in chunks
        ])

    kb = {
        "embeddings": all_embeddings,
        "messages": all_metadata
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "wb") as f:
        pickle.dump(kb, f)

    print(f"✅ Knowledge base built with {len(all_chunks)} chunks.")
    return len(all_chunks)

if __name__ == "__main__":
    build_kb_from_scraped_content()
