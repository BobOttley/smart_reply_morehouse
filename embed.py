import sys, os, json, pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_TOKENS = 8192

def count_tokens(text):
    # Approx: 1 token ≈ 4 chars in English
    return len(text) / 4

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def main():
    input_path = "scraped/content.jsonl"
    output_path = "metadata_morehouse.pkl"

    messages = []
    embeddings = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("content", "").strip()
            if not text:
                continue
            if count_tokens(text) > MAX_TOKENS:
                print(f"⚠️ Skipping oversized chunk: {obj['url']}")
                skipped += 1
                continue
            try:
                embed = embed_text(text)
                messages.append(obj)
                embeddings.append(embed)
            except Exception as e:
                print(f"❌ Failed to embed: {e}")
                skipped += 1

    with open(output_path, "wb") as f:
        pickle.dump({
            "messages": messages,
            "embeddings": np.array(embeddings)
        }, f)

    print(f"✅ Embedded {len(messages)} items into {output_path}")
    print(f"⚠️ Skipped {skipped} oversized or failed chunks")

if __name__ == "__main__":
    main()
