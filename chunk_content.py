import os, json
from pathlib import Path

INPUT_FILE  = "scraped/content.jsonl"
OUTPUT_FILE = "data/chunks.jsonl"
CHUNK_SIZE  = 1000  # target characters per chunk
MAX_ALLOWED = 4000  # hard limit to avoid OpenAI embedding errors

os.makedirs("data", exist_ok=True)

def chunk_text(text, max_len=CHUNK_SIZE):
    chunks, current = [], ""
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current) + len(paragraph) < max_len:
            current += " " + paragraph
        else:
            chunks.append(current.strip())
            current = paragraph
    if current:
        chunks.append(current.strip())

    # ✅ Skip anything too long for safety
    return [c for c in chunks if len(c) < MAX_ALLOWED]

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    count, total_chunks = 0, 0
    for line in infile:
        item = json.loads(line)
        source_url = item.get("url", "")
        content    = item.get("content", "").strip()
        doc_type   = item.get("type", "unknown")

        if not content:
            continue

        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            json.dump({
                "text"   : chunk,
                "url"    : source_url,
                "type"   : doc_type,
                "chunk"  : i + 1
            }, outfile)
            outfile.write("\n")
        count += 1
        total_chunks += len(chunks)

print(f"✅ Processed {count} items into {total_chunks} chunks → saved to {OUTPUT_FILE}")
