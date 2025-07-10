import os
import re
import json
import time
import pickle
import hashlib
import requests
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ========================== CONFIG ==========================
BASE_URL = "https://www.morehouse.org.uk/"  # Set your target URL here
MAX_DEPTH = 6
EMBEDDING_MODEL = "text-embedding-3-small"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
UPLOADED_PDFS_DIR = os.path.join(ROOT_DIR, "uploaded_pdfs")
os.makedirs(UPLOADED_PDFS_DIR, exist_ok=True)

# ======================= INIT ========================
visited, results, url_mapping = set(), [], {}
DOMAIN = urlparse(BASE_URL).netloc

try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    print("❌ FATAL: OPENAI_API_KEY environment variable not set.")
    exit()

def is_internal(url):
    parsed = urlparse(url)
    return parsed.netloc.endswith(DOMAIN) and parsed.scheme in ['http', 'https']

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    return clean_text(main.get_text(separator="\n")) if main else ""

def extract_chunks(text, min_length=50, max_length=1200):
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > min_length]
    final_chunks = []
    for para in paras:
        if len(para) > max_length:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 < max_length:
                    current_chunk += sentence + " "
                else:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk:
                final_chunks.append(current_chunk.strip())
        else:
            final_chunks.append(para)
    return final_chunks

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        return clean_text(" ".join(p.extract_text() or "" for p in reader.pages))
    except Exception as e:
        print(f"  ❌ PDF read error: {e}")
        return ""

# ----------- PDF TITLE LOGIC -----------
def get_pdf_title_from_first_page(pdf_path):
    """Extract title from first page using pdfplumber, fallback to None."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages):
                first_text = pdf.pages[0].extract_text()
                if first_text:
                    title = first_text.split('\n')[0].strip()
                    # Clean for filename
                    title = re.sub(r'[\\/*?:"<>|]', '', title)
                    title = title[:60].replace(' ', '_')
                    if title:
                        return title
    except Exception as e:
        print(f"  ⚠️ Could not extract PDF title: {e}")
    return None

def download_and_parse_pdf(url):
    try:
        hash_id = hashlib.sha1(url.encode()).hexdigest()[:12]
        temp_filename = f"pdf_{hash_id}.pdf"
        temp_path = os.path.join(UPLOADED_PDFS_DIR, temp_filename)
        # Download if not present
        if not os.path.exists(temp_path):
            print(f"  📥 Downloading PDF: {url}")
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and r.headers.get("Content-Type", "").strip() == "application/pdf":
                with open(temp_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"  ❌ PDF download failed ({r.status_code}): {url}")
                return
        else:
            print(f"  📂 Using cached PDF: {temp_filename}")

        # ---- GET TITLE AND RENAME IF FOUND ----
        new_title = get_pdf_title_from_first_page(temp_path)
        if new_title:
            final_filename = f"{new_title}.pdf"
            final_path = os.path.join(UPLOADED_PDFS_DIR, final_filename)
            if not os.path.exists(final_path):
                os.rename(temp_path, final_path)
            else:
                # If file exists (very rare), just use hash file
                final_path = temp_path
                final_filename = temp_filename
        else:
            final_path = temp_path
            final_filename = temp_filename

        text = extract_text_from_pdf(final_path)
        if text:
            for chunk in extract_chunks(text):
                results.append({
                    "content": chunk,
                    "url": f"/uploaded_pdfs/{final_filename}",
                    "source": final_filename,
                    "type": "pdf"
                })
        else:
            print(f"  ⚠️ Empty or unreadable PDF: {url}")
    except Exception as e:
        print(f"  ❌ PDF processing error: {url} — {e}")

def crawl(url, depth=0):
    if depth > MAX_DEPTH or url in visited or not is_internal(url):
        return
    visited.add(url)
    print(f"[{depth}] 🔍 Crawling: {url}")
    time.sleep(random.uniform(0.5, 1.5))
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        if "text/html" not in res.headers.get("Content-Type", ""):
            return

        soup = BeautifulSoup(res.text, "html.parser")
        title = (soup.title.string.strip() if soup.title else url).split("|")[0].strip()
        text = extract_text_from_html(res.text)
        if text:
            chunks = extract_chunks(text)
            print(f"  📝 Found {len(chunks)} chunks from '{title}'")
            for chunk in chunks:
                results.append({
                    "content": chunk,
                    "url": url,
                    "source": "",  # blank for web
                    "type": "html"
                })

        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0]
            if not href or href.startswith(('mailto:', 'tel:')): continue
            full_url = urljoin(url, href)
            anchor = a.get_text(strip=True)
            if full_url.lower().endswith(".pdf"):
                if full_url not in visited:
                    visited.add(full_url)
                    download_and_parse_pdf(full_url)
            elif is_internal(full_url):
                if anchor and 3 < len(anchor) < 80:
                    url_mapping[anchor] = full_url
                crawl(full_url, depth + 1)
    except requests.RequestException as e:
        print(f"  ❌ Network error crawling {url}: {e}")
    except Exception as e:
        print(f"  ❌ Unexpected error crawling {url}: {e}")

def generate_embeddings():
    print("\n🧠 Generating embeddings...")
    vectors = []
    for i, chunk in enumerate(results, 1):
        try:
            print(f"  Embedding {i}/{len(results)}...", end="\r")
            response = client.embeddings.create(input=[chunk["content"]], model=EMBEDDING_MODEL)
            vectors.append(response.data[0].embedding)
        except Exception as e:
            print(f"\n  ❌ Embedding failed for chunk {i}: {e}")
            vectors.append([0.0] * 1536)
    return vectors

def save_outputs(embeddings):
    # Save KB with both embeddings and messages for app.py
    meta_path = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "messages": results}, f)
    print(f"\n💾 Saved metadata.pkl with {len(results)} chunks and {len(embeddings)} embeddings.")

    # Save URL mapping (root)
    url_map_path = os.path.join(ROOT_DIR, "url_mapping.py")
    with open(url_map_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated URL mapping\n")
        f.write("URL_MAPPING = {\n")
        for k, v in sorted(url_mapping.items()):
            safe_key = k.replace('"', "'")
            f.write(f'    "{safe_key}": "{v}",\n')
        f.write("}\n")
    print("💾 Saved url_mapping.py")

def summary(embeddings):
    pages = len([c for c in results if c['type'] == 'html'])
    pdfs = len([c for c in results if c['type'] == 'pdf'])
    print("\n" + "="*50)
    print("📊 KNOWLEDGE BASE BUILD SUMMARY")
    print("="*50)
    print(f"  - 🧾 Pages scraped:     {pages}")
    print(f"  - 📄 PDFs processed:    {pdfs}")
    print(f"  - 🧩 Chunks created:    {len(results)}")
    print(f"  - 🧠 Embeddings:        {len(embeddings)}")
    print(f"  - 🔗 Anchors mapped:    {len(url_mapping)}")
    print(f"\n✅ All files saved successfully.")

# ===================== MAIN =====================

if __name__ == "__main__":
    start = time.time()
    crawl(BASE_URL)
    embeddings = generate_embeddings()
    save_outputs(embeddings)
    summary(embeddings)
    print(f"⏰ Total execution time: {time.time() - start:.2f} seconds")
