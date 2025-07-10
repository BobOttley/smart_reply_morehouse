import os
import json
import hashlib
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
import pdfplumber

START_URL = "https://www.morehouse.org.uk/"
DOMAIN = "morehouse.org.uk"
MAX_DEPTH = 5

OUTPUT_DIR = "scraped"
PDF_DIR = os.path.join("uploaded_pdfs")  # Matches app.py PDFs folder
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "content.jsonl")
URL_MAPPING_FILE = "url_mapping.py"  # Root folder for app.py import

visited = set()
results = []
url_mapping = {}

def is_internal(url):
    return urlparse(url).netloc.endswith(DOMAIN)

def clean_text(text):
    return ' '.join(text.split())

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    main = soup.find("main")
    body = soup.find("body")
    text = main.get_text(" ") if main else (body.get_text(" ") if body else soup.get_text(" "))
    return clean_text(text)

def extract_text_from_pdf(path):
    try:
        text = ''
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + ' '
        return clean_text(text)
    except Exception as e:
        print(f"❌ PDF extraction failed: {path} — {e}")
        return None

def get_safe_pdf_filename(pdf_url):
    path = urlparse(pdf_url).path
    original_name = os.path.basename(path)
    original_name = unquote(original_name)

    base, ext = os.path.splitext(original_name)
    if ext.lower() != ".pdf":
        ext = ".pdf"

    generic_names = {"document", "file", "download", "pdf"}
    if base.lower() in generic_names or len(base) < 3:
        hash_id = hashlib.sha1(pdf_url.encode()).hexdigest()[:8]
        filename = f"{base}_{hash_id}{ext}"
    else:
        filename = f"{base}{ext}"

    return filename

def download_and_parse_pdf(pdf_url):
    try:
        os.makedirs(PDF_DIR, exist_ok=True)
        filename = get_safe_pdf_filename(pdf_url)
        local_path = os.path.join(PDF_DIR, filename)

        if not os.path.exists(local_path):
            print(f"📥 Downloading: {pdf_url}")
            r = requests.get(pdf_url, timeout=15)
            if r.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"❌ PDF download error {r.status_code}: {pdf_url}")
                return None

        text = extract_text_from_pdf(local_path)
        if text:
            results.append({"url": pdf_url, "type": "pdf", "content": text})
            return text
        else:
            results.append({"url": pdf_url, "type": "pdf", "content": "", "error": "Unreadable PDF"})
            return None
    except Exception as e:
        print(f"❌ PDF processing error: {pdf_url} — {e}")
        return None

def crawl(url, depth=0):
    if depth > MAX_DEPTH or url in visited or not is_internal(url):
        return
    visited.add(url)
    print(f"🔍 Crawling ({depth}): {url}")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh) Safari/537.36"}
        res = requests.get(url, headers=headers, timeout=10)

        if res.status_code != 200 or "text/html" not in res.headers.get("Content-Type", ""):
            print(f"⚠️ Skipping non-HTML or failed URL: {url}")
            return

        html = res.text
        text = extract_text_from_html(html)
        if text:
            results.append({"url": url, "type": "html", "content": text})

        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"].split("#")[0]
            full_url = urljoin(url, href)
            anchor = link.get_text(strip=True)
            if is_internal(full_url):
                if full_url.lower().endswith(".pdf"):
                    pdf_text = download_and_parse_pdf(full_url)
                    if pdf_text and anchor and 3 < len(anchor) < 80:
                        url_mapping[anchor] = full_url
                else:
                    if anchor and 3 < len(anchor) < 80:
                        url_mapping[anchor] = full_url
                    crawl(full_url, depth + 1)

    except Exception as e:
        print(f"❌ Failed to crawl {url}: {e}")

def save_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")
    print(f"\n✅ Saved {len(results)} items to {OUTPUT_FILE}")

def save_url_mapping():
    with open(URL_MAPPING_FILE, "w", encoding="utf-8") as f:
        f.write("URL_MAPPING = {\n")
        for k, v in sorted(url_mapping.items()):
            f.write(f'    "{k}": "{v}",\n')
        f.write("}\n")
    print(f"🔗 Saved anchor → URL mapping to {URL_MAPPING_FILE}")

if __name__ == "__main__":
    crawl(START_URL)
    save_results()
    save_url_mapping()
