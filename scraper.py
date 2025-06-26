import os
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import hashlib

START_URL = "https://www.morehouse.org.uk/"
DOMAIN = "morehouse.org.uk"
OUTPUT_DIR = "scraped"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "content.jsonl")

visited = set()
results = []

def is_internal(url):
    return DOMAIN in urlparse(url).netloc

def clean_text(text):
    return ' '.join(text.split())

def extract_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    main = soup.find("main")
    text = main.get_text(separator=' ') if main else soup.get_text(separator=' ')
    return clean_text(text)


def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return clean_text(text)
    except Exception as e:
        print(f"❌ Failed to extract PDF text: {path}, error: {e}")
        return None

def download_and_parse_pdf(pdf_url):
    try:
        # Create a short, safe filename using SHA1 hash of the URL
        url_hash = hashlib.sha1(pdf_url.encode()).hexdigest()[:10]
        ext = os.path.splitext(pdf_url)[1] or ".pdf"
        filename = f"pdf_{url_hash}{ext}"
        local_path = os.path.join(PDF_DIR, filename)

        if not os.path.exists(local_path):
            print(f"📥 Downloading PDF: {pdf_url}")
            r = requests.get(pdf_url, timeout=15)
            if r.status_code != 200:
                print(f"❌ PDF not found (status {r.status_code}): {pdf_url}")
                results.append({"url": pdf_url, "type": "pdf", "content": "", "error": f"PDF not found (status {r.status_code})"})
                return
            with open(local_path, "wb") as f:
                f.write(r.content)

        text = extract_text_from_pdf(local_path)
        if text:
            results.append({"url": pdf_url, "type": "pdf", "content": text})
        else:
            results.append({"url": pdf_url, "type": "pdf", "content": "", "error": "unreadable PDF"})
    except Exception as e:
        print(f"❌ Error downloading or parsing {pdf_url}: {e}")
        results.append({"url": pdf_url, "type": "pdf", "content": "", "error": str(e)})

def crawl(url):
    if url in visited or not is_internal(url):
        return
    visited.add(url)
    print(f"🔍 Crawling: {url}")
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200 or "text/html" not in res.headers.get("Content-Type", ""):
            return
        html = res.text
        text = extract_text_from_html(html)
        if text:
            results.append({"url": url, "type": "html", "content": text})
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(url, href)
            if full_url.lower().endswith(".pdf"):
                download_and_parse_pdf(full_url)
            elif is_internal(full_url):
                crawl(full_url)
    except Exception as e:
        print(f"❌ Failed to crawl {url}: {e}")

def save_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            json.dump(item, f)
            f.write("\n")
    print(f"\n✅ Finished. {len(results)} items saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    crawl(START_URL)
    save_results()
