# ──────────────────────────────
# 🧠  SMART REPLY BACKEND — FORMATTING FIXED
# ──────────────────────────────

import os, json, pickle, re
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from url_mapping import URL_MAPPING

# ───────────── HELPERS ─────────────

def parse_url_box(url_text):
    url_map = {}
    parts = re.split(r'[;\n]+', url_text.strip())
    for part in parts:
        if '=' in part:
            anchor, url = part.split('=', 1)
            url_map[anchor.strip()] = url.strip()
    return url_map

def insert_links(text, url_map):
    def safe_replace(match):
        word = match.group(0)
        for anchor, url in url_map.items():
            if word.lower() == anchor.lower():
                # Force valid Markdown (quoted + parenthesis-safe)
                safe_url = url.replace(')', '%29').replace('(', '%28')  # escape problematic chars
                return f"[{word}]({safe_url})"
        return word

    sorted_anchors = sorted(url_map.keys(), key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(a) for a in sorted_anchors) + r')\b'
    return re.sub(pattern, safe_replace, text, flags=re.IGNORECASE)


def remove_personal_info(text: str) -> str:
    PII_PATTERNS = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b(?:\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}\b",
        r"\b(?:\+44\s?1\d{3}|\(?01\d{3}\)?|\(?02\d{3}\)?)\s?\d{3}\s?\d{3,4}\b",
        r"\+?\d[\d\s\-().]{7,}\d",
        r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b",
    ]
    for pat in PII_PATTERNS:
        text = re.sub(pat, "[redacted]", text, flags=re.I)
    text = re.sub(r"\b(my name is|i am|i'm|i'm called)\s+(mr\.?|mrs\.?|ms\.?|miss)?\s*[A-Z][a-z]+\b", "my name is [redacted]", text, flags=re.I)
    text = re.sub(r"\bDear\s+(Mr\.?|Mrs\.?|Ms\.?|Miss)?\s*[A-Z][a-z]+\b", "Dear [redacted]", text, flags=re.I)
    text = re.sub(r"\b(?:regards|thanks|thank you|sincerely|best wishes|kind regards)[,]?\s+[A-Z][a-z]+\b", "[redacted]", text, flags=re.I)
    return text

def embed_text(text: str) -> np.ndarray:
    text = text.replace("\n", " ")
    res = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(res.data[0].embedding)

def markdown_to_html(text: str) -> str:
    text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', '<br>', text.strip())
    paragraphs = re.split(r'\n\s*\n', text)
    return '\n'.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())

def markdown_to_outlook_html(md: str) -> str:
    """
    Convert Markdown to Outlook-compatible HTML with proper formatting
    Handles signatures, paragraphs, and ensures proper HTML attribute quoting
    """
    if not md.strip():
        return ""
    
    # Step 1: Handle markdown links - convert to proper HTML with quoted attributes
    md = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', 
                lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', md)
    
    # Step 2: Split content into paragraphs (separated by double line breaks)
    paragraphs = re.split(r'\n\s*\n', md.strip())
    
    processed_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        lines = paragraph.split('\n')
        
        # Step 3: Detect if this is a signature block
        # Signatures typically have:
        # - Multiple lines
        # - Contains titles, names, or contact info
        # - Short lines (names, titles, school name)
        is_signature = (
            len(lines) > 1 and 
            any(re.search(r'\b(Mrs?\.?|Ms\.?|Mr\.?|Director|Manager|School|College|University|Tel:|Email:|Phone:)', 
                         line, re.I) for line in lines)
        )
        
        if is_signature:
            # For signatures: each line should be separated by <br>
            # Remove empty lines and join with <br>
            clean_lines = [line.strip() for line in lines if line.strip()]
            processed_paragraphs.append('<br>'.join(clean_lines))
        else:
            # For regular content: replace single line breaks with <br>
            # This preserves intentional line breaks within paragraphs
            paragraph_html = paragraph.replace('\n', '<br>')
            processed_paragraphs.append(paragraph_html)
    
    # Step 4: Join all paragraphs with double <br> for proper spacing
    result = '<br><br>'.join(processed_paragraphs)
    
    # Step 5: Final cleanup — ensure all href attributes are safely quoted
    result = re.sub(r'href=([^\s">]+)', r'href="\1"', result)  # ensure all href= are quoted
    result = re.sub(r'<a\s+href="([^"]+)"\s*>([^<]+)</a>', r'<a href="\1">\2</a>', result)  # ensure well-formed links

    
    return result

def clean_gpt_email_output(md: str) -> str:
    md = md.strip()
    md = re.sub(r"^```(?:markdown)?", "", md, flags=re.I).strip()
    md = re.sub(r"```$", "", md, flags=re.I).strip()
    md = re.sub(r"^(markdown:|subject:)[\s]*", "", md, flags=re.I).strip()
    md = re.sub(r"^Subject:.*\n?", "", md, flags=re.I)
    return md.strip()

def cosine_similarity(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_safe_url(label: str) -> str: return URL_MAPPING.get(label, "")

# ───────────── APP SETUP ─────────────

load_dotenv()
app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🚀 PEN Reply Flask server starting…")

EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.30
RESPONSE_LIMIT = 3
STANDARD_MATCH_THRESHOLD = 0.85

try:
    with open("metadata_morehouse.pkl", "rb") as f:
        kb = pickle.load(f)
        doc_embeddings = np.array(kb["embeddings"])
        metadata = kb["messages"]
    print(f"✅ Loaded {len(metadata)} website chunks from metadata_morehouse.pkl")
except:
    doc_embeddings, metadata = [], []

standard_messages, standard_embeddings, standard_replies = [], [], []

def _load_standard_library():
    path = "standard_responses.json"
    if not os.path.exists(path): return
    try:
        with open(path, "r") as f: saved = json.load(f)
        for entry in saved:
            reply = entry["reply"]
            variants = entry.get("variants", [entry.get("message")])
            for msg in variants:
                redacted = remove_personal_info(msg)
                standard_messages.append(redacted)
                standard_embeddings.append(embed_text(redacted))
                standard_replies.append(reply)
        print(f"✅ Loaded {len(standard_messages)} template reply variants.")
    except Exception as e:
        print(f"❌ Failed loading templates: {e}")

_load_standard_library()

def check_standard_match(q_vec: np.ndarray) -> str:
    if not standard_embeddings: return ""
    sims = [cosine_similarity(q_vec, emb) for emb in standard_embeddings]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= STANDARD_MATCH_THRESHOLD:
        print(f"🔁 Using template (similarity {sims[best_idx]:.2f})")
        return standard_replies[best_idx]
    return ""

@app.route("/reply", methods=["POST"])
def generate_reply():
    try:
        body = request.get_json(force=True)
        question_raw = (body.get("message") or "").strip()
        url_box_text = (body.get("url_box") or "").strip()
        instruction_raw = (body.get("instruction") or "").strip()
        question = remove_personal_info(question_raw)
        instruction = remove_personal_info(instruction_raw)
        url_map = parse_url_box(url_box_text)
        if not question: return jsonify({"error":"No message received."}), 400
        q_vec = embed_text(question)

        matched = check_standard_match(q_vec)
        if matched:
            reply_md = matched
            safe_label = ""
            safe_url = ""
            reply_html = markdown_to_html(reply_md)
            reply_outlook = markdown_to_outlook_html(reply_md)
            return jsonify({
                "reply": reply_html,
                "reply_markdown": reply_md,
                "reply_outlook": reply_outlook,
                "sentiment_score": 10,
                "strategy_explanation": "Used approved template.",
                "url": safe_url,
                "link_label": safe_label
            })

        sent_prompt = f"""
You are an expert school admissions assistant.

Please analyse the following parent enquiry and return a JSON object with two keys:

- "score": an integer from 1 (very negative) to 10 (very positive)
- "strategy": a maximum 30 words strategy for how to reply to the message

Only return the JSON object — no extra explanation.

Enquiry:
\"\"\"{question}\"\"\"
""".strip()

        sent_json = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":sent_prompt}],
            temperature=0.3
        ).choices[0].message.content.strip()

        try:
            sent = json.loads(sent_json)
            score = int(sent.get("score",5))
            strat = sent.get("strategy","")
        except: score, strat = 5, ""

        sims = [(cosine_similarity(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
        top = sorted([m for m in sims if m[0] >= SIMILARITY_THRESHOLD], key=lambda x:x[0], reverse=True)[:RESPONSE_LIMIT]
        if not top:
            return jsonify({
                "reply":"<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
                "sentiment_score":score,"strategy_explanation":strat
            })

        context_blocks = [f"{m['content']}\n[Info source]({m.get('url','')})" if m.get('url') else m['content'] for _, m in top]
        top_context = "\n---\n".join(context_blocks)
        today_date = datetime.now().strftime('%d %B %Y')

        prompt = f"""
TODAY'S DATE IS {today_date}.

You are Mrs Powell De Caires , Director of Admissions and Marketing at More House School, a UK all girls school from years 5 through to Sixth Form.

Write a warm, professional email reply to the parent below, using only the approved school information provided.

Follow these essential rules:
- Always use British spelling (e.g. organise, programme, enrolment)
- The "Open Events page" must always link to: https://www.morehouse.org.uk/admissions/our-open-events/
- DO NOT fabricate or guess any information. If something is unknown, say so honestly.
- DO include relevant links using Markdown format: [Anchor Text](https://...). Embed links naturally in the body of the reply.
- DO use approved anchor phrases like "Open Events page", "Admissions page", or "registration form"
- NEVER use vague anchors like "click here", "more info", "register here", "visit page", etc.
- NEVER show raw URLs, list links at the bottom, or use markdown formatting like bold, italics, or bullet points
- NEVER include expired dates. If unsure, direct the parent to the relevant web page instead

Reply only with the full email body in Markdown format, ready to send. Do not include 'Subject:', triple backticks, or code blocks.

Parent Email:
\"\"\"{question}\"\"\"

School Info:
\"\"\"{top_context}\"\"\"

End your reply with this exact sign-off, using line breaks:

Mrs Powell De Caires  
Director of Admissions and Marketing  
More House School
""".strip()

        reply_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        reply_md = clean_gpt_email_output(reply_md)
        reply_md = insert_links(reply_md, url_map)
        reply_html = markdown_to_html(reply_md)
        reply_outlook = markdown_to_outlook_html(reply_md)

        def extract_links_from_html(html):
            matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
            return [(text.strip(), url.strip()) for url, text in matches]
        links = extract_links_from_html(reply_html)
        matched_url = links[0][1] if links else ""
        matched_source = links[0][0] if links else ""

        return jsonify({
            "reply": reply_html,
            "reply_markdown": reply_md,
            "reply_outlook": reply_outlook,
            "sentiment_score": score,
            "strategy_explanation": strat,
            "url": matched_url,
            "link_label": matched_source
        })
    except Exception as e:
        print(f"❌ REPLY ERROR: {e}")
        return jsonify({"error":"Internal server error."}), 500

@app.route("/revise", methods=["POST"])
def revise_reply():
    """
    Revise an existing reply based on user instructions
    """
    try:
        body = request.get_json(force=True)
        message = (body.get("message") or "").strip()
        previous_reply = (body.get("previous_reply") or "").strip()
        instruction = (body.get("instruction") or "").strip()
        url_box_text = (body.get("url_box") or "").strip()

        if not message or not previous_reply:
            return jsonify({"error": "Missing message or previous reply."}), 400


        # Clean and process inputs
        clean_message = remove_personal_info(message)
        clean_instruction = remove_personal_info(instruction)
        url_map = parse_url_box(url_box_text)
        
        # Get current date
        today_date = datetime.now().strftime('%d %B %Y')
        
        # Build revision prompt
        prompt = f"""
TODAY'S DATE IS {today_date}.

You are Mrs Powell De Caires, Director of Admissions and Marketing at More House School, a UK all girls school from years 5 through to Sixth Form.

Please revise the email reply below based on the parent's original enquiry and the revision instruction provided.

Follow these essential rules:
- Always use British spelling (e.g. organise, programme, enrolment)
- The "Open Events page" must always link to: https://www.morehouse.org.uk/admissions/our-open-events/
- DO include relevant links using Markdown format: [Anchor Text](https://...). Embed links naturally in the body of the reply.
- DO use approved anchor phrases like "Open Events page", "Admissions page", or "registration form"
- NEVER use vague anchors like "click here", "more info", "register here", "visit page", etc.
- NEVER show raw URLs, list links at the bottom, or use markdown formatting like bold, italics, or bullet points
- NEVER include expired dates. If unsure, direct the parent to the relevant web page instead

Reply only with the revised email body in Markdown format, ready to send. Do not include 'Subject:', triple backticks, or code blocks.

Original Parent Email:
\"\"\"{clean_message}\"\"\"

Previous Reply:
\"\"\"{previous_reply}\"\"\"

Revision Instruction:
\"\"\"{clean_instruction}\"\"\"

End your reply with this exact sign-off, using line breaks:

Mrs Powell De Caires  
Director of Admissions and Marketing  
More House School
""".strip()

        # Generate revised reply
        reply_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        
        # Clean and process the reply
        reply_md = clean_gpt_email_output(reply_md)
        reply_md = insert_links(reply_md, url_map)
        
        # Convert to different formats
        reply_html = markdown_to_html(reply_md)
        reply_outlook = markdown_to_outlook_html(reply_md)  # Uses the improved function
        
        # Extract sentiment (reuse from original if no major changes)
        try:
            sent_prompt = f"""
You are an expert school admissions assistant.

Please analyse the following parent enquiry and return a JSON object with two keys:

- "score": an integer from 1 (very negative) to 10 (very positive)
- "strategy": a maximum 30 words strategy for how to reply to the message

Only return the JSON object — no extra explanation.

Enquiry:
\"\"\"{clean_message}\"\"\"
""".strip()

            sent_json = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":sent_prompt}],
                temperature=0.3
            ).choices[0].message.content.strip()

            sent = json.loads(sent_json)
            score = int(sent.get("score", 5))
            strat = sent.get("strategy", "Revised response")
        except:
            score, strat = 5, "Revised response"
        
        # Extract links for response
        def extract_links_from_html(html):
            matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
            return [(text.strip(), url.strip()) for url, text in matches]
        
        links = extract_links_from_html(reply_html)
        matched_url = links[0][1] if links else ""
        matched_source = links[0][0] if links else ""

        return jsonify({
            "reply": reply_html,
            "reply_markdown": reply_md,
            "reply_outlook": reply_outlook,
            "sentiment_score": score,
            "strategy_explanation": strat,
            "url": matched_url,
            "link_label": matched_source
        })
        
    except Exception as e:
        print(f"❌ REVISE ERROR: {e}")
        return jsonify({"error": "Internal server error during revision."}), 500

@app.route("/")
def index(): return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)