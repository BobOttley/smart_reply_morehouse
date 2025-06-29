import os
import json
import pickle
import re
import numpy as np
from datetime import datetime
from markdown import markdown
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from markdownify import markdownify as html_to_markdown
from url_mapping import URL_MAPPING



def parse_url_box(url_text):
    """
    Parses both newline-separated and semicolon-separated anchor=URL entries.
    """
    url_map = {}
    parts = re.split(r'[;\n]+', url_text.strip())
    for part in parts:
        if '=' in part:
            anchor, url = part.split('=', 1)
            url_map[anchor.strip()] = url.strip()
    return url_map




def insert_links(text, url_map):
    """
    Finds any words/phrases in the text that match the anchors and replaces
    them with Markdown links (e.g. Head → [Head](...)).
    """
    def safe_replace(match):
        word = match.group(0)
        for anchor, url in url_map.items():
            if word.lower() == anchor.lower():
                return f"[{word}]({url})"
        return word

    sorted_anchors = sorted(url_map.keys(), key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(a) for a in sorted_anchors) + r')\b'
    return re.sub(pattern, safe_replace, text, flags=re.IGNORECASE)

# ──────────────────────────────
# ✅  SET-UP
# ──────────────────────────────
load_dotenv()
app = Flask(__name__, template_folder="templates")
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("🚀 PEN Reply Flask server starting…")

EMBED_MODEL               = "text-embedding-3-small"
SIMILARITY_THRESHOLD      = 0.30
RESPONSE_LIMIT            = 3
STANDARD_MATCH_THRESHOLD  = 0.85

# ──────────────────────────────
# 🔒  PII REDACTION
# ──────────────────────────────
PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",                # emails
    r"\b(?:\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}\b",                # UK mobile
    r"\b(?:\+44\s?1\d{3}|\(?01\d{3}\)?|\(?02\d{3}\)?)\s?\d{3}\s?\d{3,4}\b", # UK landline
    r"\+?\d[\d\s\-().]{7,}\d",                                             # general int’l format
    r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b",                               # UK postcode
]

def remove_personal_info(text: str) -> str:
    # Basic PII redaction
    for pat in PII_PATTERNS:
        text = re.sub(pat, "[redacted]", text, flags=re.I)

    # Remove intros like "My name is Mr Smith", "I'm Mrs Jones"
    text = re.sub(
        r"\b(my name is|i am|i’m|i’m called)\s+(mr\.?|mrs\.?|ms\.?|miss)?\s*[A-Z][a-z]+\b",
        "my name is [redacted]",
        text,
        flags=re.I
    )

    # Remove "Dear Mr Carter", "Dear Ms Jones"
    text = re.sub(
        r"\bDear\s+(Mr\.?|Mrs\.?|Ms\.?|Miss)?\s*[A-Z][a-z]+\b",
        "Dear [redacted]",
        text,
        flags=re.I
    )

    # Remove sign-offs like "Regards, John"
    text = re.sub(
        r"\b(?:regards|thanks|thank you|sincerely|best wishes|kind regards)[,]?\s+[A-Z][a-z]+\b",
        "[redacted]",
        text,
        flags=re.I
    )

    return text

# ──────────────────────────────
# 📦  HELPERS
# ──────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """
    Embed a text string using the OpenAI embedding model and return a NumPy array.
    """
    text = text.replace("\n", " ")
    res = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(res.data[0].embedding)


def markdown_to_html(text: str) -> str:
    """
    Convert Markdown links to HTML and preserve paragraph structure.
    Ensures hrefs are quoted and no target attributes.
    Used for clean Smart Reply preview rendering.
    """
    # Convert Markdown links [Anchor](URL) → <a href="URL">Anchor</a>
    text = re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        text
    )

    # Split into paragraphs by double line breaks
    paragraphs = re.split(r'\n\s*\n', text.strip())

    # Wrap each paragraph in <p>...</p>
    return '\n'.join(f'<p>{p.strip()}</p>' for p in paragraphs if p.strip())


def markdown_to_outlook_html(md: str) -> str:
    # Convert Markdown links to HTML with quotes
    md = re.sub(
        r'\[([^\]]+)\]\((https?://[^\)]+)\)',
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        md
    )
    # Split paragraphs by double line breaks and use <br><br>
    paragraphs = re.split(r'\n\s*\n', md.strip())
    return '<br><br>'.join(p.strip() for p in paragraphs if p.strip())




def clean_gpt_email_output(md: str) -> str:
    """
    Clean up GPT output by removing any unwanted Markdown wrappers or headings.
    Ensures clean Markdown without code fences or Subject lines.
    """
    md = md.strip()
    md = re.sub(r"^```(?:markdown)?", "", md, flags=re.I).strip()
    md = re.sub(r"```$", "", md, flags=re.I).strip()
    md = re.sub(r"^(markdown:|subject:)[\s]*", "", md, flags=re.I).strip()
    md = re.sub(r"^Subject:.*\n?", "", md, flags=re.I)
    return md.strip()


def cosine_similarity(a, b):
    """
    Return cosine similarity between two NumPy vectors.
    """
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_safe_url(label: str) -> str:
    """
    Fetch a safe URL from the URL_MAPPING dictionary based on the given label.
    """
    return URL_MAPPING.get(label, "")

# ──────────────────────────────
# 📚  LOAD SCHOOL KB
# ──────────────────────────────
try:
    with open("metadata_morehouse.pkl", "rb") as f:
        kb = pickle.load(f)
        doc_embeddings = np.array(kb["embeddings"])
        metadata = kb["messages"]

    print(f"✅ Loaded {len(metadata)} website chunks from metadata_morehouse.pkl")
except Exception as e:
    print(f"❌ Failed to load More House metadata: {e}")
    doc_embeddings = []
    metadata = []

# ──────────────────────────────
# 📁  LOAD SAVED STANDARD RESPONSES
# ──────────────────────────────
standard_messages, standard_embeddings, standard_replies = [], [], []

def _load_standard_library():
    path = "standard_responses.json"  # or change to your chosen filename
    if not os.path.exists(path):
        print("⚠️ No standard_responses.json found.")
        return
    try:
        with open(path, "r") as f:
            saved = json.load(f)
        for entry in saved:
            reply = entry["reply"]
            variants = entry.get("variants", [entry.get("message")])
            for msg in variants:
                redacted_msg = remove_personal_info(msg)
                standard_messages.append(redacted_msg)
                standard_embeddings.append(embed_text(redacted_msg))
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

# ──────────────────────────────
# 📨  POST /reply
# ──────────────────────────────
@app.route("/reply", methods=["POST"])
def generate_reply():
    try:
        body = request.get_json(force=True)
        question_raw = (body.get("message") or "").strip()
        url_box_text = (body.get("url_box") or "").strip()
        url_map = parse_url_box(url_box_text)
        instruction_raw = (body.get("instruction") or "").strip()

        # 🔒 sanitise
        question    = remove_personal_info(question_raw)
        instruction = remove_personal_info(instruction_raw)

        if not question:
            return jsonify({"error":"No message received."}), 400

        q_vec = embed_text(question)

        # 1) pre-approved template?
        matched = check_standard_match(q_vec)
        if matched:
            if isinstance(matched, str):
                reply_md = matched
                safe_label = ""
                safe_url = ""
            else:
                reply_md = matched.get("reply", "")
                safe_label = matched.get("link_label", "")
                safe_url = get_safe_url(safe_label)

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


        # 2) sentiment (mini model, cheap)
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
        except Exception:
            score, strat = 5, ""
            print("⚠️ Sentiment parse failed.")

        # 3) KB retrieval
        sims = [(cosine_similarity(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
        top = [m for m in sims if m[0] >= SIMILARITY_THRESHOLD]
        top = sorted(top, key=lambda x:x[0], reverse=True)[:RESPONSE_LIMIT]

        if not top:
            return jsonify({
                "reply":"<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
                "sentiment_score":score,"strategy_explanation":strat
            })

        context_blocks = [f"{m['content']}\n[Info source]({m.get('url','')})" if m.get('url') else m['content'] for _, m in top]

        top_context = "\n---\n".join(context_blocks)

        # 4) main reply prompt

        from datetime import datetime

        today_date = datetime.now().strftime('%d %B %Y')

        prompt = f"""

TODAY'S DATE IS {today_date}.

You are Mrs Powell De Caires , Director of Admissions and Marketing at More House School, a UK all girls school from years 5 through to Sixth Form.

Write a warm, professional email reply to the parent below, using only the approved school information provided.

Follow these essential rules:
- Always use British spelling (e.g. organise, programme, enrolment)
- The “Open Events page” must always link to: https://www.morehouse.org.uk/admissions/our-open-events/
- Do NOT fabricate or guess any information. If something is unknown, say so honestly.
- DO include relevant links using Markdown format: [Anchor Text](https://...). Embed links naturally in the body of the reply.
- DO use approved anchor phrases like “Open Events page”, “Admissions page”, or “registration form”
- NEVER use vague anchors like “click here”, “more info”, “register here”, “visit page”, etc.
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


        # Format the reply
        reply_md = insert_links(reply_md, url_map)
        reply_html = markdown_to_html(reply_md)


        # ✅ Extract URLs from HTML
        import re

        def extract_links_from_html(html):
            matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
            return [(text.strip(), url.strip()) for url, text in matches]

        links = extract_links_from_html(reply_html)
        matched_url = links[0][1] if links else ""
        matched_source = links[0][0] if links else ""

        # ✅ Generate Outlook-safe version
        reply_outlook = markdown_to_outlook_html(reply_md)

        # ✅ Return enriched result
        return jsonify({
            "reply": reply_html,                 # Smart Reply preview
            "reply_markdown": reply_md,          # Raw Markdown (for copy)
            "reply_outlook": reply_outlook,      # Clean <br><br> HTML for Outlook
            "sentiment_score": score,
            "strategy_explanation": strat,
            "url": matched_url,
            "link_label": matched_source
        })



    except Exception as e:
        print(f"❌ REPLY ERROR: {e}")
        return jsonify({"error":"Internal server error."}), 500

# ──────────────────────────────
# ✏️  POST /revise
# ──────────────────────────────

@app.route("/revise", methods=["POST"])
def revise():
    try:
        body = request.get_json(force=True)
        message_raw   = (body.get("message") or "").strip()
        prev_reply    = (body.get("previous_reply") or "").strip()
        instruction_raw = (body.get("instruction") or "").strip()
        url_box_text  = (body.get("url_box") or "").strip()

        if not (message_raw and prev_reply):
            return jsonify({"error":"Missing fields."}), 400

        message     = remove_personal_info(message_raw)
        instruction = remove_personal_info(instruction_raw)
        url_map     = parse_url_box(url_box_text)

        prompt = f"""
Revise the admissions reply below according to the instruction.

Instruction: {instruction}

Parent enquiry:
\"\"\"{message}\"\"\"

Current reply (Markdown):
\"\"\"{prev_reply}\"\"\"

Return only the revised reply in Markdown.
""".strip()

        new_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        new_md = clean_gpt_email_output(new_md)

        # 🔗 Insert anchor links (if provided)
        new_md_linked = insert_links(new_md, url_map)

        return jsonify({"reply": markdown_to_html(new_md_linked)})

    except Exception as e:
        print(f"❌ REVISION ERROR: {e}")
        return jsonify({"error":"Revision failed."}), 500

# ──────────────────────────────
# 💾  POST /save-standard
# ──────────────────────────────
@app.route("/save-standard", methods=["POST"])
def save_standard():
    try:
        body = request.get_json(force=True)
        msg_raw = (body.get("message") or "").strip()
        reply   = (body.get("reply")   or "").strip()

        if not (msg_raw and reply):
            return jsonify({"status":"error","message":"Missing fields"}), 400

        msg_redacted = remove_personal_info(msg_raw)

        # append & persist
        record = {"timestamp":datetime.now().isoformat(),"message":msg_redacted,"reply":reply}
        path="standard_responses.json"
        data=[]
        if os.path.exists(path):
            with open(path,"r") as f: data=json.load(f)
        data.append(record)
        with open(path,"w") as f: json.dump(data,f,indent=2)

        # in-memory
        standard_messages.append(msg_redacted)
        standard_embeddings.append(embed_text(msg_redacted))
        standard_replies.append(reply)

        return jsonify({"status":"ok"})
    except Exception as e:
        print(f"❌ SAVE ERROR: {e}")
        return jsonify({"status":"error","message":"Save failed"}),500

# ──────────────────────────────
# 🌐  SERVE FRONT END
# ──────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

# ──────────────────────────────
# ▶️  MAIN
# ──────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
