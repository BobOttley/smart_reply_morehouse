# ──────────────────────────────
# 🧠  SMART REPLY BACKEND — FORMATTING FIXED
# ──────────────────────────────

import os
import json
import pickle
import re
import numpy as np
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from url_mapping import URL_MAPPING

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
        is_signature = (
            len(lines) > 1 and 
            any(re.search(r'\b(Mrs?\.?|Ms\.?|Mr\.?|Director|Manager|School|College|University|Tel:|Email:|Phone:)', 
                         line, re.I) for line in lines)
        )
        
        if is_signature:
            clean_lines = [line.strip() for line in lines if line.strip()]
            processed_paragraphs.append('<br>'.join(clean_lines))
        else:
            paragraph_html = paragraph.replace('\n', '<br>')
            processed_paragraphs.append(paragraph_html)
    
    # Step 4: Join all paragraphs with double <br> for proper spacing
    result = '<br><br>'.join(processed_paragraphs)
    
    # Step 5: Final cleanup — ensure all href attributes are safely quoted
    result = re.sub(r'href=([^\s">]+)', r'href="\1"', result)
    result = re.sub(r'<a\s+href="([^"]+)"\s*>([^<]+)</a>', r'<a href="\1">\2</a>', result)
    
    return result

def clean_gpt_email_output(md: str) -> str:
    md = md.strip()
    md = re.sub(r"^```(?:markdown)?", "", md, flags=re.I).strip()
    md = re.sub(r"```$", "", md, flags=re.I).strip()
    lines = md.splitlines()
    if lines:
        first_line = lines[0].strip()
        if (
            len(first_line) < 80 and
            not first_line.lower().startswith("dear") and
            not first_line.endswith(".") and
            not first_line.endswith(":")
        ):
            lines = lines[1:]
    md = "\n".join(lines).strip()
    md = re.sub(r"\*\*(.*?)\*\*", r"\1", md)
    md = re.sub(r"\*(.*?)\*", r"\1", md)
    return md.strip()

def cosine_similarity(a, b): 
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_safe_url(label: str) -> str: 
    return URL_MAPPING.get(label, "")

# ───────────── APP SETUP ─────────────

load_dotenv()
app = Flask(__name__, template_folder="templates")
CORS(app)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger.info("🚀 PEN Reply Flask server starting…")

EMBED_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.30
RESPONSE_LIMIT = 3
STANDARD_MATCH_THRESHOLD = 0.85

try:
    with open("metadata_morehouse.pkl", "rb") as f:
        kb = pickle.load(f)
        doc_embeddings = np.array(kb["embeddings"])
        metadata = kb["messages"]
    logger.info(f"✅ Loaded {len(metadata)} website chunks from metadata_morehouse.pkl")
except Exception as e:
    logger.error(f"❌ Failed loading metadata: {e}")
    doc_embeddings, metadata = [], []

standard_messages, standard_embeddings, standard_replies = [], [], []

def _load_standard_library():
    path = "standard_responses.json"
    if not os.path.exists(path): 
        return
    try:
        with open(path, "r") as f: 
            saved = json.load(f)
        for entry in saved:
            reply = entry["reply"]
            variants = entry.get("variants", [entry.get("message")])
            for msg in variants:
                redacted = remove_personal_info(msg)
                standard_messages.append(redacted)
                standard_embeddings.append(embed_text(redacted))
                standard_replies.append(reply)
        logger.info(f"✅ Loaded {len(standard_messages)} template reply variants.")
    except Exception as e:
        logger.error(f"❌ Failed loading templates: {e}")

_load_standard_library()

def check_standard_match(q_vec: np.ndarray) -> str:
    if not standard_embeddings: 
        return ""
    sims = [cosine_similarity(q_vec, emb) for emb in standard_embeddings]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= STANDARD_MATCH_THRESHOLD:
        logger.info(f"🔁 Using template (similarity {sims[best_idx]:.2f})")
        return standard_replies[best_idx]
    return ""

@app.route("/reply", methods=["POST"])
def generate_reply():
    try:
        body = request.get_json(force=True)
        question_raw = (body.get("message") or "").strip()
        source_type = body.get("source_type", "email")
        include_cta = body.get("include_cta", True)
        url_box_text = (body.get("url_box") or "").strip()
        instruction_raw = (body.get("instruction") or "").strip()
        question = remove_personal_info(question_raw)
        instruction = remove_personal_info(instruction_raw)
        url_map = parse_url_box(url_box_text)
        
        if not question:
            logger.warning("No message received in request")
            return jsonify({
                "reply": "<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
                "reply_markdown": "Thank you for your enquiry. A member of our admissions team will contact you shortly.",
                "reply_outlook": "Thank you for your enquiry. A member of our admissions team will contact you shortly.",
                "url": "", "link_label": ""
            }), 400
        if len(question.strip()) < 10:
            logger.debug(f"Short or heavily redacted question: {question}")
            return jsonify({
                "reply": "<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
                "reply_markdown": "Thank you for your enquiry. A member of our admissions team will contact you shortly.",
                "reply_outlook": "Thank you for your enquiry. A member of our admissions team will contact you shortly.",
                "url": "", "link_label": ""
            })

        logger.debug(f"Processing enquiry - Raw: {question_raw}, Redacted: {question}")
        q_vec = embed_text(question)

        matched = check_standard_match(q_vec)
        if matched:
            reply_md = matched
            reply_html = markdown_to_html(reply_md)
            reply_outlook = markdown_to_outlook_html(reply_md)
            logger.info(f"Using template response with similarity {check_standard_match(q_vec)}")
            return jsonify({
                "reply": reply_html,
                "reply_markdown": reply_md,
                "reply_outlook": reply_outlook,
                "url": "", "link_label": ""
            })

        # Sentiment detection only for email source_type
        score, strat = None, None
        if source_type == "email":
            sent_prompt = f"""
You are an expert school admissions assistant.

Please analyse the following parent enquiry (from an email) and return a JSON object with two keys:

- "score": an integer from 1 (very negative) to 10 (very positive)
- "strategy": a maximum 30 words strategy for how to reply to the message.

Only return the JSON object — no extra explanation.

Enquiry:
\"\"\"{question}\"\"\"
"""
            try:
                sent_json = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": sent_prompt}],
                    temperature=0.3
                ).choices[0].message.content.strip()
                sent = json.loads(sent_json)
                score = int(sent.get("score", 5))
                strat = sent.get("strategy", "Address enquiry with relevant school information.")
            except Exception as e:
                logger.error(f"Failed to parse sentiment JSON: {sent_json}, Error: {e}")
                score = 5
                strat = "Address enquiry with relevant school information."

        # Search top context
        sims = [(cosine_similarity(q_vec, vec), meta) for vec, meta in zip(doc_embeddings, metadata)]
        top = sorted([m for m in sims if m[0] >= SIMILARITY_THRESHOLD], key=lambda x: x[0], reverse=True)[:RESPONSE_LIMIT]
        if not top:
            logger.info("No relevant context found, using default response")
            return jsonify({
                "reply": "<p>Thank you for your enquiry. A member of our admissions team will contact you shortly.</p>",
                "reply_markdown": "Thank you for your enquiry. A member of our admissions team will contact you shortly.",
                "reply_outlook": "Thank you for your enquiry. A member of our admissions team will contact you shortly.",
                "url": "", "link_label": ""
            })

        context_blocks = [f"{m['content']}\n[Info source]({m.get('url','')})" if m.get('url') else m['content'] for _, m in top]
        top_context = "\n---\n".join(context_blocks)
        today_date = datetime.now().strftime('%d %B %Y')

        # Topic detection for CTA
        topic = "general"
        q_lower = question.lower()
        if "visit" in q_lower or "tour" in q_lower:
            topic = "visit"
        elif "fees" in q_lower or "cost" in q_lower:
            topic = "fees"
        elif "subjects" in q_lower or "curriculum" in q_lower:
            topic = "curriculum"

        if source_type == "form":
            message_intro = "Parent Enquiry Form Submission:"
        else:
            message_intro = "Parent Email:"

        # Email prompt
        prompt = f"""
TODAY'S DATE IS {today_date}.

You are Mrs Powell De Caires, Director of Admissions and Marketing at More House School, a UK all girls school from years 5 through to Sixth Form.

Mrs Powell De Caires is also the school Registrar. If you need to reference the registrar, do not use a name but use the email address registrar@morehousemail.org.uk.

Write a warm, professional reply to the parent below. If the enquiry came from an online form rather than an email, still reply as if writing directly to the parent, but don’t reference the form itself. Use only the approved school information provided. When an enquiry form is received, the school always sends out a prospectus, so include the prospectus url hyperlinked in the email. If you do not have the prospectus url do not guess, leave a placeholder and the admissions team will insert.

Follow these essential rules:
- Always use British spelling (e.g. organise, programme, enrolment)
- The "Open Events page" must always link to: https://www.morehouse.org.uk/admissions/our-open-events/
- DO NOT fabricate or guess any information. If something is unknown, say so honestly.
- DO include relevant links using Markdown format: [Anchor Text](https://...). Embed links naturally in the body of the reply.
- DO use approved anchor phrases like "Open Events page", "Admissions page", or "registration form"
- NEVER use vague anchors like "click here", "more info", "register here", "visit page", etc.
- NEVER show raw URLs, list links at the bottom, or use markdown formatting like bold, italics, or bullet points
- NEVER include expired dates. If unsure, direct the parent to the relevant web page instead

End your reply with this exact sign-off, using line breaks:

Mrs Powell De Caires  
Director of Admissions and Marketing  
More House School

{message_intro}
\"\"\"{question}\"\"\"
School Info:
\"\"\"{top_context}\"\"\"
"""
        reply_md = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4
        ).choices[0].message.content.strip()
        reply_md = clean_gpt_email_output(reply_md)

        # Add subtle CTA if toggle is on
        if include_cta:
            if topic == "visit":
                reply_md += "\n\nIf you haven’t yet had a chance to visit us, we’d be delighted to welcome you to the school."
            elif topic == "fees":
                reply_md += "\n\nIf you’d like to discuss your child’s needs further, I’d be happy to arrange a time to speak."
            elif topic == "curriculum":
                reply_md += "\n\nWe’re always happy to share more about how we support girls to thrive academically and beyond."
            elif source_type == "email" and score and score >= 8:
                reply_md += "\n\nDo let me know if you’d like me to send a personalised prospectus tailored to your daughter’s interests."

        reply_md = insert_links(reply_md, url_map)
        reply_html = markdown_to_html(reply_md)
        reply_outlook = markdown_to_outlook_html(reply_md)

        def extract_links_from_html(html):
            matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
            return [(text.strip(), url.strip()) for url, text in matches]
        links = extract_links_from_html(reply_html)
        matched_url = links[0][1] if links else ""
        matched_source = links[0][0] if links else ""

        logger.info(f"Generated reply for {source_type} enquiry")
        response = {
            "reply": reply_html,
            "reply_markdown": reply_md,
            "reply_outlook": reply_outlook,
            "url": matched_url,
            "link_label": matched_source
        }
        if source_type == "email" and score is not None and strat is not None:
            response["sentiment_score"] = score
            response["strategy_explanation"] = strat
        return jsonify(response)

    except Exception as e:
        logger.error(f"REPLY ERROR: {e}")
        return jsonify({"error": "Internal server error."}), 500

@app.route("/revise", methods=["POST"])
def revise_reply():
    """
    Revise an existing reply based on user instructions or use provided edited reply
    """
    try:
        body = request.get_json(force=True)
        message = (body.get("message") or "").strip()
        previous_reply = (body.get("previous_reply") or "").strip()
        instruction = (body.get("instruction") or "").strip()
        edited_reply = (body.get("edited_reply") or "").strip()
        url_box_text = (body.get("url_box") or "").strip()

        if not message or not previous_reply:
            return jsonify({"error": "Missing message or previous reply."}), 400

        # Clean and process inputs
        clean_message = remove_personal_info(message)
        clean_instruction = remove_personal_info(instruction)
        url_map = parse_url_box(url_box_text)
        
        # If edited_reply is provided, use it directly
        if edited_reply:
            reply_md = clean_gpt_email_output(edited_reply)
            # Ensure sign-off is included
            sign_off = """
Mrs Powell De Caires  
Director of Admissions and Marketing  
More House School
"""
            if not reply_md.endswith(sign_off.strip()):
                reply_md += "\n\n" + sign_off.strip()
        else:
            # Get current date
            today_date = datetime.now().strftime('%d %B %Y')
            
            # Build revision prompt
            prompt = f"""
TODAY'S DATE IS {today_date}.

You are Mrs Powell De Caires, Director of Admissions and Marketing at More House School, a UK all girls school from years 5 through to Sixth Form.
In the extreme rare time you need to reference the registrar, never mention their name. 

Please revise the email reply below based on the parent's original enquiry and the revision instruction provided.

If the instruction asks for a call to action, you may add one of the following:
- An invitation to visit the school
- A prompt to request a personalised prospectus
- An offer to discuss next steps

Follow these essential rules:
- Always use British spelling (e.g. organise, programme, enrolment)
- The "Open Events page" must always link to: https://www.morehouse.org.uk/admissions/our-open-events/
- DO include relevant links using Markdown format: [Anchor Text](https://...). Embed links naturally in the body of the reply.
- DO use approved anchor phrases like "Open Events page", "Admissions page", or "registration form"
- NEVER use vague anchors like "click here", "more info", "register here", "visit page", etc.
- NEVER show raw URLs, list links at the bottom, or use markdown formatting like bold, italics, or bullet points
- NEVER include expired dates. If unsure, direct the parent to the relevant web page instead

Reply only with the revised email body in Markdown format, ready to send. Do not include 'Subject:', triple backticks, or code blocks.

- NEVER use markdown formatting like bold, italics, or bullet points

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
            reply_md = clean_gpt_email_output(reply_md)

        # Default score and strategy in case sentiment fails
        score, strat = 5, "Revised response"

        # Try sentiment detection
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
        except Exception as e:
            logger.error(f"Failed to parse sentiment JSON in revise: {sent_json}, Error: {e}")

        # Add subtle CTA
        if "visit" in clean_message.lower():
            reply_md += "\n\nIf you haven’t yet had a chance to visit us, we’d be delighted to welcome you to the school."
        elif "fees" in clean_message.lower():
            reply_md += "\n\nIf you’d like to discuss your child’s needs further, I’d be happy to arrange a time to speak."
        elif "curriculum" in clean_message.lower():
            reply_md += "\n\nWe’re always happy to share more about how we support girls to thrive academically and beyond."
        elif score >= 8:
            reply_md += "\n\nDo let me know if you’d like me to send a personalised prospectus tailored to your daughter’s interests."

        reply_md = insert_links(reply_md, url_map)
        
        # Convert to different formats
        reply_html = markdown_to_html(reply_md)
        reply_outlook = markdown_to_outlook_html(reply_md)
        
        # Extract links for response
        def extract_links_from_html(html):
            matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
            return [(text.strip(), url.strip()) for url, text in matches]
        
        links = extract_links_from_html(reply_html)
        matched_url = links[0][1] if links else ""
        matched_source = links[0][0] if links else ""

        logger.info(f"Revised reply for enquiry with sentiment score {score}")
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
        logger.error(f"REVISE ERROR: {e}")
        return jsonify({"error": "Internal server error during revision."}), 500

@app.route("/save-standard", methods=["POST"])
def save_standard_reply():
    try:
        body = request.get_json(force=True)
        message = (body.get("message") or "").strip()
        reply = (body.get("reply") or "").strip()
        urls = body.get("urls", [])

        if not message or not reply:
            return jsonify({"error": "Missing message or reply."}), 400

        # Load existing saved responses
        path = "standard_responses.json"
        saved = []
        if os.path.exists(path):
            with open(path, "r") as f:
                saved = json.load(f)

        # Add new entry
        entry = {
            "message": message,
            "reply": reply,
            "urls": urls
        }
        saved.append(entry)

        with open(path, "w") as f:
            json.dump(saved, f, indent=2)

        logger.info("Saved new standard response")
        return jsonify({"status": "saved"})
    except Exception as e:
        logger.error(f"SAVE ERROR: {e}")
        return jsonify({"error": "Internal server error during save."}), 500

@app.route("/")
def index(): 
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)