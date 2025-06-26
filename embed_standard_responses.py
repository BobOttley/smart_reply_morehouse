import json
import numpy as np
import pickle
import os
from openai import OpenAI
from dotenv import load_dotenv

# ───────────────────────────────────────────────
# ✅ SETUP
# ───────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────────────────────────────
# ✅ LOAD RESPONSES
# ───────────────────────────────────────────────
with open("standard_responses_morehouse.json", "r") as f:
    data = json.load(f)

messages = [entry["message"] for entry in data]
embeddings = []

# ───────────────────────────────────────────────
# ✅ GENERATE EMBEDDINGS
# ───────────────────────────────────────────────
for msg in messages:
    res = client.embeddings.create(
        input=[msg],
        model="text-embedding-3-small"
    )
    embeddings.append(res.data[0].embedding)

# ───────────────────────────────────────────────
# ✅ SAVE TO PICKLE
# ───────────────────────────────────────────────
with open("standard_embeddings_morehouse.pkl", "wb") as f:
    pickle.dump({
        "messages": messages,
        "embeddings": np.array(embeddings)
    }, f)

print(f"✅ Embedded {len(messages)} messages into standard_embeddings_morehouse.pkl")
