import json

url_map = {}

with open('scraped/content.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        url = item.get("url", "")
        content = item.get("content", "").lower()
        # Heuristic: use first few words as a key (customise as needed)
        key = None
        if "admission" in content:
            key = "admissions policy"
        elif "safeguard" in content:
            key = "safeguarding"
        # Add more heuristics here
        if key:
            url_map[key] = url

# Write to url_mapping.py
with open('url_mapping.py', 'w', encoding='utf-8') as f:
    f.write("url_map = {\n")
    for k, v in url_map.items():
        f.write(f"    {repr(k)}: {repr(v)},\n")
    f.write("}\n")
