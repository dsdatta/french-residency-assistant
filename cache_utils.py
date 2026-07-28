import os
import json
import hashlib


# fn generate hash
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# fn load cache
def load_json_cache(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# fn save cache
def save_json_cache(filepath: str, cache: dict):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cache, f)
