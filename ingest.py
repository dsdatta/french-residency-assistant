import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cache_utils import hash_text, load_json_cache, save_json_cache
from llm_client import get_embedding, generate_context
from vector_store import collection

CACHE_FILE = "embeddings_cache.json"
CONTEXT_CACHE_FILE = "context_cache.json"

# file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

# loop through all the source documents and save it documents[]
documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            documents.append({"source": filename, "content": f.read()})

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

cache = load_json_cache(CACHE_FILE)
cache_context = load_json_cache(CONTEXT_CACHE_FILE)

for doc in documents:
    chunks = splitter.split_text(doc["content"])
    for chunk in chunks:
        context = generate_context(doc["content"], chunk, cache_context)
        contextualized_text = f"{context}\n\n{chunk}"
        embedding = get_embedding(contextualized_text, cache)
        # add the embedding + contextualized_text + metadatas in vector db
        collection.add(
            ids=[hash_text(chunk)],
            embeddings=[embedding],
            documents=[contextualized_text],
            metadatas=[{"source": doc["source"]}],
        )

save_json_cache(CACHE_FILE, cache)
save_json_cache(CONTEXT_CACHE_FILE, cache_context)

print(f"Total items in collection: {collection.count()}")
