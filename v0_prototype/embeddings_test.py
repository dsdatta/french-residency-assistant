from dotenv import load_dotenv
import os
from openai import OpenAI
import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

# setup client and load api key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# fn generate hash
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# cache file
CACHE_FILE = "embeddings_cache.json"


# fn load cache
def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# fn save cache
def save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


# global var to count api calls
countApiCall = 0


# fn embedding the chunks
def get_embedding(text: str, cache: dict) -> list:
    global countApiCall
    key = hash_text(text)

    if key in cache:
        print(f"key detected in cache for {key}")
        return cache[key]

    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    countApiCall += 1
    embedding = response.data[0].embedding
    print(
        f"api  response call recieved: {embedding} with count api call: {countApiCall}"
    )
    cache[key] = embedding
    return embedding


CONTEXT_CACHE_FILE = "context_cache.json"


def load_context_cache() -> dict:
    if os.path.exists(CONTEXT_CACHE_FILE):
        with open(CONTEXT_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_context_cache(cache: dict):
    with open(CONTEXT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


# fn contextual retrieval (generates a meaninful context before appending them to the chunks)
def generate_context(full_document: str, chunk: str, context_cache: dict) -> str:
    global countApiCall
    key = hash_text(full_document + chunk)  # hash the stable INPUT, not the output

    if key in context_cache:
        return context_cache[key]
    prompt = f"""Here is a full document:
<document>
{full_document}
</document>

Here is a specific chunk from that document:
<chunk>
{chunk}
</chunk>

Write 1-2 short sentences that situate this chunk within the overall document, so it makes sense if read on its own. Be concise. Only output the context sentences, nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    countApiCall += 1
    context = response.choices[0].message.content.strip()
    context_cache[key] = context
    return context


# file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            documents.append({"source": filename, "content": f.read()})

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

cache = load_cache()
cache_context = load_context_cache()

all_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc["content"])
    for chunk in chunks:
        context = generate_context(doc["content"], chunk, cache_context)
        contextualized_text = f"{context}\n\n{chunk}"
        embedding = get_embedding(contextualized_text, cache)
        all_chunks.append(
            {
                "source": doc["source"],
                "text": contextualized_text,
                "embedding": embedding,
            }
        )

save_cache(cache)
save_context_cache(cache_context)
