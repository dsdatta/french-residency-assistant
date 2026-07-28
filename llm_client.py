import os
from dotenv import load_dotenv
from openai import OpenAI
from cache_utils import hash_text

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# global var to count api calls
countApiCall = 0


# fn embedding the chunks
def get_embedding(text: str, cache: dict) -> list:
    global countApiCall
    key = hash_text(text)

    if key in cache:
        return cache[key]

    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    countApiCall += 1
    embedding = response.data[0].embedding
    cache[key] = embedding
    return embedding


# fn contextual retrieval (generates a meaninful context before appending them to the chunks)
def generate_context(full_document: str, chunk: str, context_cache: dict) -> str:
    global countApiCall
    key = hash_text(full_document + chunk)

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
