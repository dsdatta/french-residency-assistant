from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            documents.append({"source": filename, "content": f.read()})

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def generate_context(full_document: str, chunk: str) -> str:
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

    return response.choices[0].message.content.strip()


contextualized_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc["content"])
    print(f"Processing {doc['source']} - {len(chunks)} chunks...")
    for chunk in chunks:
        context = generate_context(doc["content"], chunk)
        contextualized_text = f"{context}\n\n{chunk}"
        contextualized_chunks.append(
            {"source": doc["source"], "text": contextualized_text}
        )

# show first 2 as sanity check
for c in contextualized_chunks[:2]:
    print(f"\n ---from {c["source"]} ")
    print(c["text"])
