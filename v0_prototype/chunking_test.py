import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

documents = []

for filename in os.listdir(data_dir):

    if filename.endswith(".txt"):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            documents.append({"source": filename, "content": f.read()})

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

all_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc["content"])
    for chunk in chunks:
        all_chunks.append({"source": doc["source"], "text": chunk})

print(f"Loaded {len(documents)} documents, produced {len(all_chunks)} chunks\n")

for c in all_chunks:
    print(f"----from {c['source']} ----")
    print(c["text"])
    print()
