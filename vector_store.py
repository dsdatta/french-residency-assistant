import os
import chromadb

# db path and creation
script_dir = os.path.dirname(os.path.abspath(__file__))
chroma_path = os.path.join(script_dir, "chroma")

client_db = chromadb.PersistentClient(path=chroma_path)
collection = client_db.get_or_create_collection(name="residency_docs")
