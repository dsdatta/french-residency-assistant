from cache_utils import load_json_cache
from llm_client import get_embedding
from vector_store import collection

cache = load_json_cache("embeddings_cache.json")


# Querying from db
print(f"Total items in collection :{collection.count()}")
results = collection.get(limit=2, include=["documents", "metadatas"])

for i in range(len(results["ids"])):
    print(f"\n--- ID : {results['ids'][i]}---")
    print(f"Source: {results['metadatas'][i]['source']}")
    print(f"Text: {results['documents'][i][:150]}.....")

# Query db test semantic search(searches by meanings) 1 vector
query_text = "What documents do I need for a carte de résident?"
query_embedding = get_embedding(query_text, cache)

search_result = collection.query(query_embeddings=[query_embedding], n_results=3)

for i in range(len(search_result["ids"][0])):
    print(f"\nResult {i+1}:")
    print(f"Source: {search_result['metadatas'][0][i]['source']}")
    print(f"Distance: {search_result['distances'][0][i]}")
    print(f"Text: {search_result['documents'][0][i][:200]}...")


# Query db test semantic search(searches by meanings) 2 vectors
query_text1 = "What documents do I need for a carte de résident?"
query_text2 = "What is the Standard validity length of Carte de séjour pluriannuelle?"
query_embedding1 = get_embedding(query_text1, cache)
query_embedding2 = get_embedding(query_text2, cache)

search_result = collection.query(
    query_embeddings=[query_embedding1, query_embedding2], n_results=3
)

# results for query 1
for i in range(len(search_result["ids"][0])):
    print(f"\nResult : {i+1}:")
    print(f"Source: {search_result['metadatas'][0][i]['source']}")
    print(f"Distance: {search_result['distances'][0][i]}")
    print(f"Text: {search_result['documents'][0][i][:200]}...")

# results for query 2
for i in range(len(search_result["ids"][1])):
    print(f"\nResult : {i+1}:")
    print(f"Source: {search_result['metadatas'][1][i]['source']}")
    print(f"Distance: {search_result['distances'][1][i]}")
    print(f"Text: {search_result['documents'][1][i][:200]}...")
