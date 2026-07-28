from rank_bm25 import BM25Okapi
from vector_store import collection
from llm_client import get_embedding
from cache_utils import load_json_cache

# query_text = "What is the Standard validity length of Carte de séjour pluriannuelle?"
# query_text = "What is the validity length of carte de séjour?"
cache = load_json_cache("embeddings_cache.json")


# hybrid search score
def hybrid_search_score(query):
    results = collection.get(include=["documents", "metadatas"])
    tokenized_documents = []

    for i in range(len(results["ids"])):
        tokenized_doc = results["documents"][i].lower().split()
        tokenized_documents.append(tokenized_doc)

    bm25 = BM25Okapi(tokenized_documents)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # print(scores)

    best_index = scores.argmax()
    # print(f"\nBest match: index {best_index}, score {scores[best_index]}")
    # print(f"Source: {results['metadatas'][best_index]['source']}")
    # print(f"Text: {results['documents'][best_index][:200]}")

    sorted_scores = sorted(zip(scores, results["ids"]), reverse=True)

    return sorted_scores


# vector search score
def vector_search_score(query):
    query_embedding = get_embedding(query, cache)
    search_result = collection.query(query_embeddings=[query_embedding], n_results=3)

    # for i in range(len(search_result["ids"][0])):
    #     print(f"\nResult {i+1}:")
    #     print(f"Source: {search_result['metadatas'][0][i]['source']}")
    #     print(f"Distance: {search_result['distances'][0][i]}")
    #     print(f"Text: {search_result['documents'][0][i][:200]}...")

    return search_result["ids"][0]


# fn to combine hybrid search & vector search to build Reciprocal Rank Fusion
def combine_rrf(query):
    # Reciprocal Rank Fusion scores
    rrf_scores = {}

    vector_ranked_ids = vector_search_score(query)
    # print(f"vector_ranked_ids : {vector_ranked_ids}")
    bm25_ranked_ids = hybrid_search_score(query)
    # print(f"bm25_ranked_ids : {bm25_ranked_ids}")

    for position, chunk_id in enumerate(vector_ranked_ids):
        rank = position + 1
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + 60)
        # print(f"Chunk_id : {chunk_id}, rrf_scores : {rrf_scores}")

    for position, (score, chunk_id) in enumerate(bm25_ranked_ids):
        rank = position + 1
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + 60)

    final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return final_ranking


# takes a list of chunk IDs (e.g., the top 3 from combine_rrf()) and returns their actual text content
def fetch_chunk_texts(chunk_ids):
    text_contents = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    return text_contents


# if __name__ == "__main__":
#     result = combine_rrf()
# print("\n=== Final RRF Ranking ===")
# for chunk_id, score in result:
#     print(f"{chunk_id}: {score:.5f}")
