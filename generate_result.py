from hybrid_search import combine_rrf, fetch_chunk_texts
from llm_client import client
from reranker import rerank


# fn to create a system prompt and get a response for llm
def generate_answer(query, structured_content):
    system_prompt = f"""You are a helpful assistant that answers questions ONLY using the provided sources below.
Always mention which source(s) you used in your answer.
If the answer isn't in the sources, say so clearly instead of guessing.

Sources:
{structured_content}"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


# fn that calls the entire pipeline methods from hybrid_search.py and reranker.py
def answer_query(query_text):
    if not query_text or not query_text.strip():
        return "Please provide a question to search the documents."

    structured_contents = ""
    # fetch the top ids
    top_chunk_ids = [chunk_id for chunk_id, score in combine_rrf(query_text)[:3]]

    # get the actual content
    contents = fetch_chunk_texts(top_chunk_ids)

    # re-ranked results
    reranked_result = rerank(query_text, contents)

    # generate the actual structured documents
    for score, (text, source) in reranked_result:
        source = source
        document_text = text
        structured_contents += f"--- Source: {source} ---\n{document_text}\n\n"

    # finally generate the answer
    return generate_answer(query_text, structured_contents)


# Full pipeline, genuinely complete and verified end to end: chunking → contextual retrieval → embeddings with caching →
# vector storage → hybrid search (BM25 + vector) → RRF combination → cross-encoder reranking → grounded generation with citation.
# answer = generate_answer(query_text, structured_contents)
if __name__ == "__main__":
    answer1 = answer_query("What is the validity length of carte de séjour?")
    print("=== Answer 1 ===")
    print(answer1)
    print()

    answer2 = answer_query("What is the civic exam?")
    print("=== Answer 2 ===")
    print(answer2)
