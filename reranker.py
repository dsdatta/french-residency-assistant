from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


def rerank(query, contents):
    # pairs = [(query, text) for text in chunk_texts]
    chunk_texts = contents["documents"]
    pairs = [(query, text) for text in chunk_texts]
    scores = model.predict(pairs)

    chunk_text_withsource = []
    for i in range(len(contents["ids"])):
        chunk_text_withsource.append(
            (contents["documents"][i], contents["metadatas"][i]["source"])
        )

    reranked = sorted(zip(scores, chunk_text_withsource), reverse=True)
    return reranked


# reranked_result = rerank(query, contents)
# print(f"Result Rerank : {reranked_result}")
