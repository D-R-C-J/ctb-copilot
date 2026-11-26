from .corpus import load_passages

# Produces a dictionary where each key is a passage ID and each value is that passage's vector
def build_index(passages, model) -> dict:
    #passages_list = load_passages(passages)
    
    ids = [i["id"] for i in passages]
    texts = [f"passage: {i["text"]}" for i in passages]
    embeddings = model.encode(texts, normalize_embeddings=True)
    for i in range(len(embeddings)):
        print(f"ID: {ids[i]}\nEmbedding: {embeddings[i]}\n\n\n")
    return dict(zip(ids, embeddings, strict=True))
