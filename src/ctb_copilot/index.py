from .corpus import load_passages

# Produces a dictionary where each key is a passage ID and each value is that passage's vector
def build_index(passages, model) -> dict:    
    ids = [i["id"] for i in passages]
    texts = [f"passage: {i["text"]}" for i in passages]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return dict(zip(ids, embeddings, strict=True))
