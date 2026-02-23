from flashrank import Ranker, RerankRequest

# Downloads model on first run (~50MB), cached after that
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

def rerank(query: str, chunks: list, top_k: int = 3) -> list:
    passages = [
        {"id": i, "text": c["text"], "meta": c}
        for i, c in enumerate(chunks)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    reranked = []
    for r in results[:top_k]:
        chunk = r["meta"]
        chunk["rerank_score"] = round(r["score"], 4)
        reranked.append(chunk)
    
    return reranked

if __name__ == "__main__":
    from src.retriever import retrieve

    query = "Apple iPhone revenue sales growth guidance"
    
    print(f"Query: '{query}'")
    print("="*60)
    
    # Get top 20 from vector search
    chunks = retrieve(query, top_k=20)
    
    print("\n BEFORE Re-ranking (top 5 by vector search):")
    for i, c in enumerate(chunks[:5], 1):
        print(f"  [{i}] {c['source']} | score: {c['score']} | {c['text'][:100]}...")
    
    # Re-rank and get top 3
    reranked = rerank(query, chunks, top_k=3)
    
    print("\n AFTER Re-ranking (top 3 by cross-encoder):")
    for i, c in enumerate(reranked, 1):
        print(f"  [{i}] {c['source']} | rerank_score: {c['rerank_score']} | {c['text'][:100]}...")
    
    print("\n Notice how the ordering changed — that's re-ranking working!")