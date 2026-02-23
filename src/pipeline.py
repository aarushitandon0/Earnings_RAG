from src.rewriter import rewrite_query
from src.hybrid_retriever import hybrid_retrieve
from src.reranker import rerank
from src.crag import apply_crag
from src.generator import generate_answer

def run_pipeline(query: str) -> dict:
    # Stage 1: Query Rewriting
    print("   Stage 1: Rewriting query...")
    rewritten = rewrite_query(query)
    
    # Stage 2: Hybrid Retrieval (Vector + BM25)
    print("   Stage 2: Hybrid retrieval (Vector + BM25)...")
    raw_chunks = hybrid_retrieve(rewritten, top_k=20)
    
    # Stage 3: Re-ranking
    print("   Stage 3: Re-ranking top chunks...")
    reranked = rerank(rewritten, raw_chunks, top_k=3)
    
    # Stage 4: CRAG - Grade relevance, correct if needed
    print("   Stage 4: Corrective RAG grading...")
    final_chunks, crag_status = apply_crag(query, reranked, hybrid_retrieve)
    
    # Stage 5: Generate Answer
    print("   Stage 5: Generating answer...")
    answer = generate_answer(query, final_chunks)
    
    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "reranked_chunks": reranked,
        "final_chunks": final_chunks,
        "crag_status": crag_status,
        "answer": answer
    }