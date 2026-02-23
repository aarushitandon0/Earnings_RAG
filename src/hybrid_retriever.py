import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

CHROMA_PATH = "chroma_db"

# Cache BM25 index so we don't rebuild every call
_bm25_index = None
_all_chunks = None

def _build_bm25_index():
    global _bm25_index, _all_chunks
    
    if _bm25_index is not None:
        return _bm25_index, _all_chunks
    
    print("   Building BM25 index (first time only)...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("transcripts", embedding_function=ef)
    
    # Pull all documents from ChromaDB
    all_data = collection.get(include=["documents", "metadatas"])
    
    docs = all_data["documents"]
    metas = all_data["metadatas"]
    
    _all_chunks = [
        {"text": docs[i], "source": metas[i]["source"]}
        for i in range(len(docs))
    ]
    
    # Tokenize for BM25
    tokenized = [doc["text"].lower().split() for doc in _all_chunks]
    _bm25_index = BM25Okapi(tokenized)
    
    print(f"   BM25 index built over {len(_all_chunks)} chunks")
    return _bm25_index, _all_chunks

def hybrid_retrieve(query: str, top_k: int = 20) -> list:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("transcripts", embedding_function=ef)
    
    # --- Vector Search ---
    vector_results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    vector_chunks = {}
    for i, doc in enumerate(vector_results["documents"][0]):
        source = vector_results["metadatas"][0][i]["source"]
        score = 1 - vector_results["distances"][0][i]
        key = doc[:100]  # use first 100 chars as unique key
        vector_chunks[key] = {
            "text": doc,
            "source": source,
            "vector_score": round(score, 4),
            "bm25_score": 0.0
        }
    
    # --- BM25 Search ---
    bm25, all_chunks = _build_bm25_index()
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Get top BM25 results
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
    
    # Normalize BM25 scores to 0-1
    max_bm25 = bm25_scores[top_bm25_idx[0]] if bm25_scores[top_bm25_idx[0]] > 0 else 1
    
    for idx in top_bm25_idx:
        chunk = all_chunks[idx]
        key = chunk["text"][:100]
        norm_score = round(float(bm25_scores[idx]) / max_bm25, 4)
        
        if key in vector_chunks:
            vector_chunks[key]["bm25_score"] = norm_score
        else:
            vector_chunks[key] = {
                "text": chunk["text"],
                "source": chunk["source"],
                "vector_score": 0.0,
                "bm25_score": norm_score
            }
    
    # --- Combine Scores (60% vector, 40% BM25) ---
    for key in vector_chunks:
        c = vector_chunks[key]
        c["hybrid_score"] = round(
            0.6 * c["vector_score"] + 0.4 * c["bm25_score"], 4
        )
    
    # Sort by hybrid score
    final = sorted(
        vector_chunks.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )
    
    return final[:top_k]