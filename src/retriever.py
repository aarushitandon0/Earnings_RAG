import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "chroma_db"

def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("transcripts", embedding_function=ef)
    return collection

def retrieve(query: str, top_k: int = 20) -> list:
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "source": results["metadatas"][0][i]["source"],
            "score": round(1 - results["distances"][0][i], 4)
        })
    return chunks

if __name__ == "__main__":
    test_query = "Apple iPhone revenue sales growth guidance"
    
    print(f" Retrieving for: '{test_query}'")
    print("="*60)
    
    results = retrieve(test_query, top_k=5)
    
    for i, chunk in enumerate(results, 1):
        print(f"\n[{i}] Source : {chunk['source']}")
        print(f"    Score  : {chunk['score']}")
        print(f"    Preview: {chunk['text'][:200]}...")