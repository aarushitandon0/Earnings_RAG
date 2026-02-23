import os
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

DATA_PATH = Path("data/transcripts")
CHROMA_PATH = "chroma_db"
CHUNK_SIZE = 400      # words per chunk
CHUNK_OVERLAP = 50    # word overlap between chunks

def parse_filename(filename):
    # Handles format like: 2019-Dec-18-MU.txt → ("MU", "2019-Dec-18")
    name = filename.replace(".txt", "")
    parts = name.split("-")
    ticker = parts[-1]
    date = "-".join(parts[:-1])
    return ticker, date

def load_transcripts(data_path):
    documents = []
    path = Path(data_path)
    txt_files = list(path.glob("*.txt"))
    
    if not txt_files:
        print(" No .txt files found! Check your data/transcripts folder.")
        return []
    
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            ticker, date = parse_filename(txt_file.name)
            documents.append({
                "text": text,
                "source": f"{ticker} | {date}",
                "filename": txt_file.name
            })
            print(f"   Loaded: {txt_file.name}")
        except Exception as e:
            print(f"    Skipped {txt_file.name}: {e}")
    
    return documents

def chunk_text(text, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk_words) > 80:  # skip tiny end chunks
            chunks.append({
                "text": chunk,
                "source": source
            })
    return chunks

def build_vectorstore():
    print("\n Loading transcripts...")
    documents = load_transcripts(DATA_PATH)
    
    if not documents:
        return
    
    print(f"\n  Chunking {len(documents)} transcripts...")
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"], doc["source"])
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")
    
    print(f"\n Total chunks: {len(all_chunks)}")
    
    print("\n🔮 Setting up ChromaDB with sentence-transformers embeddings...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Delete old collection if rebuilding
    try:
        client.delete_collection("transcripts")
        print("    Cleared old collection")
    except:
        pass
    
    collection = client.create_collection("transcripts", embedding_function=ef)
    
    print("\n Embedding and storing chunks (this takes 5-10 mins)...")
    
    texts = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"]} for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    
    # Insert in batches of 100
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"  Stored {min(i+batch_size, len(all_chunks))}/{len(all_chunks)} chunks...", end="\r")
    
    print(f"\n\n Done! Vector store built with {len(all_chunks)} chunks.")
    print(f" Saved to: {CHROMA_PATH}/")

if __name__ == "__main__":
    build_vectorstore()