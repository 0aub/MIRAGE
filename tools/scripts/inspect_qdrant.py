
from qdrant_client import QdrantClient
import sys

# Connection details
HOST = "qdrant"
PORT = 6333

def inspect_qdrant():
    print(f"Connecting to Qdrant at {HOST}:{PORT}...")
    try:
        client = QdrantClient(host=HOST, port=PORT)
        cols = client.get_collections()
        print(f"Collections: {[c.name for c in cols.collections]}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    collection_name = "mirage_chunks"
    
    print(f"Inspecting '{collection_name}'...")
    
    # Scroll through points
    try:
        results, next_page = client.scroll(
            collection_name=collection_name,
            limit=20,
            with_payload=True,
            with_vectors=False
        )
        
        for point in results:
            payload = point.payload
            text = payload.get("text", "")
            doc_id = payload.get("document_id", "")
            
            print(f"--- Chunk {point.id} (Doc: {doc_id}) ---")
            print(f"Text: {text[:100]}...")
            print(f"Bytes: {text[:50].encode('utf-8')}")
            
    except Exception as e:
        print(f"Failed to scroll: {e}")

if __name__ == "__main__":
    inspect_qdrant()
