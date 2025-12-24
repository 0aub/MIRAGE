
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Connection details
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ("neo4j", "password")
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333

def reset_system():
    print("WARNING: INITIATING FULL SYSTEM WIPE")
    print("====================================")
    
    # 1. Wipe Neo4j
    print("Wiping Neo4j Graph...", end=" ")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        driver.close()
        print("DONE.")
    except Exception as e:
        print(f"FAILED: {e}")

    # 2. Wipe Qdrant
    print("Wiping Qdrant Vector Store...", end=" ")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if "mirage_chunks" in collections:
            client.delete_collection("mirage_chunks")
            print("Collection deleted.", end=" ")
            
            # Recreate immediately to be ready for ingestion?
            # Actually, let's let the app recreate it on startup/first use if possible,
            # BUT the app ensures it on init. Since app is running, it won't recreate until restart or check.
            # Best to recreate it fresh now to ensure clean state.
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name="mirage_chunks",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE) # Default dims
            )
            print("Collection recreated empty.")
        else:
            print("Collection did not exist.")
            
    except Exception as e:
        print(f"FAILED: {e}")

    print("System Reset Complete.")

if __name__ == "__main__":
    reset_system()
