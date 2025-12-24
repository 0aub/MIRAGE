
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Connection details
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ("neo4j", "password")
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
DOC_ID = "document"

def delete_broken():
    print(f"Purging document '{DOC_ID}'...")
    
    # 1. Neo4j Delete
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        query = """
        MATCH (d:Document {document_id: $id})
        DETACH DELETE d
        """
        # Also clean orphaned chunks just in case
        query_chunks = """
        MATCH (c:Chunk)
        WHERE c.document_id = $id
        DETACH DELETE c
        """
        
        with driver.session() as session:
            session.run(query, id=DOC_ID)
            session.run(query_chunks, id=DOC_ID)
            print("Deleted from Neo4j.")
        driver.close()
    except Exception as e:
        print(f"Neo4j Delete Failed: {e}")

    # 2. Qdrant Delete
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collection_name = "mirage_chunks"
        
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=DOC_ID)
                    )
                ]
            )
        )
        print("Deleted from Qdrant.")
    except Exception as e:
        print(f"Qdrant Delete Failed: {e}")

if __name__ == "__main__":
    delete_broken()
