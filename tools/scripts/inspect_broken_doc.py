
from neo4j import GraphDatabase
import sys

# Connection details
URI = "bolt://neo4j:7687"
AUTH = ("neo4j", "password")

def inspect_broken():
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Inspecting chunks for 'document'...")
    
    query = """
    MATCH (d:Document {document_id: 'document'})-[:HAS_CHUNK]->(c:Chunk)
    RETURN elementId(c) as id, c.text as text
    LIMIT 20
    """
    
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            text = record["text"]
            print(f"--- Chunk {record['id']} ---")
            print(f"Text: {text[:200]}...")
            
    driver.close()

if __name__ == "__main__":
    inspect_broken()
