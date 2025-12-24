
from neo4j import GraphDatabase
import sys

# Connection details (adjust if needed)
URI = "bolt://neo4j:7687"
AUTH = ("neo4j", "password")

def inspect_data():
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("Connected to Neo4j.")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    query = """
    MATCH (c:Chunk)
    RETURN c.chunk_id as id, c.text as text
    LIMIT 20
    """

    with driver.session() as session:
        result = session.run(query)
        for record in result:
            text = record["text"]
            print(f"--- Chunk {record['id']} ---")
            print(f"Text: {text}")
            print(f"Bytes: {text.encode('utf-8')}")
            
            # Heuristic check
            try:
                # Try to reverse the Windows-1252 misinterpretation
                # If text is "Š ÅÅ¼", it's Latin-1 bytes of 1256 content
                latin1_bytes = text.encode("windows-1252")
                print(f"Latin1 Bytes: {latin1_bytes}")
                
                decoded_1256 = latin1_bytes.decode("windows-1256")
                print(f"Reversed (1256): {decoded_1256}")
            except Exception as e:
                print(f"Reversal attempt failed: {e}")

    driver.close()

if __name__ == "__main__":
    inspect_data()
