
from neo4j import GraphDatabase
import sys

# Connection details
URI = "bolt://neo4j:7687"
AUTH = ("neo4j", "password")

def inspect_ar_chunks():
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Inspecting chunks for 'implementingregulation'...")
    
    query = """
    MATCH (d:Document {document_id: 'implementingregulation'})-[:HAS_CHUNK]->(c:Chunk)
    RETURN elementId(c) as id, c.text as text
    LIMIT 20
    """
    
    with driver.session() as session:
        result = session.run(query)
        count = 0
        for record in result:
            count += 1
            text = record["text"]
            print(f"--- Chunk {record['id']} ---")
            print(f"Text: {text[:200]}...")
            print(f"Bytes: {text[:50].encode('utf-8')}")
            
            # Helper to check for Mojibake
            try:
                latin1 = text.encode("windows-1252")
                reversed_txt = latin1.decode("windows-1256")
                print(f"Reversed Attempt: {reversed_txt[:200]}...")
            except:
                pass

        if count == 0:
            print("No chunks found for 'policiesar'.")

    driver.close()

if __name__ == "__main__":
    inspect_ar_chunks()
