
from neo4j import GraphDatabase
import sys

# Connection details
URI = "bolt://neo4j:7687"
AUTH = ("neo4j", "password")

def audit_db():
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Auditing for non-ASCII text...")
    
    # Regex for non-ascii
    query = """
    MATCH (c:Chunk)
    WHERE c.text =~ '.*[^\\u0000-\\u007F].*'
    RETURN elementId(c) as id, c.text as text
    LIMIT 20
    """
    
    with driver.session() as session:
        result = session.run(query)
        count = 0
        for record in result:
            count += 1
            text = record["text"]
            print(f"--- Chunk {record['id']} [Non-ASCII] ---")
            print(f"Text: {text[:200]}")
            print(f"Repr: {ascii(text[:200])}")
            
            # Debug Code Points
            ords = [ord(c) for c in text if ord(c) > 127]
            print(f"High Code Points: {ords[:20]}")

    driver.close()
    print(f"Found {count} non-ASCII chunks (Limited to 20).")

if __name__ == "__main__":
    audit_db()
