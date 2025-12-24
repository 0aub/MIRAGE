
from neo4j import GraphDatabase
import sys

# Connection details
URI = "bolt://neo4j:7687"
AUTH = ("neo4j", "password")

def inspect_docs():
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Listing Documents...")
    
    query = """
    MATCH (d:Document)
    RETURN d.document_id as id, d.title as title, d.file_type as type
    LIMIT 50
    """
    
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            print(f"ID: {record['id']} | Title: {record['title']} | Type: {record['type']}")
    
    driver.close()

if __name__ == "__main__":
    inspect_docs()
