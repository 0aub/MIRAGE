
from neo4j import GraphDatabase
import sys

# Connection details
URI = "bolt://neo4j:7687"
AUTH = ("neo4j", "password")

def repair_db():
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Starting repair process...")
    
    # 1. Fetch all chunks that might be corrupted (contain non-ascii)
    # Actually, let's just fetch all and check in python to be safe and precise
    query_fetch = """
    MATCH (c:Chunk)
    WHERE c.text IS NOT NULL
    RETURN elementId(c) as id, c.text as text
    """
    
    query_update = """
    MATCH (c:Chunk)
    WHERE elementId(c) = $id
    SET c.text = $new_text
    RETURN count(c) as updated
    """
    
    repaired_count = 0
    scanned_count = 0
    
    with driver.session() as session:
        result = session.run(query_fetch)
        
        updates = []
        
        for record in result:
            scanned_count += 1
            node_id = record["id"]
            text = record["text"]
            
            # Check for Mojibake (Latin1 chars representing Arabic)
            # Heuristic: If we can reverse 1252->1256 and get Arabic, it's a match.
            try:
                # 1. Encode as Windows-1252 (Latin-1 superset, usually what Python uses for "unknown 8-bit")
                # But wait, if it was read as UTF-8 from disk (wrongly), the bytes 0x80+ might have been 
                # interpreted as valid UTF-8 sequences (unlikely for 1256) OR replaced with replacement char?
                # If "errors=replace" was used, we lost data ().
                # If "errors=ignore", we lost data.
                # If "open(..., encoding='utf-8')" was used on 1256 file:
                # 0xC7 (1256 Alef) is invalid start byte in UTF-8.
                # So python 'utf-8' decoder would raise Error.
                # UNLESS `errors='ignore'` or `replace` was default? No, strict is default.
                # SO: How did the data get in?
                # Maybe the file was uploaded as binary, then "read as string"?
                # Or maybe the browser sent it?
                # The user said: "Š ÅÅ¼".
                # 0x8A (Š) in 1252.
                # 0xC5 (Å) in 1252.
                # This strongly implies the bytes 0x8A, 0xC5 are present in the string (as unicode codepoints U+008A, U+00C5).
                
                latin1_bytes = text.encode("windows-1252")
                candidate = latin1_bytes.decode("windows-1256")
                
                # Check if candidate has Arabic
                has_arabic = any('\u0600' <= c <= '\u06FF' for c in candidate)
                
                # Check if original had Arabic (if original already had Arabic, don't touch!)
                original_has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
                
                if has_arabic and not original_has_arabic:
                    print(f"Repearing Chunk {node_id}...")
                    print(f"Original: {text[:50]}...")
                    print(f"Repaired: {candidate[:50]}...")
                    updates.append({"id": node_id, "new_text": candidate})
                    repaired_count += 1
                    
            except UnicodeEncodeError:
                # Can't encode as 1252 (contains chars > 255), so probably mixed or proper unicode
                pass
            except UnicodeDecodeError:
                pass
            except Exception as e:
                pass
                
        # Batch execute updates
        print(f"Applying {len(updates)} updates...")
        for up in updates:
            session.run(query_update, id=up["id"], new_text=up["new_text"])
            
    driver.close()
    print(f"Finished. Scanned: {scanned_count}, Repaired: {repaired_count}")

if __name__ == "__main__":
    repair_db()
