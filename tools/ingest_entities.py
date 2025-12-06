#!/usr/bin/env python3
"""
Ingest missing entity documents with proper embeddings.
This script generates embeddings for chunks before adding to Qdrant.
"""
import sys
sys.path.insert(0, '/app')

from src.core.graph_builder import Neo4jClient
from src.core.vector_store import QdrantVectorStore
from src.core.embeddings.jina_embedder import JinaEmbedder
import hashlib

def main():
    print("=" * 60)
    print("MIRAGE Entity Document Ingestion V2 (with embeddings)")
    print("=" * 60)

    # Connect to databases
    print("\n[1/5] Connecting to databases...")
    neo4j = Neo4jClient()
    neo4j.connect()
    qdrant = QdrantVectorStore()
    print("  Connected!")

    # Initialize embedder
    print("\n[2/5] Initializing embedding model...")
    embedder = JinaEmbedder()
    print(f"  Model: {embedder.model}, Dim: {embedder.embedding_dim}")

    # Entity documents to ingest
    documents = [
        {
            "file": "/app/missing_entities/zatca.txt",
            "title": "هيئة الزكاة والضريبة والجمارك",
            "doc_id": "entity_zatca"
        },
        {
            "file": "/app/missing_entities/elm.txt",
            "title": "شركة علم",
            "doc_id": "entity_elm"
        },
        {
            "file": "/app/missing_entities/yamamah.txt",
            "title": "منصة يمامة",
            "doc_id": "entity_yamamah"
        }
    ]

    print(f"\n[3/5] Processing {len(documents)} documents...")

    for i, doc in enumerate(documents, 1):
        print(f"\n  [{i}/{len(documents)}] Processing: {doc['title']}")

        try:
            # Read content
            with open(doc["file"], "r", encoding="utf-8") as f:
                content = f.read()

            print(f"    Content: {len(content)} chars")

            # Create document in Neo4j
            neo4j.store_document_metadata(
                document_id=doc["doc_id"],
                title=doc["title"],
                content_type="file",
                metadata={"source": "manual_entity_ingestion"}
            )

            # Store full text
            with neo4j.driver.session() as session:
                session.run("""
                    MATCH (d:Document {document_id: $doc_id})
                    SET d.full_text = $content
                """, doc_id=doc["doc_id"], content=content)

            # Chunk the document
            chunks = []
            chunk_size = 500
            overlap = 50
            chunk_idx = 0

            for j in range(0, len(content), chunk_size - overlap):
                chunk_text = content[j:j + chunk_size]
                if len(chunk_text) > 50:  # Skip tiny chunks
                    chunk_id = f"{doc['doc_id']}_chunk_{chunk_idx}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "document_id": doc["doc_id"],
                        "metadata": {
                            "title": doc["title"],
                            "chunk_index": chunk_idx
                        }
                    })
                    chunk_idx += 1

            print(f"    Created {len(chunks)} chunks")

            # Generate embeddings for chunks
            print(f"    Generating embeddings...")
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embedder.embed(chunk_texts, task="retrieval.passage")

            # Add embeddings to chunks
            for idx, chunk in enumerate(chunks):
                # Handle both numpy array and list
                emb = embeddings[idx] if len(embeddings.shape) > 1 else embeddings
                chunk["embedding"] = emb.tolist() if hasattr(emb, 'tolist') else list(emb)

            print(f"    Embedding dim: {len(chunks[0]['embedding'])}")

            # Add to Qdrant
            vector_result = qdrant.add_chunks(chunks, doc["doc_id"])
            print(f"    Added to vector DB: {vector_result}")

            # Create basic entities from title
            entity_name = doc["title"]
            entity_id = hashlib.md5(entity_name.encode()).hexdigest()[:12]

            with neo4j.driver.session() as session:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    ON CREATE SET
                        e.entity_id = $entity_id,
                        e.type = 'ORGANIZATION',
                        e.source_documents = [$doc_id],
                        e.created_at = datetime()
                    ON MATCH SET
                        e.source_documents = CASE
                            WHEN NOT $doc_id IN e.source_documents
                            THEN e.source_documents + $doc_id
                            ELSE e.source_documents
                        END
                """, name=entity_name, entity_id=entity_id, doc_id=doc["doc_id"])

            print(f"    Created entity: {entity_name}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Verification
    print("\n[4/5] Verifying ingestion...")

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(host="qdrant", port=6333)
        collection_info = qc.get_collection("mirage_chunks")
        print(f"  Qdrant collection: {collection_info.points_count} points")
    except Exception as e:
        print(f"  Qdrant check error: {e}")

    # Check Neo4j
    try:
        result = neo4j.execute_query("""
            MATCH (d:Document)
            WHERE d.document_id STARTS WITH 'entity_'
            RETURN d.document_id as id, d.title as title
        """)
        print(f"  Neo4j entity docs: {len(result)}")
        for r in result:
            print(f"    - {r['id']}: {r['title']}")
    except Exception as e:
        print(f"  Neo4j check error: {e}")

    print("\n[5/5] Testing retrieval...")

    # Quick test query
    test_queries = ["هيئة الزكاة", "شركة علم", "منصة يمامة"]
    for query in test_queries:
        try:
            # Generate query embedding
            query_emb = embedder.embed(query, task="retrieval.query")
            results = qdrant.search(query_emb.tolist(), top_k=3)

            if results:
                top_result = results[0]
                print(f"  '{query}' -> {top_result.get('document_id', 'unknown')} (score: {top_result.get('score', 0):.3f})")
            else:
                print(f"  '{query}' -> No results")
        except Exception as e:
            print(f"  '{query}' -> Error: {e}")

    print("\n" + "=" * 60)
    print("DONE! Entities ingested with embeddings.")
    print("=" * 60)


if __name__ == "__main__":
    main()
