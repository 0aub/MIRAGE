"""
Re-embed chunks with correct 768-dim model for API compatibility.
Uses existing Neo4j data, just updates Qdrant embeddings.
"""

import os
import sys
sys.path.insert(0, '/app')

from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from src.core.graph_builder.neo4j_client import Neo4jClient


def main():
    """Re-embed all chunks with 768-dim model."""

    # Initialize 768-dim embedding model (matches API)
    logger.info("Loading 768-dim embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # Connect to databases
    logger.info("Connecting to Neo4j...")
    neo4j = Neo4jClient()
    neo4j.connect()

    logger.info("Connecting to Qdrant...")
    qdrant = QdrantClient(host='qdrant', port=6333)

    # Get all chunks from Neo4j
    logger.info("Fetching chunks from Neo4j...")
    with neo4j.driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk)-[:HAS_CHUNK]-(d:Document)
            RETURN c.id as chunk_id, c.text as text, c.language as language,
                   d.id as doc_id, d.title as doc_title, c.page as page
            ORDER BY c.id
        """)
        chunks = list(result)

    logger.info(f"Found {len(chunks)} chunks to re-embed")

    # Process chunks in batches
    batch_size = 50
    points = []

    for i, record in enumerate(chunks):
        text = record['text']
        if not text or len(text) < 10:
            continue

        # Create embedding
        embedding = embedding_model.encode(text).tolist()

        # Create Qdrant point
        point = PointStruct(
            id=abs(hash(record['chunk_id'])) % (2**63),
            vector=embedding,
            payload={
                "text": text,
                "language": record['language'] or "unknown",
                "document_id": record['doc_id'] or "",
                "document_title": record['doc_title'] or "",
                "page_number": record['page'] or 0,
                "chunk_id": record['chunk_id']
            }
        )
        points.append(point)

        # Upload in batches
        if len(points) >= batch_size:
            qdrant.upsert(collection_name="mirage_chunks", points=points)
            logger.info(f"Uploaded {i+1}/{len(chunks)} chunks")
            points = []

    # Upload remaining
    if points:
        qdrant.upsert(collection_name="mirage_chunks", points=points)

    # Verify
    collection = qdrant.get_collection("mirage_chunks")
    logger.info(f"Done! Qdrant now has {collection.points_count} chunks with 768-dim vectors")

    # Count by language
    with neo4j.driver.session() as session:
        ar_count = session.run("MATCH (c:Chunk) WHERE c.language='ar' RETURN count(c) as count").single()['count']
        en_count = session.run("MATCH (c:Chunk) WHERE c.language='en' RETURN count(c) as count").single()['count']
        print(f"\n=== Final Stats ===")
        print(f"Arabic chunks: {ar_count}")
        print(f"English chunks: {en_count}")
        print(f"Total in Qdrant: {collection.points_count}")


if __name__ == "__main__":
    main()
