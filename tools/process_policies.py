"""
Process all policy PDFs with entity extraction and graph generation.
Observe the extraction process in real-time.
"""

import os
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, '/app')

from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import fitz  # PyMuPDF

from src.core.graph_builder.neo4j_client import Neo4jClient
from src.core.graph_builder.llm_entity_extractor import LLMEntityExtractor


# Configuration
POLICIES_DIR = "/app/data/policies"
TGI_ENDPOINT = "http://tgi:80"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def detect_language(text: str) -> str:
    """Detect if text is Arabic or English."""
    arabic_chars = len([c for c in text if '\u0600' <= c <= '\u06FF'])
    ratio = arabic_chars / max(len(text), 1)
    return "ar" if ratio > 0.3 else "en"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def main():
    """Process all policy documents."""

    # Initialize components
    logger.info("=" * 60)
    logger.info("INITIALIZING COMPONENTS")
    logger.info("=" * 60)

    logger.info("Loading embedding model (768 dims)...")
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    logger.info("Connecting to Neo4j...")
    neo4j = Neo4jClient()
    neo4j.connect()

    logger.info("Connecting to Qdrant...")
    qdrant = QdrantClient(host='qdrant', port=6333)

    logger.info("Initializing LLM Entity Extractor (NO FIXED FILTERS)...")
    extractor = LLMEntityExtractor()

    # Get list of PDFs
    pdf_files = sorted([f for f in os.listdir(POLICIES_DIR) if f.endswith('.pdf')])
    logger.info(f"\nFound {len(pdf_files)} PDF files to process")

    total_entities = 0
    total_relationships = 0
    total_chunks = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info("=" * 60)
        logger.info(f"PROCESSING [{i}/{len(pdf_files)}]: {pdf_file}")
        logger.info("=" * 60)

        pdf_path = os.path.join(POLICIES_DIR, pdf_file)
        doc_id = pdf_file.replace(" ", "_").replace(".pdf", "").lower()

        # Extract text
        logger.info("Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_path)
        language = detect_language(text)
        logger.info(f"  Language: {language}, Total chars: {len(text)}")

        # Create document node
        with neo4j.driver.session() as session:
            session.run("""
                MERGE (d:Document {document_id: $doc_id})
                SET d.title = $title,
                    d.language = $language,
                    d.content_type = 'file',
                    d.total_chars = $total_chars,
                    d.created_at = datetime().epochMillis
            """, doc_id=doc_id, title=pdf_file.replace(".pdf", ""),
                language=language, total_chars=len(text))

        # Chunk text
        chunks = chunk_text(text)
        logger.info(f"  Created {len(chunks)} chunks")

        # Process each chunk
        doc_entities = []
        doc_relationships = []

        for j, chunk in enumerate(chunks, 1):
            if len(chunk) < 50:  # Skip very short chunks
                continue

            logger.info(f"  Chunk [{j}/{len(chunks)}] ({len(chunk)} chars)...")

            # Extract entities and relationships using LLM
            try:
                result = extractor._extract_from_chunk(chunk, language)
                entities = result.get("entities", [])
                relationships = result.get("relationships", [])

                if entities:
                    logger.info(f"    -> Extracted {len(entities)} entities: {[e.get('text', '')[:30] for e in entities[:5]]}")
                if relationships:
                    logger.info(f"    -> Extracted {len(relationships)} relationships")

                doc_entities.extend(entities)
                doc_relationships.extend(relationships)

            except Exception as e:
                logger.error(f"    -> Error: {e}")
                continue

            # Create chunk embedding and store in Qdrant
            embedding = embedding_model.encode(chunk).tolist()
            chunk_id = f"{doc_id}_c{j}"

            point = PointStruct(
                id=abs(hash(chunk_id)) % (2**63),
                vector=embedding,
                payload={
                    "text": chunk,
                    "language": language,
                    "document_id": doc_id,
                    "chunk_id": chunk_id,
                }
            )
            qdrant.upsert(collection_name="mirage_chunks", points=[point])

            # Store chunk in Neo4j
            with neo4j.driver.session() as session:
                session.run("""
                    MATCH (d:Document {document_id: $doc_id})
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text, c.language = $language, c.document_id = $doc_id
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, doc_id=doc_id, chunk_id=chunk_id, text=chunk[:1000], language=language)

        # Deduplicate and store entities
        seen_entities = set()
        unique_entities = []
        for e in doc_entities:
            key = e.get("text", "").lower().strip()
            if key and key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(e)

        logger.info(f"\n  DOCUMENT SUMMARY:")
        logger.info(f"    Chunks: {len(chunks)}")
        logger.info(f"    Raw entities: {len(doc_entities)} -> Unique: {len(unique_entities)}")
        logger.info(f"    Relationships: {len(doc_relationships)}")

        # Store entities in Neo4j
        with neo4j.driver.session() as session:
            for entity in unique_entities:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.language = $language,
                        e.importance = $importance,
                        e.source_documents = CASE
                            WHEN e.source_documents IS NULL THEN [$doc_id]
                            WHEN NOT $doc_id IN e.source_documents THEN e.source_documents + $doc_id
                            ELSE e.source_documents
                        END
                """, name=entity.get("text", ""),
                    type=entity.get("type", "CONCEPT"),
                    language=language,
                    importance=entity.get("importance", "medium"),
                    doc_id=doc_id)

        # Store relationships in Neo4j
        with neo4j.driver.session() as session:
            for rel in doc_relationships:
                source = rel.get("source", "")
                target = rel.get("target", "")
                # Clean relationship type - Neo4j only allows letters, numbers, underscores
                rel_type = rel.get("type", "RELATED_TO").upper()
                rel_type = ''.join(c if c.isalnum() or c == '_' else '_' for c in rel_type)
                rel_type = '_'.join(filter(None, rel_type.split('_')))  # Remove consecutive underscores
                rel_type = rel_type or "RELATED_TO"

                if source and target and source.lower() != target.lower():
                    session.run("""
                        MATCH (e1:Entity {name: $source})
                        MATCH (e2:Entity {name: $target})
                        MERGE (e1)-[r:""" + rel_type + """]->(e2)
                        SET r.weight = $weight,
                            r.source_documents = CASE
                                WHEN r.source_documents IS NULL THEN [$doc_id]
                                ELSE r.source_documents + $doc_id
                            END
                    """, source=source, target=target,
                        weight=rel.get("weight", 0.5), doc_id=doc_id)

        total_entities += len(unique_entities)
        total_relationships += len(doc_relationships)
        total_chunks += len(chunks)

        logger.info(f"\n  Stored in Neo4j and Qdrant\n")

    # Final summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Documents processed: {len(pdf_files)}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Total unique entities: {total_entities}")
    logger.info(f"Total relationships: {total_relationships}")

    # Verify in databases
    with neo4j.driver.session() as session:
        docs = session.run("MATCH (d:Document) RETURN count(d) as count").single()['count']
        ents = session.run("MATCH (e:Entity) RETURN count(e) as count").single()['count']
        rels = session.run("MATCH ()-[r]->() WHERE NOT type(r) IN ['HAS_CHUNK', 'MENTIONS'] RETURN count(r) as count").single()['count']
        logger.info(f"\nNeo4j Stats:")
        logger.info(f"  Documents: {docs}")
        logger.info(f"  Entities: {ents}")
        logger.info(f"  Entity-Entity Relationships: {rels}")

    collection = qdrant.get_collection("mirage_chunks")
    logger.info(f"\nQdrant Stats:")
    logger.info(f"  Chunks: {collection.points_count}")


if __name__ == "__main__":
    main()
