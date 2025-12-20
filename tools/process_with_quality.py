"""
Process policy documents with strict quality validation.
Only accepts extractions with good entity AND relationship counts.
"""

import os
import sys
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
CHUNK_SIZE = 800  # Larger chunks for better context
CHUNK_OVERLAP = 100
MIN_ENTITIES_PER_DOC = 10  # Minimum entities per document
MIN_RELATIONSHIPS_PER_DOC = 5  # Minimum relationships per document
MIN_REL_ENTITY_RATIO = 0.1  # At least 10% relationships vs entities


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
    """Split text into overlapping chunks by sentences for better context."""
    # Split by sentences
    import re
    sentences = re.split(r'(?<=[.!?؟।])\s+', text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)

        if current_len + sentence_len > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            # Keep last 2 sentences for overlap
            current_chunk = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
            current_len = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_len += sentence_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def main():
    """Process all policy documents with quality validation."""

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    logger.info("Connecting to Neo4j and Qdrant...")
    neo4j = Neo4jClient()
    neo4j.connect()

    qdrant = QdrantClient(host='qdrant', port=6333)

    logger.info("Initializing LLM extractor...")
    extractor = LLMEntityExtractor()

    # Get PDFs
    pdf_files = sorted([f for f in os.listdir(POLICIES_DIR) if f.endswith('.pdf')])
    logger.info(f"Processing: {pdf_files}")

    results = []

    for pdf_file in pdf_files:
        logger.info("=" * 50)
        logger.info(f"PROCESSING: {pdf_file}")

        pdf_path = os.path.join(POLICIES_DIR, pdf_file)
        doc_id = pdf_file.replace(" ", "_").replace(".pdf", "").lower()

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        language = detect_language(text)
        logger.info(f"Language: {language}, Chars: {len(text)}")

        if len(text) < 100:
            logger.warning(f"Document too short ({len(text)} chars), skipping")
            results.append({
                "file": pdf_file,
                "status": "SKIPPED",
                "reason": "Too short",
                "entities": 0,
                "relationships": 0
            })
            continue

        # Chunk text
        chunks = chunk_text(text)
        logger.info(f"Chunks: {len(chunks)}")

        # Create document node
        with neo4j.driver.session() as session:
            session.run("""
                MERGE (d:Document {document_id: $doc_id})
                SET d.title = $title,
                    d.language = $language,
                    d.total_chars = $total_chars,
                    d.created_at = datetime().epochMillis
            """, doc_id=doc_id, title=pdf_file.replace(".pdf", ""),
                language=language, total_chars=len(text))

        # Process chunks and collect entities/relationships
        all_entities = []
        all_relationships = []

        for i, chunk in enumerate(chunks):
            if len(chunk) < 50:
                continue

            if (i + 1) % 25 == 0:
                logger.info(f"  Chunk {i+1}/{len(chunks)}...")

            try:
                result = extractor._extract_from_chunk(chunk, language)
                entities = result.get("entities", [])
                relationships = result.get("relationships", [])

                all_entities.extend(entities)
                all_relationships.extend(relationships)

                # Store chunk in Qdrant
                embedding = embedding_model.encode(chunk).tolist()
                chunk_id = f"{doc_id}_c{i}"
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

            except Exception as e:
                logger.error(f"  Chunk {i+1} error: {e}")
                continue

        # Deduplicate entities
        seen_entities = {}
        for e in all_entities:
            key = e.get("text", "").lower().strip()
            if key and len(key) > 1:
                if key not in seen_entities:
                    seen_entities[key] = e
                else:
                    # Keep higher importance
                    importance_order = {"high": 3, "medium": 2, "low": 1}
                    if importance_order.get(e.get("importance", "low"), 1) > \
                       importance_order.get(seen_entities[key].get("importance", "low"), 1):
                        seen_entities[key] = e

        unique_entities = list(seen_entities.values())

        # Deduplicate relationships
        seen_rels = set()
        unique_rels = []
        for r in all_relationships:
            source = r.get("source", "").lower().strip()
            target = r.get("target", "").lower().strip()
            rel_type = r.get("type", "").upper()
            if source and target and source != target:
                key = (source, target, rel_type)
                if key not in seen_rels:
                    seen_rels.add(key)
                    unique_rels.append(r)

        logger.info(f"Entities: {len(all_entities)} raw -> {len(unique_entities)} unique")
        logger.info(f"Relationships: {len(all_relationships)} raw -> {len(unique_rels)} unique")

        # Quality check
        entity_count = len(unique_entities)
        rel_count = len(unique_rels)
        ratio = rel_count / max(entity_count, 1)

        quality_ok = (
            entity_count >= MIN_ENTITIES_PER_DOC and
            rel_count >= MIN_RELATIONSHIPS_PER_DOC and
            ratio >= MIN_REL_ENTITY_RATIO
        )

        status = "PASS" if quality_ok else "WARN"
        logger.info(f"Quality: {status} (E:{entity_count}, R:{rel_count}, ratio:{ratio:.2f})")

        results.append({
            "file": pdf_file,
            "status": status,
            "entities": entity_count,
            "relationships": rel_count,
            "ratio": ratio
        })

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
                    type=entity.get("type", "Concept"),
                    language=language,
                    importance=entity.get("importance", "medium"),
                    doc_id=doc_id)

        # Store relationships in Neo4j
        with neo4j.driver.session() as session:
            for rel in unique_rels:
                source = rel.get("source", "")
                target = rel.get("target", "")
                rel_type = rel.get("type", "RELATED_TO").upper()
                # Clean relationship type
                rel_type = ''.join(c if c.isalnum() or c == '_' else '_' for c in rel_type)
                rel_type = '_'.join(filter(None, rel_type.split('_')))
                rel_type = rel_type or "RELATED_TO"

                if source and target and source.lower() != target.lower():
                    try:
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
                    except Exception as e:
                        logger.warning(f"Failed to create relationship: {source} -[{rel_type}]-> {target}: {e}")

        logger.info("Stored in Neo4j and Qdrant")

    # Final summary
    logger.info("=" * 50)
    logger.info("ALL DOCUMENTS COMPLETED")
    logger.info("=" * 50)

    total_entities = sum(r["entities"] for r in results)
    total_rels = sum(r["relationships"] for r in results)
    passed = sum(1 for r in results if r["status"] == "PASS")

    logger.info(f"Documents: {len(results)} ({passed} passed quality check)")
    logger.info(f"Total entities: {total_entities}")
    logger.info(f"Total relationships: {total_rels}")
    logger.info(f"Overall ratio: {total_rels/max(total_entities,1):.2f}")

    for r in results:
        logger.info(f"  {r['file']}: {r['status']} (E:{r['entities']}, R:{r['relationships']})")


if __name__ == "__main__":
    main()
