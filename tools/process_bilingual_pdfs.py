"""
Bilingual PDF Processing Script for NDMO Policies

This script processes both Arabic and English NDMO policy PDFs using Qwen2.5-7B
for entity extraction, creating a unified knowledge graph with both languages.

Pipeline:
1. Extract text from PDFs (Arabic and English)
2. Chunk text with language-aware chunking
3. Use Qwen2.5-7B for entity extraction via TGI
4. Store chunks in Qdrant with language metadata
5. Build knowledge graph in Neo4j with bilingual entities
"""

import os
import sys
import json
import requests
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, '/app')

from loguru import logger
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# Database clients
from src.core.graph_builder.neo4j_client import Neo4jClient
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


@dataclass
class ProcessedChunk:
    """A processed text chunk with metadata."""
    id: str
    text: str
    language: str
    document_id: str
    document_title: str
    page_number: int
    chunk_index: int
    embedding: Optional[List[float]] = None
    entities: Optional[List[Dict]] = None


class BilingualPDFProcessor:
    """Process bilingual PDFs for GraphRAG."""

    def __init__(
        self,
        tgi_endpoint: str = "http://tgi:80",
        neo4j_uri: str = "bolt://neo4j:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        qdrant_host: str = "qdrant",
        qdrant_port: int = 6333,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.tgi_endpoint = tgi_endpoint
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embedding model - using same model as API (768 dims)
        logger.info("Loading embedding model (768 dims for API compatibility)...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

        # Initialize Neo4j
        logger.info("Connecting to Neo4j...")
        self.neo4j = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
        self.neo4j.connect()

        # Initialize Qdrant
        logger.info("Connecting to Qdrant...")
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._ensure_qdrant_collection()

        # Stats tracking
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_created": 0,
            "ar_chunks": 0,
            "en_chunks": 0
        }

    def _ensure_qdrant_collection(self):
        """Ensure Qdrant collection exists."""
        collections = self.qdrant.get_collections().collections
        collection_names = [c.name for c in collections]

        if "mirage_chunks" not in collection_names:
            self.qdrant.create_collection(
                collection_name="mirage_chunks",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            logger.info("Created Qdrant collection: mirage_chunks (768 dims)")

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page information."""
        logger.info(f"Extracting text from: {pdf_path}")

        doc = fitz.open(pdf_path)
        pages = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "page_number": page_num,
                    "text": text.strip()
                })

        doc.close()
        logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
        return pages

    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Arabic or English."""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
        total_chars = len(text.replace(' ', '').replace('\n', ''))

        if total_chars == 0:
            return "en"

        arabic_ratio = arabic_chars / total_chars
        return "ar" if arabic_ratio > 0.3 else "en"

    def chunk_text(self, text: str, language: str) -> List[str]:
        """Split text into chunks with language-aware chunking."""
        if language == "ar":
            # Arabic: split on periods, question marks, or newlines
            separators = ['۔', '؟', '。', '.', '\n\n']
        else:
            # English: standard sentence splitting
            separators = ['. ', '? ', '! ', '\n\n']

        chunks = []
        current_chunk = ""

        # Simple word-based chunking
        words = text.split()

        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                current_chunk += (" " if current_chunk else "") + word
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if len(c) > 50]  # Filter very short chunks

    def extract_entities_with_tgi(self, text: str, language: str) -> List[Dict]:
        """Extract entities using Qwen2.5-7B via TGI."""

        if language == "ar":
            prompt = f"""أنت متخصص في استخراج الكيانات من النصوص العربية.
استخرج الكيانات من النص التالي وقدمها بتنسيق JSON.

النص: {text[:1500]}

أنواع الكيانات المطلوبة:
- ORGANIZATION: منظمات، هيئات حكومية، شركات
- PERSON: أسماء الأشخاص
- POLICY: سياسات، لوائح، قوانين
- CONCEPT: مفاهيم، مصطلحات تقنية
- LOCATION: مواقع، مناطق

قدم الإجابة بتنسيق JSON فقط:
{{"entities": [{{"name": "اسم الكيان", "type": "النوع", "description": "وصف مختصر"}}]}}

JSON:"""
        else:
            prompt = f"""You are an entity extraction specialist.
Extract entities from the following text and provide them in JSON format.

Text: {text[:1500]}

Entity types:
- ORGANIZATION: Government agencies, companies, institutions
- PERSON: Names of people
- POLICY: Policies, regulations, laws
- CONCEPT: Technical concepts, terms
- LOCATION: Places, regions

Provide response in JSON format only:
{{"entities": [{{"name": "entity name", "type": "TYPE", "description": "brief description"}}]}}

JSON:"""

        try:
            response = requests.post(
                f"{self.tgi_endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.1,
                        "do_sample": False,
                        "return_full_text": False
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("generated_text", "")

                # Try to parse JSON from response
                try:
                    # Find JSON in response
                    json_start = generated_text.find('{')
                    json_end = generated_text.rfind('}') + 1

                    if json_start != -1 and json_end > json_start:
                        json_str = generated_text[json_start:json_end]
                        parsed = json.loads(json_str)
                        entities = parsed.get("entities", [])
                        return entities
                except json.JSONDecodeError:
                    pass

            return []

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def store_chunk_in_qdrant(self, chunk: ProcessedChunk):
        """Store chunk in Qdrant."""
        if chunk.embedding is None:
            chunk.embedding = self.create_embedding(chunk.text)

        payload = {
            "text": chunk.text,
            "language": chunk.language,
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "entities": chunk.entities or []
        }

        point = PointStruct(
            id=abs(hash(chunk.id)) % (2**63),  # Convert to positive integer
            vector=chunk.embedding,
            payload=payload
        )

        self.qdrant.upsert(
            collection_name="mirage_chunks",
            points=[point]
        )

    def store_entities_in_neo4j(self, chunk: ProcessedChunk):
        """Store entities and relationships in Neo4j."""
        if not chunk.entities:
            return

        with self.neo4j.driver.session() as session:
            # Create Document node
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.title = $title, d.language = $language
            """, doc_id=chunk.document_id, title=chunk.document_title, language=chunk.language)

            # Create Chunk node
            session.run("""
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text, c.language = $language, c.page = $page
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, chunk_id=chunk.id, text=chunk.text[:500], language=chunk.language,
                page=chunk.page_number, doc_id=chunk.document_id)

            # Create Entity nodes and relationships
            for entity in chunk.entities:
                entity_id = hashlib.md5(f"{entity['name']}_{entity['type']}".encode()).hexdigest()

                session.run("""
                    MERGE (e:Entity {id: $entity_id})
                    SET e.name = $name, e.type = $type, e.description = $description, e.language = $language
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                """, entity_id=entity_id, name=entity.get('name', ''),
                    type=entity.get('type', 'UNKNOWN'),
                    description=entity.get('description', ''),
                    language=chunk.language, chunk_id=chunk.id)

                self.stats["entities_extracted"] += 1

            self.stats["relationships_created"] += len(chunk.entities)

    def process_pdf(self, pdf_path: str, document_id: str, document_title: str, language: str):
        """Process a single PDF file."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {document_title} ({language.upper()})")
        logger.info(f"Path: {pdf_path}")
        logger.info(f"{'='*60}")

        # Extract text
        pages = self.extract_text_from_pdf(pdf_path)

        all_chunks = []
        chunk_index = 0

        for page_data in pages:
            page_num = page_data["page_number"]
            page_text = page_data["text"]

            # Detect language if not specified
            detected_lang = self.detect_language(page_text)
            effective_lang = language or detected_lang

            # Chunk the page
            text_chunks = self.chunk_text(page_text, effective_lang)

            for chunk_text in text_chunks:
                chunk_id = f"{document_id}_p{page_num}_c{chunk_index}"

                chunk = ProcessedChunk(
                    id=chunk_id,
                    text=chunk_text,
                    language=effective_lang,
                    document_id=document_id,
                    document_title=document_title,
                    page_number=page_num,
                    chunk_index=chunk_index
                )

                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")

        # Process chunks with entity extraction
        for i, chunk in enumerate(all_chunks):
            logger.info(f"Processing chunk {i+1}/{len(all_chunks)} ({chunk.language.upper()})...")

            # Extract entities using TGI
            chunk.entities = self.extract_entities_with_tgi(chunk.text, chunk.language)
            logger.info(f"  Extracted {len(chunk.entities)} entities")

            # Create embedding
            chunk.embedding = self.create_embedding(chunk.text)

            # Store in Qdrant
            self.store_chunk_in_qdrant(chunk)

            # Store in Neo4j
            self.store_entities_in_neo4j(chunk)

            self.stats["chunks_created"] += 1
            if chunk.language == "ar":
                self.stats["ar_chunks"] += 1
            else:
                self.stats["en_chunks"] += 1

        self.stats["documents_processed"] += 1
        logger.info(f"Completed: {document_title}")

    def create_community_summaries(self):
        """Create community summaries for global search."""
        logger.info("\nCreating community summaries...")

        with self.neo4j.driver.session() as session:
            # Get all entity types as communities
            result = session.run("""
                MATCH (e:Entity)
                WITH e.type as type, collect(e.name) as entities, count(*) as count
                WHERE count > 2
                RETURN type, entities, count
                ORDER BY count DESC
            """)

            communities = list(result)
            logger.info(f"Found {len(communities)} entity communities")

            for record in communities:
                entity_type = record["type"]
                entities = record["entities"][:10]  # Top 10 entities
                count = record["count"]

                # Create community node
                summary = f"Community of {count} {entity_type} entities including: {', '.join(entities[:5])}"

                session.run("""
                    MERGE (c:Community {type: $type})
                    SET c.summary = $summary, c.entity_count = $count
                """, type=entity_type, summary=summary, count=count)

        logger.info("Community summaries created")

    def print_stats(self):
        """Print processing statistics."""
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Documents processed: {self.stats['documents_processed']}")
        print(f"Total chunks created: {self.stats['chunks_created']}")
        print(f"  - Arabic chunks: {self.stats['ar_chunks']}")
        print(f"  - English chunks: {self.stats['en_chunks']}")
        print(f"Entities extracted: {self.stats['entities_extracted']}")
        print(f"Relationships created: {self.stats['relationships_created']}")
        print("="*60)


def main():
    """Main processing function."""

    # Define documents to process
    documents = [
        {
            "path": "/app/data/sdaia_policies/ndmo_policies_en.pdf",
            "id": "ndmo_policies_en",
            "title": "NDMO Master Policies (English)",
            "language": "en"
        },
        {
            "path": "/app/data/sdaia_policies/ndmo_policies_ar.pdf",
            "id": "ndmo_policies_ar",
            "title": "NDMO Master Policies (Arabic)",
            "language": "ar"
        }
    ]

    # Check files exist
    for doc in documents:
        if not os.path.exists(doc["path"]):
            logger.warning(f"File not found: {doc['path']}")
            # Try alternate paths
            alt_paths = [
                f"/app/data/{os.path.basename(doc['path'])}",
                f"/data/{os.path.basename(doc['path'])}",
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    doc["path"] = alt
                    logger.info(f"Using alternate path: {alt}")
                    break

    # Initialize processor
    processor = BilingualPDFProcessor()

    # Process each document
    for doc in documents:
        if os.path.exists(doc["path"]):
            processor.process_pdf(
                pdf_path=doc["path"],
                document_id=doc["id"],
                document_title=doc["title"],
                language=doc["language"]
            )
        else:
            logger.error(f"Skipping - file not found: {doc['path']}")

    # Create community summaries
    processor.create_community_summaries()

    # Print final stats
    processor.print_stats()

    return processor.stats


if __name__ == "__main__":
    stats = main()
