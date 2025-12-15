#!/usr/bin/env python3
"""
Comprehensive Entity Extraction Benchmark
Compares entity extraction quality between different LLMs
"""

import sys
sys.path.insert(0, "/app")

import json
import time
import fitz  # PyMuPDF
from datetime import datetime
from collections import Counter
from loguru import logger
import requests

from src.core.graph_builder.neo4j_client import Neo4jClient
from src.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid


def get_current_model():
    """Get the currently loaded TGI model."""
    try:
        resp = requests.get("http://tgi:80/info", timeout=5)
        return resp.json().get("model_id", "unknown")
    except:
        return "unknown"


def extract_entities_with_tgi(text: str, chunk_id: str) -> dict:
    """Extract entities using TGI LLM."""
    prompt = f"""Extract all named entities and their relationships from this text about data governance policies.

For each entity, provide:
- name: The entity name
- type: One of [Organization, Person, Policy, Technology, Process, Event, Location, Concept, Program]
- description: Brief description

For each relationship, provide:
- source: Source entity name
- target: Target entity name
- type: Relationship type (e.g., MANAGES, IMPLEMENTS, REGULATES, BELONGS_TO)

Text:
{text[:2000]}

Respond in JSON format:
{{"entities": [...], "relationships": [...]}}"""

    try:
        resp = requests.post(
            "http://tgi:80/generate",
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.1,
                    "do_sample": False
                }
            },
            timeout=60
        )

        if resp.status_code == 200:
            result = resp.json()
            generated = result.get("generated_text", "")

            # Try to parse JSON from response
            try:
                # Find JSON in response
                start = generated.find("{")
                end = generated.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = generated[start:end]
                    data = json.loads(json_str)
                    return {
                        "entities": data.get("entities", []),
                        "relationships": data.get("relationships", []),
                        "raw_response": generated[:500]
                    }
            except json.JSONDecodeError:
                pass

            return {"entities": [], "relationships": [], "raw_response": generated[:500], "parse_error": True}
        else:
            return {"entities": [], "relationships": [], "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"entities": [], "relationships": [], "error": str(e)}


def clear_databases():
    """Clear Neo4j and Qdrant."""
    logger.info("Clearing databases...")

    # Clear Neo4j
    neo4j = Neo4jClient()
    neo4j.connect()
    with neo4j.driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    # neo4j client stays connected for later use

    # Clear Qdrant
    try:
        qdrant = QdrantClient(url=f"http://{settings.qdrant_host}:{settings.qdrant_port}")
        try:
            qdrant.delete_collection("mirage_chunks")
        except:
            pass
        qdrant.create_collection(
            collection_name="mirage_chunks",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    except Exception as e:
        logger.warning(f"Qdrant clear error: {e}")

    logger.info("Databases cleared")


def run_extraction_benchmark(pdf_path: str = "/app/data/ndmo_policies_en.pdf", max_chunks: int = 20):
    """Run entity extraction benchmark on PDF."""

    model_name = get_current_model()
    logger.info(f"Running entity extraction benchmark with model: {model_name}")

    # Clear databases first
    clear_databases()

    # Extract text from PDF
    logger.info(f"Extracting text from {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    logger.info(f"Extracted {len(full_text)} characters")

    # Chunk the text using simple splitting
    logger.info("Chunking text...")
    chunk_size = 1000
    overlap = 100
    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunk_text = full_text[i:i + chunk_size]
        if len(chunk_text) > 100:  # Skip very small chunks
            chunks.append({"text": chunk_text, "index": len(chunks)})
    logger.info(f"Created {len(chunks)} chunks")

    # Limit chunks for benchmark
    chunks = chunks[:max_chunks]
    logger.info(f"Processing {len(chunks)} chunks for benchmark")

    # Initialize results
    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "pdf_path": pdf_path,
        "total_chunks": len(chunks),
        "entities": [],
        "relationships": [],
        "entity_types": Counter(),
        "relationship_types": Counter(),
        "extraction_times": [],
        "errors": [],
        "sample_extractions": []
    }

    # Process each chunk
    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        chunk_id = f"chunk_{i}"

        if not chunk_text or len(chunk_text) < 50:
            continue

        logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk_text)} chars)")

        start_time = time.time()
        extraction = extract_entities_with_tgi(chunk_text, chunk_id)
        elapsed = time.time() - start_time

        results["extraction_times"].append(elapsed)

        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        # Store sample extractions (first 5)
        if i < 5:
            results["sample_extractions"].append({
                "chunk_id": chunk_id,
                "chunk_text": chunk_text[:300] + "...",
                "entities": entities,
                "relationships": relationships,
                "raw_response": extraction.get("raw_response", ""),
                "latency": elapsed
            })

        if extraction.get("error"):
            results["errors"].append({
                "chunk_id": chunk_id,
                "error": extraction["error"]
            })
            continue

        # Count entity types
        for entity in entities:
            if isinstance(entity, dict):
                entity_type = entity.get("type", "Unknown")
                results["entity_types"][entity_type] += 1
                all_entities.append(entity)

        # Count relationship types
        for rel in relationships:
            if isinstance(rel, dict):
                rel_type = rel.get("type", "Unknown")
                results["relationship_types"][rel_type] += 1
                all_relationships.append(rel)

        logger.info(f"  Found {len(entities)} entities, {len(relationships)} relationships ({elapsed:.2f}s)")

    # Calculate statistics
    results["total_entities"] = len(all_entities)
    results["total_relationships"] = len(all_relationships)
    results["unique_entity_names"] = len(set(
        e.get("name", "") for e in all_entities if isinstance(e, dict)
    ))
    results["avg_extraction_time"] = sum(results["extraction_times"]) / len(results["extraction_times"]) if results["extraction_times"] else 0
    results["total_extraction_time"] = sum(results["extraction_times"])
    results["error_rate"] = len(results["errors"]) / len(chunks) if chunks else 0

    # Convert Counters to dicts for JSON
    results["entity_types"] = dict(results["entity_types"])
    results["relationship_types"] = dict(results["relationship_types"])

    # Store all entities for quality analysis
    results["all_entities"] = all_entities[:100]  # Store first 100 for analysis
    results["all_relationships"] = all_relationships[:100]

    return results


def store_in_graph(results: dict):
    """Store extracted entities in Neo4j for later benchmarking."""
    logger.info("Storing entities in Neo4j...")

    neo4j = Neo4jClient()
    neo4j.connect()

    # Create entity nodes
    entities_created = 0
    for entity in results.get("all_entities", []):
        if isinstance(entity, dict) and entity.get("name"):
            try:
                with neo4j.driver.session() as s:
                    s.run("""
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type, e.description = $description
                    """, {
                        "name": entity.get("name", ""),
                        "type": entity.get("type", "Unknown"),
                        "description": entity.get("description", "")
                    })
                    entities_created += 1
            except Exception as e:
                logger.warning(f"Error creating entity: {e}")

    # Create relationships
    rels_created = 0
    for rel in results.get("all_relationships", []):
        if isinstance(rel, dict) and rel.get("source") and rel.get("target"):
            try:
                with neo4j.driver.session() as s:
                    s.run("""
                        MATCH (a:Entity {name: $source})
                        MATCH (b:Entity {name: $target})
                        MERGE (a)-[r:RELATED_TO {type: $type}]->(b)
                    """, {
                        "source": rel.get("source", ""),
                        "target": rel.get("target", ""),
                        "type": rel.get("type", "RELATED_TO")
                    })
                    rels_created += 1
            except Exception as e:
                logger.warning(f"Error creating relationship: {e}")

    logger.info(f"Created {entities_created} entities, {rels_created} relationships in Neo4j")

    return entities_created, rels_created


def main():
    """Run the benchmark and save results."""
    logger.info("=" * 60)
    logger.info("ENTITY EXTRACTION BENCHMARK")
    logger.info("=" * 60)

    # Run benchmark
    results = run_extraction_benchmark(max_chunks=30)

    # Store in graph
    entities_stored, rels_stored = store_in_graph(results)
    results["entities_stored"] = entities_stored
    results["relationships_stored"] = rels_stored

    # Save results
    model_safe_name = results["model"].replace("/", "_").replace("-", "_")
    output_file = f"/app/benchmark_results/entity_extraction_{model_safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {results['model']}")
    print(f"Chunks processed: {results['total_chunks']}")
    print(f"Total entities: {results['total_entities']}")
    print(f"Unique entities: {results['unique_entity_names']}")
    print(f"Total relationships: {results['total_relationships']}")
    print(f"Avg extraction time: {results['avg_extraction_time']:.2f}s")
    print(f"Error rate: {results['error_rate']:.1%}")
    print(f"\nEntity types: {results['entity_types']}")
    print(f"\nRelationship types: {results['relationship_types']}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
