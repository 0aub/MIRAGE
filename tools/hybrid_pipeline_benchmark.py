#!/usr/bin/env python3
"""
Hybrid Pipeline Benchmark
Phase 1: Build knowledge graph with current model (Qwen2.5-7B)
Phase 2: Run after switching to ALLaM for inference comparison
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
from src.core.embeddings.jina_embedder import JinaEmbedder
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


def clear_databases():
    """Clear Neo4j and Qdrant."""
    logger.info("Clearing databases...")

    # Clear Neo4j
    neo4j = Neo4jClient()
    neo4j.connect()
    with neo4j.driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")

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


def extract_entities_with_tgi(text: str) -> dict:
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
                start = generated.find("{")
                end = generated.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = generated[start:end]
                    data = json.loads(json_str)
                    return {
                        "entities": data.get("entities", []),
                        "relationships": data.get("relationships", [])
                    }
            except json.JSONDecodeError:
                pass

            return {"entities": [], "relationships": [], "parse_error": True}
        else:
            return {"entities": [], "relationships": [], "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"entities": [], "relationships": [], "error": str(e)}


def build_knowledge_graph(pdf_path: str = "/app/data/ndmo_policies_en.pdf", max_chunks: int = 50):
    """Build knowledge graph using current model for entity extraction."""

    model_name = get_current_model()
    logger.info(f"=" * 60)
    logger.info(f"BUILDING KNOWLEDGE GRAPH WITH: {model_name}")
    logger.info(f"=" * 60)

    # Clear databases
    clear_databases()

    # Extract text from PDF
    logger.info(f"Extracting text from {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    logger.info(f"Extracted {len(full_text)} characters")

    # Chunk the text
    chunk_size = 1000
    overlap = 100
    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        chunk_text = full_text[i:i + chunk_size]
        if len(chunk_text) > 100:
            chunks.append({"text": chunk_text, "index": len(chunks)})

    chunks = chunks[:max_chunks]
    logger.info(f"Processing {len(chunks)} chunks")

    # Results tracking
    results = {
        "extraction_model": model_name,
        "timestamp": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "entities": [],
        "relationships": [],
        "entity_types": Counter(),
        "relationship_types": Counter(),
        "extraction_times": []
    }

    # Neo4j client
    neo4j = Neo4jClient()
    neo4j.connect()

    # Embedding model
    embedder = JinaEmbedder()

    # Qdrant client
    qdrant = QdrantClient(url=f"http://{settings.qdrant_host}:{settings.qdrant_port}")

    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        chunk_id = f"chunk_{i}"

        logger.info(f"Processing chunk {i+1}/{len(chunks)}")

        # Extract entities
        start_time = time.time()
        extraction = extract_entities_with_tgi(chunk_text)
        elapsed = time.time() - start_time
        results["extraction_times"].append(elapsed)

        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        # Store chunk in Neo4j
        try:
            with neo4j.driver.session() as s:
                s.run("""
                    CREATE (c:Chunk {
                        id: $id,
                        text: $text,
                        document_id: 'ndmo_policies'
                    })
                """, {"id": chunk_id, "text": chunk_text[:2000]})
        except Exception as e:
            logger.warning(f"Error creating chunk: {e}")

        # Store chunk in Qdrant
        try:
            emb = embedder.embed(chunk_text)
            embedding = emb[0].tolist() if len(emb.shape) > 1 else emb.tolist()
            qdrant.upsert(
                collection_name="mirage_chunks",
                points=[PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)),
                    vector=embedding,
                    payload={"chunk_id": chunk_id, "text": chunk_text[:2000]}
                )]
            )
        except Exception as e:
            logger.warning(f"Error storing embedding: {e}")

        # Store entities in Neo4j
        for entity in entities:
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
                    results["entity_types"][entity.get("type", "Unknown")] += 1
                    all_entities.append(entity)
                except Exception as e:
                    logger.warning(f"Error creating entity: {e}")

        # Store relationships in Neo4j
        for rel in relationships:
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
                    results["relationship_types"][rel.get("type", "Unknown")] += 1
                    all_relationships.append(rel)
                except Exception as e:
                    pass

        logger.info(f"  Found {len(entities)} entities, {len(relationships)} rels ({elapsed:.2f}s)")

    # Final statistics
    results["total_entities"] = len(all_entities)
    results["unique_entities"] = len(set(e.get("name", "") for e in all_entities if isinstance(e, dict)))
    results["total_relationships"] = len(all_relationships)
    results["avg_extraction_time"] = sum(results["extraction_times"]) / len(results["extraction_times"]) if results["extraction_times"] else 0
    results["entity_types"] = dict(results["entity_types"])
    results["relationship_types"] = dict(results["relationship_types"])

    # Get final counts from Neo4j
    with neo4j.driver.session() as s:
        entity_count = s.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
        rel_count = s.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count").single()["count"]
        chunk_count = s.run("MATCH (c:Chunk) RETURN count(c) as count").single()["count"]

    results["neo4j_entities"] = entity_count
    results["neo4j_relationships"] = rel_count
    results["neo4j_chunks"] = chunk_count

    # Save results
    output_file = f"/app/benchmark_results/kg_build_{model_name.replace('/', '_').replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"=" * 60)
    logger.info(f"KNOWLEDGE GRAPH BUILD COMPLETE")
    logger.info(f"Model: {model_name}")
    logger.info(f"Entities in Neo4j: {entity_count}")
    logger.info(f"Relationships in Neo4j: {rel_count}")
    logger.info(f"Chunks: {chunk_count}")
    logger.info(f"Avg extraction time: {results['avg_extraction_time']:.2f}s")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"=" * 60)

    return results


def run_inference_benchmark():
    """Run inference benchmark using current model against existing knowledge graph."""

    model_name = get_current_model()
    logger.info(f"=" * 60)
    logger.info(f"RUNNING INFERENCE BENCHMARK WITH: {model_name}")
    logger.info(f"=" * 60)

    # Test queries - mix of English and Arabic
    test_queries = [
        {"query": "What is data classification?", "lang": "en"},
        {"query": "What are the roles of a Data Owner?", "lang": "en"},
        {"query": "Explain the National Data Governance Framework", "lang": "en"},
        {"query": "What is the purpose of data quality management?", "lang": "en"},
        {"query": "ما هو تصنيف البيانات؟", "lang": "ar"},
        {"query": "ما هي مسؤوليات مالك البيانات؟", "lang": "ar"},
    ]

    modes = ["naive", "local", "global", "hybrid", "mix"]

    results = {
        "inference_model": model_name,
        "timestamp": datetime.now().isoformat(),
        "queries": [],
        "mode_summary": {}
    }

    for mode in modes:
        mode_times = []
        mode_results = []

        for tq in test_queries:
            query = tq["query"]
            logger.info(f"Mode: {mode}, Query: {query[:50]}...")

            try:
                start_time = time.time()
                resp = requests.post(
                    "http://mirage:8000/chat/ask",
                    json={
                        "message": query,
                        "retrieval_mode": mode,
                        "top_k": 5
                    },
                    timeout=120
                )
                elapsed = time.time() - start_time

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")

                    mode_times.append(elapsed)
                    mode_results.append({
                        "query": query,
                        "mode": mode,
                        "latency": elapsed,
                        "answer": answer[:500],
                        "success": True
                    })
                    logger.info(f"  {elapsed:.2f}s - {len(answer)} chars")
                else:
                    mode_results.append({
                        "query": query,
                        "mode": mode,
                        "error": f"HTTP {resp.status_code}",
                        "success": False
                    })
            except Exception as e:
                mode_results.append({
                    "query": query,
                    "mode": mode,
                    "error": str(e),
                    "success": False
                })

        results["queries"].extend(mode_results)
        results["mode_summary"][mode] = {
            "avg_latency": sum(mode_times) / len(mode_times) if mode_times else 0,
            "min_latency": min(mode_times) if mode_times else 0,
            "max_latency": max(mode_times) if mode_times else 0,
            "success_rate": sum(1 for r in mode_results if r.get("success")) / len(mode_results)
        }

    # Overall stats
    all_times = [r["latency"] for r in results["queries"] if r.get("success")]
    results["overall"] = {
        "avg_latency": sum(all_times) / len(all_times) if all_times else 0,
        "total_queries": len(test_queries) * len(modes),
        "successful_queries": len(all_times)
    }

    # Save results
    output_file = f"/app/benchmark_results/inference_{model_name.replace('/', '_').replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"=" * 60)
    logger.info(f"INFERENCE BENCHMARK COMPLETE")
    logger.info(f"Model: {model_name}")
    logger.info(f"Overall avg latency: {results['overall']['avg_latency']:.2f}s")
    for mode, summary in results["mode_summary"].items():
        logger.info(f"  {mode}: {summary['avg_latency']:.2f}s avg")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"=" * 60)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["build", "inference", "both"], default="both")
    parser.add_argument("--chunks", type=int, default=50)
    args = parser.parse_args()

    if args.phase in ["build", "both"]:
        build_knowledge_graph(max_chunks=args.chunks)

    if args.phase in ["inference", "both"]:
        run_inference_benchmark()
