"""
Database Management Service API
Handles graph database and vector database management
Provides UI integration for database operations
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from loguru import logger

from ..core.graph_builder import Neo4jClient
from ..core.vector_store import QdrantVectorStore
from ..config import settings

router = APIRouter()

# Initialize database clients
neo4j_client = Neo4jClient()
qdrant_client = QdrantVectorStore()


# Models
class DatabaseStats(BaseModel):
    name: str
    status: str
    total_records: int
    size_mb: Optional[float] = None
    metadata: Dict[str, Any]


class DatabaseHealth(BaseModel):
    database: str
    healthy: bool
    connected: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class GraphDatabaseInfo(BaseModel):
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    edge_types: Dict[str, int]
    document_count: int


class VectorDatabaseInfo(BaseModel):
    collection_name: str
    vectors_count: int
    points_count: int
    status: str


# Graph Database Endpoints
@router.get("/graph/health")
async def check_graph_health():
    """
    Check Neo4j graph database health and connectivity
    """
    logger.info("Checking graph database health")

    try:
        # Try to connect
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Get basic stats to verify connectivity
        stats = neo4j_client.get_graph_stats()

        return DatabaseHealth(
            database="neo4j",
            healthy=True,
            connected=True,
            message="Graph database is healthy and connected",
            details={
                "uri": neo4j_client.uri,
                "nodes": stats.get("total_nodes", 0),
                "edges": stats.get("total_edges", 0),
            },
        )

    except Exception as e:
        logger.error(f"Graph database health check failed: {e}")
        return DatabaseHealth(
            database="neo4j",
            healthy=False,
            connected=False,
            message=f"Graph database is not healthy: {str(e)}",
            details={"error": str(e)},
        )


@router.get("/graph/stats")
async def get_graph_statistics():
    """
    Get detailed statistics about the graph database
    """
    logger.info("Getting graph database statistics")

    try:
        # Ensure connection
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Get stats
        stats = neo4j_client.get_graph_stats()
        document_ids = neo4j_client.get_document_ids()

        return GraphDatabaseInfo(
            total_nodes=stats.get("total_nodes", 0),
            total_edges=stats.get("total_edges", 0),
            node_types=stats.get("node_types", {}),
            edge_types=stats.get("edge_types", {}),
            document_count=len(document_ids),
        )

    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/clear")
async def clear_graph_database():
    """
    Clear all data from the graph database
    WARNING: This is a destructive operation
    """
    logger.warning("Clearing graph database")

    try:
        # Ensure connection
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Clear graph
        result = neo4j_client.clear_graph()

        return {
            "status": "success",
            "message": "Graph database cleared successfully",
            "nodes_deleted": result["nodes_deleted"],
            "relationships_deleted": result["relationships_deleted"],
        }

    except Exception as e:
        logger.error(f"Error clearing graph database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph/rebuild")
async def rebuild_graph_database():
    """
    Rebuild the graph database from all documents
    This clears the graph and returns document IDs that need reprocessing
    """
    logger.warning("Rebuilding graph database")

    try:
        # Ensure connection
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Get document IDs before clearing
        document_ids = neo4j_client.get_document_ids()

        # Clear graph
        result = neo4j_client.clear_graph()

        return {
            "status": "cleared",
            "message": f"Graph cleared. {len(document_ids)} documents need reprocessing",
            "nodes_deleted": result["nodes_deleted"],
            "relationships_deleted": result["relationships_deleted"],
            "documents_to_reprocess": document_ids,
            "document_count": len(document_ids),
        }

    except Exception as e:
        logger.error(f"Error rebuilding graph database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/graph/document/{document_id}")
async def delete_document_from_graph(document_id: str):
    """
    Remove a specific document from the graph database
    """
    logger.info(f"Deleting document {document_id} from graph")

    try:
        # Ensure connection
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Delete document
        result = neo4j_client.delete_by_document(document_id)

        return {
            "status": "success",
            "message": f"Document {document_id} removed from graph",
            "nodes_deleted": result["nodes_deleted"],
            "nodes_updated": result["nodes_updated"],
            "relationships_deleted": result["relationships_deleted"],
        }

    except Exception as e:
        logger.error(f"Error deleting document from graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Vector Database Endpoints
@router.get("/vector/health")
async def check_vector_health():
    """
    Check Qdrant vector database health and connectivity
    """
    logger.info("Checking vector database health")

    try:
        # Get collection info to verify connectivity
        stats = qdrant_client.get_stats()

        return DatabaseHealth(
            database="qdrant",
            healthy=True,
            connected=True,
            message="Vector database is healthy and connected",
            details={
                "host": f"{qdrant_client.host}:{qdrant_client.port}",
                "collection": qdrant_client.collection_name,
                "vectors": stats.get("vectors_count", 0),
            },
        )

    except Exception as e:
        logger.error(f"Vector database health check failed: {e}")
        return DatabaseHealth(
            database="qdrant",
            healthy=False,
            connected=False,
            message=f"Vector database is not healthy: {str(e)}",
            details={"error": str(e)},
        )


@router.get("/vector/stats")
async def get_vector_statistics():
    """
    Get detailed statistics about the vector database for UI display
    """
    logger.info("Getting vector database statistics")

    try:
        # Get basic stats from Qdrant
        stats = qdrant_client.get_stats()
        points_count = stats.get("points_count", 0)

        # Get unique document count by scrolling through all points
        # This is expensive but necessary for accurate stats
        unique_docs = set()
        total_chars = 0
        chunk_count = 0

        if points_count > 0:
            try:
                # Scroll through all points to get unique documents and avg size
                offset = None
                while True:
                    results, offset = qdrant_client.client.scroll(
                        collection_name=qdrant_client.collection_name,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )

                    if not results:
                        break

                    for point in results:
                        if point.payload:
                            doc_id = point.payload.get("document_id")
                            if doc_id:
                                unique_docs.add(doc_id)
                            char_count = point.payload.get("char_count", 0)
                            total_chars += char_count
                            chunk_count += 1

                    if offset is None:
                        break
            except Exception as e:
                logger.warning(f"Error scrolling chunks for stats: {e}")

        avg_chunk_size = total_chars / chunk_count if chunk_count > 0 else 0

        return {
            "total_chunks": points_count,
            "total_documents": len(unique_docs),
            "avg_chunk_size": avg_chunk_size,
            "total_vectors": stats.get("vectors_count", 0),
        }

    except Exception as e:
        logger.error(f"Error getting vector statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vector/clear")
async def clear_vector_database():
    """
    Clear all data from the vector database collection
    WARNING: This is a destructive operation
    """
    logger.warning("Clearing vector database")

    try:
        # Get stats before clearing
        stats_before = qdrant_client.get_stats()
        vectors_before = stats_before.get("vectors_count", 0)

        # Delete and recreate collection
        qdrant_client.client.delete_collection(qdrant_client.collection_name)
        qdrant_client._ensure_collection()

        logger.info(f"Cleared vector database: {vectors_before} vectors deleted")

        return {
            "status": "success",
            "message": "Vector database cleared successfully",
            "vectors_deleted": vectors_before,
            "collection": qdrant_client.collection_name,
        }

    except Exception as e:
        logger.error(f"Error clearing vector database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vector/document/{document_id}")
async def delete_document_from_vector(document_id: str):
    """
    Remove a specific document from the vector database
    """
    logger.info(f"Deleting document {document_id} from vector database")

    try:
        # Delete document
        result = qdrant_client.delete_by_document(document_id)

        return {
            "status": "success",
            "message": f"Document {document_id} removed from vector database",
            "document_id": document_id,
        }

    except Exception as e:
        logger.error(f"Error deleting document from vector database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Combined Operations
@router.get("/health")
async def check_all_databases():
    """
    Check health of all databases
    """
    logger.info("Checking all database health")

    results = []

    # Check Neo4j
    try:
        if not neo4j_client._connected:
            neo4j_client.connect()
        stats = neo4j_client.get_graph_stats()
        results.append({
            "database": "neo4j",
            "status": "healthy",
            "connected": True,
            "nodes": stats.get("total_nodes", 0),
            "edges": stats.get("total_edges", 0),
        })
    except Exception as e:
        results.append({
            "database": "neo4j",
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        })

    # Check Qdrant
    try:
        stats = qdrant_client.get_stats()
        results.append({
            "database": "qdrant",
            "status": "healthy",
            "connected": True,
            "vectors": stats.get("vectors_count", 0),
            "points": stats.get("points_count", 0),
        })
    except Exception as e:
        results.append({
            "database": "qdrant",
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        })

    # Overall status
    all_healthy = all(r.get("status") == "healthy" for r in results)

    from datetime import datetime
    return {
        "overall_status": "healthy" if all_healthy else "degraded",
        "databases": results,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/clear-all")
async def clear_all_databases():
    """
    Clear all data from both graph and vector databases
    WARNING: This is a destructive operation that cannot be undone
    """
    logger.warning("Clearing all databases")

    results = {}
    errors = []

    # Clear graph database
    try:
        if not neo4j_client._connected:
            neo4j_client.connect()
        graph_result = neo4j_client.clear_graph()
        results["graph"] = {
            "status": "success",
            "nodes_deleted": graph_result["nodes_deleted"],
            "relationships_deleted": graph_result["relationships_deleted"],
        }
    except Exception as e:
        logger.error(f"Error clearing graph database: {e}")
        errors.append({"database": "graph", "error": str(e)})

    # Clear vector database
    try:
        stats_before = qdrant_client.get_stats()
        qdrant_client.client.delete_collection(qdrant_client.collection_name)
        qdrant_client._ensure_collection()
        results["vector"] = {
            "status": "success",
            "vectors_deleted": stats_before.get("vectors_count", 0),
        }
    except Exception as e:
        logger.error(f"Error clearing vector database: {e}")
        errors.append({"database": "vector", "error": str(e)})

    return {
        "status": "completed" if not errors else "partial",
        "message": "All databases cleared" if not errors else "Some databases failed to clear",
        "results": results,
        "errors": errors,
    }


@router.get("/stats")
async def get_all_database_stats():
    """
    Get statistics from all databases in one call
    """
    logger.info("Getting all database statistics")

    stats = {}

    # Graph stats
    try:
        if not neo4j_client._connected:
            neo4j_client.connect()
        graph_stats = neo4j_client.get_graph_stats()
        document_ids = neo4j_client.get_document_ids()
        stats["graph"] = {
            "status": "available",
            "total_nodes": graph_stats.get("total_nodes", 0),
            "total_edges": graph_stats.get("total_edges", 0),
            "node_types": graph_stats.get("node_types", {}),
            "edge_types": graph_stats.get("edge_types", {}),
            "document_count": len(document_ids),
        }
    except Exception as e:
        stats["graph"] = {"status": "error", "error": str(e)}

    # Vector stats
    try:
        vector_stats = qdrant_client.get_stats()
        stats["vector"] = {
            "status": "available",
            "collection": vector_stats.get("collection_name", "unknown"),
            "vectors_count": vector_stats.get("vectors_count", 0),
            "points_count": vector_stats.get("points_count", 0),
        }
    except Exception as e:
        stats["vector"] = {"status": "error", "error": str(e)}

    return stats


@router.get("/vector/chunks")
async def get_vector_chunks(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    document_id: Optional[str] = Query(default=None),
):
    """
    Get vector store chunks with pagination
    """
    logger.info(f"Getting vector chunks: limit={limit}, offset={offset}, document_id={document_id}")

    try:
        # Use scroll method to paginate through points
        scroll_filter = None
        if document_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        # Scroll through points
        points, next_offset = qdrant_client.client.scroll(
            collection_name=qdrant_client.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        # Format chunks
        chunks = []
        for point in points:
            chunk = {
                "id": str(point.id),
                "document_id": point.payload.get("document_id", ""),
                "chunk_index": point.payload.get("chunk_index", 0),
                "text": point.payload.get("text", ""),
                "char_count": point.payload.get("char_count", 0),
                "word_count": point.payload.get("word_count", 0),
                "sentence_count": point.payload.get("sentence_count", 0),
                "metadata": point.payload.get("metadata", {}),
            }
            chunks.append(chunk)

        # Get total count
        stats = qdrant_client.get_stats()
        total = stats.get("points_count", 0)

        return {
            "chunks": chunks,
            "total": total,
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset,
            "has_more": next_offset is not None,
        }

    except Exception as e:
        logger.error(f"Error getting vector chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings")
async def get_system_settings():
    """
    Get current system configuration from all YAML files
    """
    logger.info("Getting system settings")

    try:
        import yaml
        from pathlib import Path

        # Get config directory
        config_dir = Path(__file__).parent.parent / "config"

        settings_data = {}

        # Read models.yaml
        models_file = config_dir / "models.yaml"
        if models_file.exists():
            with open(models_file, "r", encoding="utf-8") as f:
                settings_data["models"] = yaml.safe_load(f)

        # Read processing.yaml
        processing_file = config_dir / "processing.yaml"
        if processing_file.exists():
            with open(processing_file, "r", encoding="utf-8") as f:
                settings_data["processing"] = yaml.safe_load(f)

        # Read databases.yaml
        databases_file = config_dir / "databases.yaml"
        if databases_file.exists():
            with open(databases_file, "r", encoding="utf-8") as f:
                settings_data["databases"] = yaml.safe_load(f)

        # Read chunking_strategies.yaml
        chunking_file = config_dir / "chunking_strategies.yaml"
        if chunking_file.exists():
            with open(chunking_file, "r", encoding="utf-8") as f:
                settings_data["chunking_strategies"] = yaml.safe_load(f)

        # Read prompts from prompts directory (file-based prompts)
        prompts_dir = config_dir / "prompts"
        prompts = {}

        # Load file-based prompts
        if prompts_dir.exists() and prompts_dir.is_dir():
            for prompt_file in prompts_dir.glob("*.txt"):
                try:
                    with open(prompt_file, "r", encoding="utf-8") as f:
                        prompts[f"file:{prompt_file.name}"] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to load prompt {prompt_file.name}: {e}")

        # Load built-in prompts from PromptManager
        try:
            from src.core.generation.prompt_manager import PromptManager
            prompt_manager = PromptManager()

            # Get all registered templates
            for prompt_type, versions in prompt_manager._templates.items():
                for version, template in versions.items():
                    prompt_key = f"builtin:{template.name}"
                    prompt_content = f"""--- {template.name} (v{template.version}) ---
Type: {template.type.value}
Description: {template.description}

=== System Message ===
{template.system_message}

=== Template ===
{template.template}

=== Settings ===
Chain-of-Thought: {'Yes' if template.use_cot else 'No'}
Languages: {', '.join(template.languages)}"""
                    prompts[prompt_key] = prompt_content
        except Exception as e:
            logger.warning(f"Failed to load built-in prompts: {e}")

        settings_data["prompts"] = prompts

        return {
            "settings": settings_data,
            "config_dir": str(config_dir),
        }

    except Exception as e:
        logger.error(f"Error getting system settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
