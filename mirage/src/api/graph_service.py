"""
Graph Service API
Provides access to the knowledge graph
Supports exploration, search, and visualization
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from loguru import logger

from ..core.graph_builder import Neo4jClient

router = APIRouter()

# Initialize Neo4j client
neo4j_client = Neo4jClient()


# Models
class Node(BaseModel):
    id: str
    label: str
    type: str  # entity, concept, document
    properties: Dict[str, Any]


class Edge(BaseModel):
    from_node: str
    to_node: str
    relationship: str
    weight: float


class GraphData(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    total_nodes: int
    total_edges: int


class EntitySearchResult(BaseModel):
    id: str
    name: str
    type: str
    confidence: float
    source_documents: List[str]


class GraphStats(BaseModel):
    total_nodes: int
    total_edges: int
    node_types: Dict[str, int]
    edge_types: Dict[str, int]
    largest_component_size: int
    average_degree: float


# Endpoints
@router.get("/explore", response_model=GraphData)
async def explore_graph(
    node_id: Optional[str] = None,
    depth: int = Query(2, ge=1, le=5),
    max_nodes: int = Query(100, ge=10, le=1000),
):
    """
    Explore the knowledge graph
    If node_id is provided, returns subgraph around that node
    Otherwise, returns most connected nodes
    """
    logger.info(f"Graph exploration request: node_id={node_id}, depth={depth}")

    # TODO: Implement Neo4j query
    # - If node_id: BFS from node with depth limit
    # - If None: Get high-centrality nodes

    return GraphData(
        nodes=[],
        edges=[],
        total_nodes=0,
        total_edges=0,
    )


@router.get("/search", response_model=List[EntitySearchResult])
async def search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
):
    """
    Search for entities in the graph
    Supports fuzzy matching and type filtering
    """
    logger.info(f"Entity search: query='{query}', type={entity_type}")

    try:
        entities = neo4j_client.search_entities(query, entity_type, limit)

        results = []
        for entity in entities:
            results.append(EntitySearchResult(
                id=entity.get("name", ""),
                name=entity.get("name", ""),
                type=entity.get("type", ""),
                confidence=entity.get("confidence", 0.5),
                source_documents=entity.get("source_documents", []),
            ))

        return results
    except Exception as e:
        logger.error(f"Error searching entities: {e}")
        return []


@router.get("/stats", response_model=GraphStats)
async def get_graph_statistics():
    """
    Get overall graph statistics
    Useful for monitoring and visualization
    """
    logger.info("Fetching graph statistics")

    try:
        stats = neo4j_client.get_graph_stats()

        return GraphStats(
            total_nodes=stats.get("total_nodes", 0),
            total_edges=stats.get("total_edges", 0),
            node_types=stats.get("node_types", {}),
            edge_types=stats.get("edge_types", {}),
            largest_component_size=0,  # TODO: Calculate
            average_degree=0.0,  # TODO: Calculate
        )
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        return GraphStats(
            total_nodes=0,
            total_edges=0,
            node_types={},
            edge_types={},
            largest_component_size=0,
            average_degree=0.0,
        )


@router.get("/neighbors/{node_id}")
async def get_neighbors(
    node_id: str,
    relationship_type: Optional[str] = None,
):
    """Get direct neighbors of a node"""
    logger.info(f"Fetching neighbors for node: {node_id}")

    # TODO: Implement neighbor query

    return {
        "node_id": node_id,
        "neighbors": [],
    }


@router.get("/path/{start_node_id}/{end_node_id}")
async def find_path(
    start_node_id: str,
    end_node_id: str,
    max_length: int = Query(5, ge=1, le=10),
):
    """
    Find shortest path between two nodes
    Useful for understanding relationships
    """
    logger.info(f"Finding path: {start_node_id} -> {end_node_id}")

    # TODO: Implement shortest path algorithm

    return {
        "start": start_node_id,
        "end": end_node_id,
        "path": [],
        "length": 0,
    }


@router.get("/visualize/document/{document_id}")
async def visualize_document_graph(document_id: str):
    """
    Get graph data for visualizing a specific document's knowledge graph
    Returns nodes and edges in a format suitable for UI visualization
    """
    logger.info(f"Getting visualization data for document: {document_id}")

    try:
        # Ensure Neo4j is connected
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Query all entities for this document
        query = """
        MATCH (n:Entity)
        WHERE $document_id IN n.source_documents
        RETURN n
        LIMIT 100
        """

        nodes = []
        with neo4j_client.driver.session() as session:
            result = session.run(query, {"document_id": document_id})
            for record in result:
                node = dict(record["n"])
                nodes.append({
                    "id": node.get("name", ""),
                    "label": node.get("name", ""),
                    "type": node.get("type", "Entity"),
                    "confidence": node.get("confidence", 0.5),
                })

        # Query relationships
        rel_query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        WHERE $document_id IN source.source_documents
        AND $document_id IN target.source_documents
        RETURN source.name as source, target.name as target, type(r) as relationship, r
        LIMIT 100
        """

        edges = []
        with neo4j_client.driver.session() as session:
            result = session.run(rel_query, {"document_id": document_id})
            for record in result:
                rel_data = dict(record["r"]) if record["r"] else {}
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "relationship": record["relationship"],
                    "confidence": rel_data.get("confidence", 0.5),
                })

        return {
            "document_id": document_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        return {
            "document_id": document_id,
            "nodes": [],
            "edges": [],
            "error": str(e),
        }


@router.get("/visualize/full")
async def visualize_full_graph(limit: int = Query(500, ge=10, le=2000)):
    """
    Get full graph data for visualization
    Returns up to 'limit' most connected nodes and their relationships
    """
    logger.info(f"Getting full graph visualization data (limit={limit})")

    try:
        # Ensure Neo4j is connected
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Get most connected nodes
        query = f"""
        MATCH (n:Entity)
        WITH n, COUNT {{ (n)--() }} as degree
        ORDER BY degree DESC
        LIMIT {limit}
        RETURN n, degree
        """

        nodes = []
        node_names = set()
        with neo4j_client.driver.session() as session:
            result = session.run(query)
            for record in result:
                node = dict(record["n"])
                node_name = node.get("name", "")
                node_names.add(node_name)
                nodes.append({
                    "id": node_name,
                    "label": node_name,
                    "type": node.get("type", "Entity"),
                    "confidence": node.get("confidence", 0.5),
                    "degree": record["degree"],
                })

        # Get relationships between these nodes
        rel_query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        WHERE source.name IN $node_names AND target.name IN $node_names
        RETURN source.name as source, target.name as target, type(r) as relationship, r
        LIMIT 200
        """

        edges = []
        with neo4j_client.driver.session() as session:
            result = session.run(rel_query, {"node_names": list(node_names)})
            for record in result:
                rel_data = dict(record["r"]) if record["r"] else {}
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "relationship": record["relationship"],
                    "confidence": rel_data.get("confidence", 0.5),
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    except Exception as e:
        logger.error(f"Error getting full graph visualization: {e}")
        return {
            "nodes": [],
            "edges": [],
            "error": str(e),
        }


@router.post("/clear")
async def clear_graph():
    """
    Clear the entire knowledge graph
    WARNING: This is a destructive operation that cannot be undone
    """
    logger.warning("Graph clear triggered")

    try:
        # Ensure Neo4j is connected
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Clear the graph
        result = neo4j_client.clear_graph()

        return {
            "status": "success",
            "message": f"Graph cleared: {result['nodes_deleted']} nodes and {result['relationships_deleted']} relationships deleted",
            "nodes_deleted": result["nodes_deleted"],
            "relationships_deleted": result["relationships_deleted"],
        }

    except Exception as e:
        logger.error(f"Error clearing graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/document/{document_id}")
async def delete_document_from_graph(document_id: str):
    """
    Delete all entities and relationships for a specific document from the graph
    """
    logger.info(f"Deleting document {document_id} from graph")

    try:
        # Ensure Neo4j is connected
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Delete document from graph
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


@router.get("/documents")
async def list_graph_documents():
    """
    Get list of all document IDs that have data in the graph
    """
    logger.info("Fetching document IDs from graph")

    try:
        # Ensure Neo4j is connected
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Get document IDs
        document_ids = neo4j_client.get_document_ids()

        return {
            "total": len(document_ids),
            "document_ids": document_ids,
        }

    except Exception as e:
        logger.error(f"Error fetching document IDs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebuild")
async def rebuild_graph():
    """
    Rebuild the entire knowledge graph from all stored documents
    WARNING: This is a heavy operation
    Note: This endpoint clears the graph and triggers reprocessing of all documents
    The actual document reprocessing should be triggered from the document service
    """
    logger.warning("Graph rebuild triggered")

    try:
        # Ensure Neo4j is connected
        if not neo4j_client._connected:
            neo4j_client.connect()

        # Get current document count before clearing
        document_ids = neo4j_client.get_document_ids()
        doc_count = len(document_ids)

        # Clear the graph
        clear_result = neo4j_client.clear_graph()

        return {
            "status": "cleared",
            "message": f"Graph cleared. {doc_count} documents need to be reprocessed. Please trigger document reprocessing from the document service.",
            "nodes_deleted": clear_result["nodes_deleted"],
            "relationships_deleted": clear_result["relationships_deleted"],
            "documents_to_reprocess": document_ids,
        }

    except Exception as e:
        logger.error(f"Error rebuilding graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))
