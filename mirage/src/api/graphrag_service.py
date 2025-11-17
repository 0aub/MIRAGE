"""
GraphRAG API Service
Exposes GraphRAG search endpoints (Global, Local, Hybrid) for the UI

Endpoints:
- POST /graphrag/search - Auto-routed hybrid search
- POST /graphrag/global - Global search (community summaries)
- POST /graphrag/local - Local search (entity traversal)
- GET /graphrag/stats - Get search statistics
- GET /graphrag/communities - Get community hierarchy
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from loguru import logger

from ..core.graph_builder import (
    Neo4jClient,
    GraphTraversal,
    HybridSearchEngine,
    GlobalSearchEngine,
    LocalSearchEngine
)
from ..config.settings import settings

router = APIRouter(prefix="/graphrag", tags=["graphrag"])

# Initialize GraphRAG components (lazy loading)
_neo4j_client: Optional[Neo4jClient] = None
_graph_traversal: Optional[GraphTraversal] = None
_hybrid_engine: Optional[HybridSearchEngine] = None
_global_engine: Optional[GlobalSearchEngine] = None
_local_engine: Optional[LocalSearchEngine] = None


def get_neo4j_client() -> Neo4jClient:
    """Get or create Neo4j client"""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
    return _neo4j_client


def get_hybrid_engine() -> HybridSearchEngine:
    """Get or create HybridSearchEngine"""
    global _hybrid_engine, _graph_traversal
    if _hybrid_engine is None:
        neo4j = get_neo4j_client()
        _graph_traversal = GraphTraversal(neo4j)
        _hybrid_engine = HybridSearchEngine(
            neo4j_client=neo4j,
            graph_traversal=_graph_traversal,
            llm_endpoint=settings.tgi_endpoint,
            auto_route=True
        )
    return _hybrid_engine


def get_global_engine() -> GlobalSearchEngine:
    """Get or create GlobalSearchEngine"""
    global _global_engine
    if _global_engine is None:
        neo4j = get_neo4j_client()
        _global_engine = GlobalSearchEngine(
            neo4j_client=neo4j,
            llm_endpoint=settings.tgi_endpoint
        )
    return _global_engine


def get_local_engine() -> LocalSearchEngine:
    """Get or create LocalSearchEngine"""
    global _local_engine, _graph_traversal
    if _local_engine is None:
        neo4j = get_neo4j_client()
        if _graph_traversal is None:
            _graph_traversal = GraphTraversal(neo4j)
        _local_engine = LocalSearchEngine(
            neo4j_client=neo4j,
            graph_traversal=_graph_traversal,
            llm_endpoint=settings.tgi_endpoint
        )
    return _local_engine


# Request/Response Models

class SearchRequest(BaseModel):
    """Search request"""
    query: str
    mode: Optional[str] = None  # "global", "local", "hybrid", or None (auto)


class SearchResponse(BaseModel):
    """Search response"""
    query: str
    answer: str
    search_mode: str
    confidence: float
    metadata: Dict[str, Any]


class CommunityResponse(BaseModel):
    """Community data response"""
    communities: List[Dict[str, Any]]
    total: int
    levels: int


class StatsResponse(BaseModel):
    """Statistics response"""
    global_stats: Dict[str, Any]
    local_stats: Dict[str, Any]


# API Endpoints

@router.post("/search", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """
    Execute hybrid search with automatic routing

    Auto-routes to:
    - Global: For holistic questions ("What are the main themes?")
    - Local: For specific questions ("Who works at IBM?")
    - Hybrid: For complex questions

    Args:
        request: SearchRequest with query and optional mode

    Returns:
        SearchResponse with answer and metadata
    """
    try:
        logger.info(f"GraphRAG hybrid search: '{request.query}' (mode: {request.mode or 'auto'})")

        engine = get_hybrid_engine()
        result = engine.search(
            query=request.query,
            mode=request.mode
        )

        return SearchResponse(
            query=result.query,
            answer=result.answer,
            search_mode=result.search_mode,
            confidence=result.confidence,
            metadata=result.metadata
        )

    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/global", response_model=SearchResponse)
async def global_search(request: SearchRequest):
    """
    Execute global search (community summaries)

    Best for:
    - "What are the main themes?"
    - "Summarize the knowledge base"
    - Holistic questions

    Args:
        request: SearchRequest with query

    Returns:
        SearchResponse with answer from community summaries
    """
    try:
        logger.info(f"GraphRAG global search: '{request.query}'")

        engine = get_global_engine()
        result = engine.search(query=request.query, level=1)

        return SearchResponse(
            query=result.query,
            answer=result.answer,
            search_mode="global",
            confidence=result.confidence,
            metadata={
                "communities_searched": result.communities_searched,
                "themes": result.themes
            }
        )

    except Exception as e:
        logger.error(f"Global search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/local", response_model=SearchResponse)
async def local_search(request: SearchRequest):
    """
    Execute local search (entity-centric traversal)

    Best for:
    - "Who works at IBM?"
    - "What is Ahmed's role?"
    - Specific entity questions

    Args:
        request: SearchRequest with query

    Returns:
        SearchResponse with answer from entity traversal
    """
    try:
        logger.info(f"GraphRAG local search: '{request.query}'")

        engine = get_local_engine()
        result = engine.search(query=request.query)

        return SearchResponse(
            query=result.query,
            answer=result.answer,
            search_mode="local",
            confidence=result.confidence,
            metadata={
                "seed_entities": result.seed_entities,
                "discovered_entities": result.discovered_entities,
                "relationships": result.relationships
            }
        )

    except Exception as e:
        logger.error(f"Local search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """
    Get GraphRAG search statistics

    Returns:
        Statistics about global and local search capabilities
    """
    try:
        logger.info("Fetching GraphRAG statistics")

        engine = get_hybrid_engine()
        stats = engine.get_statistics()

        return StatsResponse(
            global_stats=stats.get("global", {}),
            local_stats=stats.get("local", {})
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communities", response_model=CommunityResponse)
async def get_communities(level: Optional[int] = None):
    """
    Get community hierarchy from Neo4j

    Args:
        level: Optional filter by community level (0, 1, 2, etc.)

    Returns:
        CommunityResponse with hierarchical community data
    """
    try:
        logger.info(f"Fetching communities (level: {level or 'all'})")

        neo4j = get_neo4j_client()

        # Build query
        if level is not None:
            query = f"""
            MATCH (c:Community)
            WHERE c.level = {level}
            OPTIONAL MATCH (c)<-[:BELONGS_TO]-(e:Entity)
            WITH c, COUNT(e) as member_count, COLLECT(e.name)[0..10] as sample_members
            OPTIONAL MATCH (c)-[:PARENT]->(parent:Community)
            RETURN
                c.id as id,
                c.level as level,
                c.summary as summary,
                c.themes as themes,
                member_count,
                sample_members,
                parent.id as parent_id
            ORDER BY c.level, c.id
            """
        else:
            query = """
            MATCH (c:Community)
            OPTIONAL MATCH (c)<-[:BELONGS_TO]-(e:Entity)
            WITH c, COUNT(e) as member_count, COLLECT(e.name)[0..10] as sample_members
            OPTIONAL MATCH (c)-[:PARENT]->(parent:Community)
            RETURN
                c.id as id,
                c.level as level,
                c.summary as summary,
                c.themes as themes,
                member_count,
                sample_members,
                parent.id as parent_id
            ORDER BY c.level, c.id
            """

        results = neo4j.execute_query(query)

        communities = []
        max_level = 0

        for record in results:
            level_val = record.get("level", 0)
            max_level = max(max_level, level_val)

            communities.append({
                "id": record.get("id"),
                "level": level_val,
                "summary": record.get("summary", ""),
                "themes": record.get("themes", []),
                "member_count": record.get("member_count", 0),
                "sample_members": record.get("sample_members", []),
                "parent_id": record.get("parent_id"),
            })

        return CommunityResponse(
            communities=communities,
            total=len(communities),
            levels=max_level + 1
        )

    except Exception as e:
        logger.error(f"Communities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns:
        Status of GraphRAG components
    """
    try:
        neo4j = get_neo4j_client()

        # Test Neo4j connection
        result = neo4j.execute_query("MATCH (e:Entity) RETURN COUNT(e) as count LIMIT 1")
        entity_count = result[0].get("count", 0) if result else 0

        # Check communities
        result = neo4j.execute_query("MATCH (c:Community) RETURN COUNT(c) as count LIMIT 1")
        community_count = result[0].get("count", 0) if result else 0

        return {
            "status": "healthy",
            "neo4j_connected": True,
            "entity_count": entity_count,
            "community_count": community_count,
            "tgi_endpoint": settings.tgi_endpoint,
        }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
