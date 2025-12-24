import pytest
from unittest.mock import MagicMock, patch, ANY
from src.core.retrieval.retrieval_engine import RetrievalEngine, RetrievalMode
from src.core.retrieval.base_retriever import RetrievalResult

@pytest.fixture
def mock_graph_client():
    client = MagicMock()
    # Mock get_relationships_between
    client.get_relationships_between.return_value = [
        {"source": "EntityA", "target": "EntityB", "type": "RELATED_TO", "description": "Works with"},
        {"source": "EntityB", "target": "EntityC", "type": "FOUNDED", "description": "Started company"}
    ]
    # Mock get_entity_communities
    client.get_entity_communities.return_value = [
        {"community_id": "c1", "title": "Tech Giants", "summary": "A community of large tech companies."}
    ]
    return client

@pytest.fixture
def mock_index_manager():
    manager = MagicMock()
    # Mock search_chunks to return some results
    manager.search_chunks.return_value = [
        MagicMock(id="chunk1", score=0.9, payload={"text": "EntityA is a great person.", "document_id": "doc1"}),
        MagicMock(id="chunk2", score=0.8, payload={"text": "EntityB works at EntityC.", "document_id": "doc1"})
    ]
    # Mock keyword_search
    manager.keyword_search.return_value = []
    return manager

@pytest.fixture
def engine(mock_graph_client, mock_index_manager):
    engine = RetrievalEngine(
        graph_client=mock_graph_client,
        index_manager=mock_index_manager
    )
    # Mock embedder
    engine._embedder = MagicMock()
    engine._embedder.embed.return_value = [0.1] * 1536
    
    # Mock entity extraction from chunks
    engine.graph_client.get_entities_from_chunks.return_value = [
        {"name": "EntityA", "type": "Person"},
        {"name": "EntityB", "type": "Person"},
        {"name": "EntityC", "type": "Organization"}
    ]
    
    return engine

def test_local_search_injects_graph_context(engine):
    """Test that local search injects relationships and community summaries."""
    
    response = engine.retrieve(
        query="Who is EntityA?",
        mode=RetrievalMode.LOCAL,
        top_k=5
    )
    
    # Check results
    assert len(response.results) > 0
    
    # Verify Graph Relationship Injection
    graph_rels = [r for r in response.results if r.retrieval_mode == "graph_relationship"]
    assert len(graph_rels) == 2
    assert "Works with" in graph_rels[0].text
    
    # Verify Community Context Injection
    comm_context = [r for r in response.results if r.retrieval_mode == "community_context"]
    assert len(comm_context) == 1
    assert "Tech Giants" in comm_context[0].text
    
    # Verify Metadata
    assert response.metadata.get("graph_context_added") is True

def test_graph_client_calls(engine, mock_graph_client):
    """Verify that graph client methods are called with correct parameters."""
    
    engine.retrieve("Who is EntityA?", mode=RetrievalMode.LOCAL)
    
    # Check if get_relationships_between was called
    mock_graph_client.get_relationships_between.assert_called()
    call_args = mock_graph_client.get_relationships_between.call_args
    assert len(call_args[0][0]) > 0  # Should pass a list of entities
    
    # Check if get_entity_communities was called
    mock_graph_client.get_entity_communities.assert_called()

def test_context_limit_safety(engine, mock_graph_client):
    """Verify that limits are respected (to fit SLM context)."""
    
    # Return many relationships
    mock_graph_client.get_relationships_between.return_value = [
        {"source": f"A{i}", "target": f"B{i}", "type": "REL", "description": "desc"}
        for i in range(50)
    ]
    
    response = engine.retrieve("Query", mode=RetrievalMode.LOCAL)
    
    # Should not crash/error, and should handle the list
    # The logic in retrieval_engine limits use (e.g. loops over result)
    # But wait, retrieval_engine calls .get_relationships_between(..., limit=15)
    
    mock_graph_client.get_relationships_between.assert_called_with(
        ANY, # entity names
        limit=15
    )
