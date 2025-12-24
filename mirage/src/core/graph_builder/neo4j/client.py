"""
Neo4j Client - Main Client Class

Dynamic graph storage without predefined schemas.
Nodes and relationships emerge from the data.

MIRAGE V4 Enhancement:
- Dynamic relationship types (not just RELATED_TO)
- Relationship type normalization (Arabic/English)
- Semantic relationship categories
"""

from .base import Neo4jClientBase
from .entity_ops import EntityOperationsMixin
from .document_ops import DocumentOperationsMixin
from .search_ops import SearchOperationsMixin


class Neo4jClient(
    Neo4jClientBase,
    EntityOperationsMixin,
    DocumentOperationsMixin,
    SearchOperationsMixin
):
    """
    Client for interacting with Neo4j graph database.

    Combines all operations:
    - Connection management (base)
    - Entity and relationship creation (entity_ops)
    - Document storage (document_ops)
    - Search and query operations (search_ops)

    Usage:
        # As context manager (recommended)
        with Neo4jClient() as client:
            client.store_graph(entities, relationships, document_id)

        # Or manual connection
        client = Neo4jClient()
        client.connect()
        try:
            client.store_graph(entities, relationships, document_id)
        finally:
            client.close()
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize Neo4j client

        Args:
            uri: Neo4j connection URI (defaults to settings.neo4j_uri)
            user: Username (defaults to settings.neo4j_user)
            password: Password (defaults to settings.neo4j_password)
        """
        super().__init__(uri=uri, user=user, password=password)
