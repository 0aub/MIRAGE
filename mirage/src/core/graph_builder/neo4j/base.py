"""
Neo4j Client - Base Connection Management

Connection handling, configuration, and low-level query execution.
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver
from loguru import logger

from ....config import settings


class Neo4jClientBase:
    """Base class for Neo4j connection management"""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Initialize Neo4j client

        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
        """
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password

        self.driver: Optional[Driver] = None
        self._connected = False

    def connect(self):
        """Establish connection to Neo4j"""
        if self._connected:
            return

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Closed Neo4j connection")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query and return results

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Returns:
            List of result records as dictionaries
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get overall graph statistics

        Returns:
            Statistics dict
        """
        if not self._connected:
            self.connect()

        stats = {}

        try:
            with self.driver.session() as session:
                # Total nodes
                result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                stats["total_nodes"] = result.single()["count"]

                # Total relationships
                result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                stats["total_edges"] = result.single()["count"]

                # Nodes by type
                result = session.run("""
                    MATCH (n:Entity)
                    RETURN n.type as type, count(n) as count
                    ORDER BY count DESC
                """)
                stats["node_types"] = {record["type"]: record["count"] for record in result}

                # Relationship types
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY count DESC
                """)
                stats["edge_types"] = {record["type"]: record["count"] for record in result}

        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")

        return stats

    def clear_graph(self) -> Dict[str, Any]:
        """
        Clear all nodes and relationships from the graph
        WARNING: This is a destructive operation

        Returns:
            Statistics about cleared data
        """
        if not self._connected:
            self.connect()

        try:
            with self.driver.session() as session:
                # Get counts before deletion
                node_count_result = session.run("MATCH (n) RETURN count(n) as count")
                nodes_before = node_count_result.single()["count"]

                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rels_before = rel_count_result.single()["count"]

                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")

                logger.info(f"Cleared graph: {nodes_before} nodes, {rels_before} relationships")

                return {
                    "nodes_deleted": nodes_before,
                    "relationships_deleted": rels_before,
                    "status": "success",
                }

        except Exception as e:
            logger.error(f"Error clearing graph: {e}")
            raise
