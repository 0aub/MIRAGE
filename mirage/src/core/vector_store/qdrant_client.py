"""
Qdrant Vector Store
Stores and retrieves chunk embeddings for semantic search
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from loguru import logger
import uuid

from ...config import settings
from ...config.config_loader import get_config_loader


class QdrantVectorStore:
    """Vector store using Qdrant for semantic search"""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize Qdrant vector store

        Args:
            collection_name: Name of the collection (loaded from config if not provided)
            host: Qdrant host (loaded from config if not provided)
            port: Qdrant port (loaded from config if not provided)
        """
        # Load configuration
        config_loader = get_config_loader()
        qdrant_config = config_loader.get_qdrant_config()

        self.collection_name = collection_name or qdrant_config.get("collection_name", "mirage_chunks")
        self.host = host or qdrant_config.get("host", "qdrant")
        self.port = port or qdrant_config.get("port", 6333)

        # Load vector configuration
        vector_config = config_loader.get_qdrant_vector_config()
        self.vector_dimension = vector_config.get("dimension", 1024)
        self.distance_metric = vector_config.get("distance", "COSINE")

        # Load search configuration
        search_config = config_loader.get_qdrant_search_config()
        self.default_limit = search_config.get("default_limit", 10)
        self.score_threshold = search_config.get("score_threshold", 0.5)

        # Initialize client
        self.client = QdrantClient(host=self.host, port=self.port)

        # Ensure collection exists
        self._ensure_collection()

        logger.info(
            f"Initialized QdrantVectorStore: collection={self.collection_name}, "
            f"host={self.host}:{self.port}, dim={self.vector_dimension}, "
            f"distance={self.distance_metric}"
        )
    
    def _ensure_collection(self):
        """Ensure the collection exists"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection with configured dimensions and distance metric
                distance_map = {
                    "COSINE": Distance.COSINE,
                    "EUCLIDEAN": Distance.EUCLID,
                    "DOT": Distance.DOT,
                }
                distance = distance_map.get(self.distance_metric, Distance.COSINE)

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=distance,
                    ),
                )
                logger.info(
                    f"Created collection: {self.collection_name} "
                    f"(dim={self.vector_dimension}, distance={self.distance_metric})"
                )
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Add chunks with embeddings to vector store
        
        Args:
            chunks: List of chunks with embeddings
            document_id: Document ID
            
        Returns:
            Storage statistics
        """
        if not chunks:
            return {"chunks_added": 0}
        
        logger.info(f"Adding {len(chunks)} chunks to vector store for doc {document_id}")
        
        points = []
        for i, chunk in enumerate(chunks):
            # Get embedding
            embedding = chunk.get("embedding")
            if not embedding:
                logger.warning(f"Chunk {i} missing embedding, skipping")
                continue
            
            # Create point
            point_id = str(uuid.uuid4())
            
            # Prepare payload (metadata)
            payload = {
                "document_id": document_id,
                "chunk_index": chunk.get("chunk_index", i),
                "text": chunk.get("text", ""),
                "char_count": chunk.get("char_count", 0),
                "word_count": chunk.get("word_count", 0),
                "sentence_count": chunk.get("sentence_count", 0),
            }
            
            # Add metadata if present
            if "metadata" in chunk:
                payload["metadata"] = chunk["metadata"]
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
            
            points.append(point)
        
        # Upload points
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            
            logger.info(f"Added {len(points)} points to vector store")
            
            return {
                "chunks_added": len(points),
                "document_id": document_id,
            }
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        limit: Optional[int] = None,
        document_id: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks

        Args:
            query_embedding: Query embedding vector
            limit: Maximum results (uses config default if not provided)
            document_id: Optional filter by document
            score_threshold: Minimum similarity score (uses config default if not provided)

        Returns:
            List of matching chunks with scores
        """
        # Use configured defaults if not provided
        limit = limit or self.default_limit
        score_threshold = score_threshold or self.score_threshold

        try:
            # Build filter if document_id provided
            query_filter = None
            if document_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                )
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )
            
            # Format results
            chunks = []
            for result in results:
                chunk = {
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "document_id": result.payload.get("document_id", ""),
                    "metadata": result.payload.get("metadata", {}),
                }
                chunks.append(chunk)
            
            logger.info(f"Vector search returned {len(chunks)} results")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def delete_by_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            Deletion statistics
        """
        try:
            # Delete points matching document_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            
            logger.info(f"Deleted chunks for document: {document_id}")
            
            return {
                "deleted": True,
                "document_id": document_id,
            }
            
        except Exception as e:
            logger.error(f"Error deleting from vector store: {e}")
            raise
    
    def count_chunks(self, document_id: str) -> int:
        """
        Count chunks for a specific document

        Args:
            document_id: Document ID

        Returns:
            Number of chunks for the document
        """
        try:
            # Count points matching document_id
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )

            count = count_result.count if count_result else 0
            logger.debug(f"Document {document_id} has {count} chunks in vector store")

            return count

        except Exception as e:
            logger.error(f"Error counting chunks for document {document_id}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "indexed": collection_info.status,
            }

        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {}
