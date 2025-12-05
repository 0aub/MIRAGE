"""
Incremental Graph Updater - Add Documents Without Full Reindex

Key Innovation from LightRAG:
- Traditional GraphRAG requires full graph rebuild on new documents
- Incremental updates only modify affected communities
- O(ΔV) complexity instead of O(V)

Algorithm:
1. Extract entities and relationships from new document
2. Match to existing entities (or create new)
3. Add relationships (may bridge communities)
4. Identify affected communities
5. Update ONLY affected community summaries

This enables:
- Sub-second document additions
- Continuous document ingestion
- Production-ready scalability
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger
import time


@dataclass
class UpdateResult:
    """Result from incremental update."""
    document_id: str
    new_entities: int = 0
    resolved_entities: int = 0
    new_relationships: int = 0
    affected_communities: int = 0
    communities_merged: int = 0
    update_time_ms: float = 0.0
    full_reindex_required: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class AffectedCommunity:
    """Community affected by update."""
    community_id: str
    reason: str  # "new_entity", "new_relationship", "bridge", "merge"
    entities_added: int = 0
    needs_summary_update: bool = True


class IncrementalGraphUpdater:
    """
    Add documents to graph without full rebuild.

    Key Principle:
    - Entities are resolved to existing graph nodes when possible
    - Only affected communities need summary regeneration
    - Community structure is updated incrementally

    Usage:
        updater = IncrementalGraphUpdater(neo4j_client, entity_extractor)
        result = updater.add_document(document)
        # Graph is now updated, only affected communities modified
    """

    def __init__(
        self,
        neo4j_client,
        entity_extractor,
        relationship_extractor=None,
        embedder=None,
        community_summarizer=None,
        similarity_threshold: float = 0.85,
        auto_merge_threshold: float = 0.7
    ):
        """
        Initialize incremental updater.

        Args:
            neo4j_client: Neo4j client for graph operations
            entity_extractor: Entity extraction component
            relationship_extractor: Relationship extraction component
            embedder: Embedding manager for similarity
            community_summarizer: Community summary generator
            similarity_threshold: Threshold for entity matching
            auto_merge_threshold: Threshold for automatic community merge
        """
        self.neo4j_client = neo4j_client
        self.entity_extractor = entity_extractor
        self.relationship_extractor = relationship_extractor
        self.embedder = embedder
        self.community_summarizer = community_summarizer
        self.similarity_threshold = similarity_threshold
        self.auto_merge_threshold = auto_merge_threshold

        logger.info(
            f"IncrementalGraphUpdater initialized: "
            f"similarity_threshold={similarity_threshold}, "
            f"auto_merge_threshold={auto_merge_threshold}"
        )

    def add_document(
        self,
        document: Dict,
        update_communities: bool = True
    ) -> UpdateResult:
        """
        Incrementally add a document to the knowledge graph.

        Steps:
        1. Extract entities from document
        2. Resolve entities to existing graph nodes (or create new)
        3. Extract and add relationships
        4. Identify affected communities
        5. Update affected community summaries

        Args:
            document: Document dict with 'id', 'text', 'chunks' etc.
            update_communities: Whether to update community summaries

        Returns:
            UpdateResult with statistics
        """
        start_time = time.time()

        document_id = document.get('id', '')
        text = document.get('text', '')
        chunks = document.get('chunks', [])

        result = UpdateResult(document_id=document_id)

        try:
            # Step 1: Extract entities
            entities = self._extract_entities(text, chunks)
            logger.info(f"Extracted {len(entities)} entities from document {document_id}")

            # Step 2: Resolve entities
            resolved_entities = []
            new_entities = []

            for entity in entities:
                existing = self._find_existing_entity(entity)
                if existing:
                    resolved_entities.append({
                        'original': entity,
                        'resolved': existing,
                        'is_new': False
                    })
                    # Update mentions
                    self._update_entity_mentions(existing, document)
                else:
                    # Create new entity
                    new_entity = self._create_entity(entity, document)
                    new_entities.append(new_entity)
                    resolved_entities.append({
                        'original': entity,
                        'resolved': new_entity,
                        'is_new': True
                    })

            result.new_entities = len(new_entities)
            result.resolved_entities = len(resolved_entities) - len(new_entities)

            # Step 3: Extract and add relationships
            if self.relationship_extractor:
                relationships = self._extract_relationships(
                    text, chunks, resolved_entities
                )
                new_relationships = self._add_relationships(relationships)
                result.new_relationships = len(new_relationships)
            else:
                new_relationships = []

            # Step 4: Identify affected communities
            affected_communities = self._find_affected_communities(
                new_entities, new_relationships
            )
            result.affected_communities = len(affected_communities)

            # Step 5: Update affected communities
            if update_communities and affected_communities:
                merged = self._update_communities(affected_communities)
                result.communities_merged = merged

            result.update_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Document {document_id} added: "
                f"{result.new_entities} new entities, "
                f"{result.resolved_entities} resolved, "
                f"{result.new_relationships} relationships, "
                f"{result.affected_communities} communities affected, "
                f"{result.update_time_ms:.1f}ms"
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error adding document {document_id}: {e}")

        return result

    def _extract_entities(
        self,
        text: str,
        chunks: List[Dict]
    ) -> List[Dict]:
        """Extract entities from document text and chunks."""
        entities = []

        try:
            if chunks:
                for chunk in chunks:
                    chunk_text = chunk.get('text', '')
                    if chunk_text:
                        chunk_entities = self.entity_extractor.extract(chunk_text)
                        for entity in chunk_entities:
                            entity['chunk_id'] = chunk.get('id', '')
                            entities.append(entity)
            else:
                entities = self.entity_extractor.extract(text)

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

        # Deduplicate by name
        seen = set()
        unique_entities = []
        for entity in entities:
            name = entity.get('name', '').lower()
            if name and name not in seen:
                seen.add(name)
                unique_entities.append(entity)

        return unique_entities

    def _find_existing_entity(self, entity: Dict) -> Optional[Dict]:
        """
        Find existing entity in graph matching the extracted entity.

        Uses multiple matching strategies:
        1. Exact name match
        2. Normalized name match
        3. Embedding similarity (if available)
        """
        entity_name = entity.get('name', '')
        entity_type = entity.get('type', '')

        if not entity_name:
            return None

        try:
            # Strategy 1: Exact match
            exact_query = """
            MATCH (e:Entity)
            WHERE e.name = $name OR e.normalized_name = $normalized
            RETURN e.name as name, e.type as type, e.id as id
            LIMIT 1
            """

            normalized = self._normalize_entity_name(entity_name)
            results = self.neo4j_client.execute_query(exact_query, {
                'name': entity_name,
                'normalized': normalized
            })

            if results:
                return results[0]

            # Strategy 2: Fuzzy match
            fuzzy_query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $term OR $term CONTAINS e.name
            RETURN e.name as name, e.type as type, e.id as id
            ORDER BY size(e.name) DESC
            LIMIT 5
            """

            results = self.neo4j_client.execute_query(fuzzy_query, {
                'term': entity_name
            })

            for r in results:
                # Check if close enough match
                existing_name = r.get('name', '')
                similarity = self._name_similarity(entity_name, existing_name)
                if similarity >= self.similarity_threshold:
                    return r

            # Strategy 3: Embedding similarity (if embedder available)
            if self.embedder:
                return self._find_by_embedding_similarity(entity_name, entity_type)

        except Exception as e:
            logger.debug(f"Entity match error for '{entity_name}': {e}")

        return None

    def _find_by_embedding_similarity(
        self,
        entity_name: str,
        entity_type: Optional[str]
    ) -> Optional[Dict]:
        """Find entity by embedding similarity."""
        try:
            # Get entity embedding
            entity_embedding = self.embedder.embed(entity_name)

            # Get candidate entities with embeddings
            query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            RETURN e.name as name, e.type as type, e.embedding as embedding, e.id as id
            LIMIT 100
            """

            results = self.neo4j_client.execute_query(query)

            best_match = None
            best_similarity = 0.0

            for r in results:
                embedding = r.get('embedding')
                if embedding:
                    import numpy as np
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)

                    similarity = np.dot(entity_embedding, embedding) / (
                        np.linalg.norm(entity_embedding) * np.linalg.norm(embedding)
                    )

                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = r

            return best_match

        except Exception:
            return None

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        import re

        # Lowercase
        normalized = name.lower().strip()

        # Remove common prefixes/suffixes
        prefixes = ['ال', 'the ', 'a ', 'an ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names."""
        n1 = self._normalize_entity_name(name1)
        n2 = self._normalize_entity_name(name2)

        # Simple Jaccard similarity on character trigrams
        def get_trigrams(s: str) -> Set[str]:
            s = f"  {s}  "
            return set(s[i:i+3] for i in range(len(s) - 2))

        t1 = get_trigrams(n1)
        t2 = get_trigrams(n2)

        if not t1 or not t2:
            return 0.0

        intersection = len(t1 & t2)
        union = len(t1 | t2)

        return intersection / union if union > 0 else 0.0

    def _update_entity_mentions(self, entity: Dict, document: Dict):
        """Update entity with new document mention."""
        try:
            query = """
            MATCH (e:Entity {name: $name})
            SET e.mention_count = COALESCE(e.mention_count, 0) + 1,
                e.document_ids = COALESCE(e.document_ids, []) + [$doc_id]
            """

            self.neo4j_client.execute_query(query, {
                'name': entity.get('name'),
                'doc_id': document.get('id', '')
            })

        except Exception as e:
            logger.debug(f"Failed to update entity mentions: {e}")

    def _create_entity(self, entity: Dict, document: Dict) -> Dict:
        """Create new entity in graph."""
        try:
            name = entity.get('name', '')
            entity_type = entity.get('type', 'ENTITY')
            normalized = self._normalize_entity_name(name)

            # Generate embedding if available
            embedding = None
            if self.embedder:
                try:
                    embedding = self.embedder.embed(name)
                    if isinstance(embedding, list):
                        embedding = embedding
                    else:
                        embedding = embedding.tolist()
                except Exception:
                    pass

            query = """
            CREATE (e:Entity {
                name: $name,
                type: $type,
                normalized_name: $normalized,
                document_ids: [$doc_id],
                mention_count: 1,
                created_at: datetime()
            })
            RETURN e.name as name, e.type as type
            """

            params = {
                'name': name,
                'type': entity_type,
                'normalized': normalized,
                'doc_id': document.get('id', '')
            }

            if embedding:
                query = """
                CREATE (e:Entity {
                    name: $name,
                    type: $type,
                    normalized_name: $normalized,
                    document_ids: [$doc_id],
                    mention_count: 1,
                    embedding: $embedding,
                    created_at: datetime()
                })
                RETURN e.name as name, e.type as type
                """
                params['embedding'] = embedding

            results = self.neo4j_client.execute_query(query, params)
            return results[0] if results else {'name': name, 'type': entity_type}

        except Exception as e:
            logger.error(f"Failed to create entity '{entity.get('name', '')}': {e}")
            return entity

    def _extract_relationships(
        self,
        text: str,
        chunks: List[Dict],
        resolved_entities: List[Dict]
    ) -> List[Dict]:
        """Extract relationships between resolved entities."""
        relationships = []

        if not self.relationship_extractor:
            return relationships

        try:
            # Get entity names for extraction
            entity_names = [e['resolved'].get('name', '') for e in resolved_entities]

            if chunks:
                for chunk in chunks:
                    chunk_text = chunk.get('text', '')
                    if chunk_text:
                        chunk_rels = self.relationship_extractor.extract(
                            chunk_text, entity_names
                        )
                        for rel in chunk_rels:
                            rel['chunk_id'] = chunk.get('id', '')
                            relationships.append(rel)
            else:
                relationships = self.relationship_extractor.extract(text, entity_names)

        except Exception as e:
            logger.warning(f"Relationship extraction failed: {e}")

        return relationships

    def _add_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Add relationships to graph."""
        added = []

        for rel in relationships:
            try:
                source = rel.get('source', '')
                target = rel.get('target', '')
                rel_type = rel.get('type', 'RELATED_TO')

                if not source or not target:
                    continue

                # Clean relationship type for Cypher
                rel_type = rel_type.upper().replace(' ', '_').replace('-', '_')

                query = f"""
                MATCH (s:Entity {{name: $source}})
                MATCH (t:Entity {{name: $target}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r.weight = COALESCE(r.weight, 0) + 1
                RETURN s.name as source, t.name as target
                """

                results = self.neo4j_client.execute_query(query, {
                    'source': source,
                    'target': target
                })

                if results:
                    added.append({
                        'source': source,
                        'target': target,
                        'type': rel_type
                    })

            except Exception as e:
                logger.debug(f"Failed to add relationship: {e}")

        return added

    def _find_affected_communities(
        self,
        new_entities: List[Dict],
        new_relationships: List[Dict]
    ) -> List[AffectedCommunity]:
        """
        Find communities affected by updates.

        A community is affected if:
        1. Contains a new entity (needs to include it)
        2. Connected to a community with new entity
        3. New relationship bridges two communities (may merge)
        """
        affected = {}  # community_id -> AffectedCommunity

        # Affected by new entities
        for entity in new_entities:
            entity_name = entity.get('name', '')
            community = self._detect_community_for_entity(entity_name)
            if community:
                if community not in affected:
                    affected[community] = AffectedCommunity(
                        community_id=community,
                        reason="new_entity",
                        entities_added=0
                    )
                affected[community].entities_added += 1

        # Check for community bridges
        for rel in new_relationships:
            source = rel.get('source', '')
            target = rel.get('target', '')

            source_community = self._get_entity_community(source)
            target_community = self._get_entity_community(target)

            if source_community and target_community:
                if source_community != target_community:
                    # Relationship bridges two communities
                    if source_community not in affected:
                        affected[source_community] = AffectedCommunity(
                            community_id=source_community,
                            reason="bridge"
                        )
                    if target_community not in affected:
                        affected[target_community] = AffectedCommunity(
                            community_id=target_community,
                            reason="bridge"
                        )

        return list(affected.values())

    def _detect_community_for_entity(self, entity_name: str) -> Optional[str]:
        """Detect which community a new entity should belong to."""
        try:
            # Find neighbors and their communities
            query = """
            MATCH (e:Entity {name: $name})-[]-(neighbor:Entity)
            MATCH (neighbor)-[:BELONGS_TO]->(c:Community)
            RETURN c.id as community_id, COUNT(*) as connection_count
            ORDER BY connection_count DESC
            LIMIT 1
            """

            results = self.neo4j_client.execute_query(query, {'name': entity_name})
            if results:
                return results[0].get('community_id')

        except Exception:
            pass

        return None

    def _get_entity_community(self, entity_name: str) -> Optional[str]:
        """Get the community an entity belongs to."""
        try:
            query = """
            MATCH (e:Entity {name: $name})-[:BELONGS_TO]->(c:Community)
            RETURN c.id as community_id
            LIMIT 1
            """

            results = self.neo4j_client.execute_query(query, {'name': entity_name})
            if results:
                return results[0].get('community_id')

        except Exception:
            pass

        return None

    def _update_communities(
        self,
        affected: List[AffectedCommunity]
    ) -> int:
        """Update affected communities."""
        merged_count = 0

        for community in affected:
            try:
                if community.reason == "bridge":
                    # Check if communities should merge
                    # For now, just mark for summary update
                    pass

                if community.needs_summary_update and self.community_summarizer:
                    self._regenerate_community_summary(community.community_id)

            except Exception as e:
                logger.debug(f"Failed to update community {community.community_id}: {e}")

        return merged_count

    def _regenerate_community_summary(self, community_id: str):
        """Regenerate summary for a single community."""
        if not self.community_summarizer:
            return

        try:
            # Get community entities
            query = """
            MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: $id})
            RETURN COLLECT(e.name) as entities
            """

            results = self.neo4j_client.execute_query(query, {'id': community_id})
            if not results:
                return

            entities = results[0].get('entities', [])

            # Generate new summary
            summary = self.community_summarizer.summarize_community(entities)

            # Update in graph
            update_query = """
            MATCH (c:Community {id: $id})
            SET c.summary = $summary,
                c.updated_at = datetime()
            """

            self.neo4j_client.execute_query(update_query, {
                'id': community_id,
                'summary': summary
            })

            logger.debug(f"Updated summary for community {community_id}")

        except Exception as e:
            logger.warning(f"Failed to regenerate summary for {community_id}: {e}")

    def remove_document(self, document_id: str) -> UpdateResult:
        """
        Remove a document and update graph accordingly.

        Note: This is more complex than adding, as we need to:
        1. Remove chunk nodes
        2. Potentially remove orphan entities
        3. Update community memberships
        """
        result = UpdateResult(document_id=document_id)
        start_time = time.time()

        try:
            # Remove chunks for this document
            chunk_query = """
            MATCH (chunk:Chunk {document_id: $doc_id})
            DETACH DELETE chunk
            RETURN COUNT(*) as deleted
            """

            chunk_result = self.neo4j_client.execute_query(
                chunk_query, {'doc_id': document_id}
            )

            # Remove document reference from entities
            entity_query = """
            MATCH (e:Entity)
            WHERE $doc_id IN e.document_ids
            SET e.document_ids = [id IN e.document_ids WHERE id <> $doc_id],
                e.mention_count = e.mention_count - 1
            RETURN e.name as name
            """

            self.neo4j_client.execute_query(entity_query, {'doc_id': document_id})

            # Clean up orphan entities (no more document references)
            orphan_query = """
            MATCH (e:Entity)
            WHERE e.mention_count <= 0 OR size(e.document_ids) = 0
            DETACH DELETE e
            RETURN COUNT(*) as deleted
            """

            self.neo4j_client.execute_query(orphan_query)

            result.update_time_ms = (time.time() - start_time) * 1000

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error removing document {document_id}: {e}")

        return result


# Singleton
_incremental_updater: Optional[IncrementalGraphUpdater] = None


def get_incremental_updater(
    neo4j_client=None,
    entity_extractor=None,
    **kwargs
) -> IncrementalGraphUpdater:
    """Get or create incremental updater singleton."""
    global _incremental_updater

    if _incremental_updater is None:
        if neo4j_client is None:
            from . import Neo4jClient
            neo4j_client = Neo4jClient()

        if entity_extractor is None:
            from . import EntityExtractor
            entity_extractor = EntityExtractor()

        _incremental_updater = IncrementalGraphUpdater(
            neo4j_client=neo4j_client,
            entity_extractor=entity_extractor,
            **kwargs
        )

    return _incremental_updater
