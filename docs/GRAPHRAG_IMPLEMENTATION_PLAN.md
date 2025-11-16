# GraphRAG Implementation Plan

**Date:** January 2025
**Project:** MIRAGE - Evolution to Production GraphRAG
**Reference:** [GRAPHRAG_ANALYSIS.md](./GRAPHRAG_ANALYSIS.md)

---

## Overview

This document provides a detailed, phased implementation plan to transform MIRAGE from a basic hybrid system into a production-grade GraphRAG system following Microsoft's architecture and 2024-2025 best practices.

**Estimated Timeline:** 10-12 weeks
**Complexity:** High
**Expected Impact:** 70-80% improvement in answer quality for complex queries

---

## Implementation Philosophy

### Principles

1. **Incremental Development** - Build and test each component separately
2. **Backward Compatibility** - Keep existing functionality working
3. **Evaluation-Driven** - Measure improvements at each phase
4. **Cost-Conscious** - Optimize LLM calls and compute resources

### Success Criteria

- ✅ Answer global questions ("What are the main themes?")
- ✅ Achieve 0.90+ answer relevancy on evaluation dataset
- ✅ Support 3 query types: Global, Local, Hybrid
- ✅ Process and index 1M+ token datasets
- ✅ Sub-5s query response time (p95)

---

## Phase 1: Foundation & Normalization (Weeks 1-2)

### Goals
- Fix entity normalization issues
- Enhance graph traversal capabilities
- Build evaluation framework

### Task 1.1: Entity Normalization & Deduplication

**Priority:** Critical
**Estimated Time:** 3-4 days

**Implementation:**

Create `mirage/src/core/graph_builder/entity_normalizer.py`:

```python
from typing import List, Dict, Tuple
from anthropic import Anthropic
import re

class EntityNormalizer:
    """
    Normalizes entity names to prevent duplicates like
    "Officer Johnson" and "Inspector Johnson" both existing
    when they refer to the same person.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.normalization_cache = {}

    def normalize_entity_name(self, entity_name: str, entity_type: str) -> str:
        """
        Normalize an entity name using heuristics and LLM when needed.

        Args:
            entity_name: Raw entity name
            entity_type: Entity type (Person, Organization, etc.)

        Returns:
            Normalized entity name
        """
        # Step 1: Basic normalization
        normalized = self._basic_normalization(entity_name, entity_type)

        # Step 2: Check cache
        cache_key = (normalized, entity_type)
        if cache_key in self.normalization_cache:
            return self.normalization_cache[cache_key]

        # Step 3: Advanced normalization (LLM-based for ambiguous cases)
        if entity_type == "Person":
            normalized = self._normalize_person_name(normalized)
        elif entity_type == "Organization":
            normalized = self._normalize_organization_name(normalized)

        self.normalization_cache[cache_key] = normalized
        return normalized

    def _basic_normalization(self, name: str, entity_type: str) -> str:
        """Basic string normalization."""
        # Remove extra whitespace
        name = " ".join(name.split())

        # Remove titles for persons
        if entity_type == "Person":
            titles = [
                r'\b(Dr|Mr|Mrs|Ms|Prof|Professor|Officer|Inspector|Detective|Captain|Lieutenant)\.?\s+',
                r'\b(PhD|MD|Esq)\.?\s*$'
            ]
            for title_pattern in titles:
                name = re.sub(title_pattern, '', name, flags=re.IGNORECASE)

        # Capitalize properly
        name = " ".join(word.capitalize() for word in name.split())

        return name.strip()

    def _normalize_person_name(self, name: str) -> str:
        """Normalize person names (e.g., 'John Smith' vs 'Smith, John')."""
        # Handle "Last, First" format
        if ',' in name:
            parts = name.split(',')
            if len(parts) == 2:
                last, first = parts
                name = f"{first.strip()} {last.strip()}"

        return name

    def _normalize_organization_name(self, name: str) -> str:
        """Normalize organization names."""
        # Remove common suffixes variations
        name = re.sub(r'\b(Inc|Corp|LLC|Ltd|Co)\.?\s*$', '', name, flags=re.IGNORECASE)

        # Standardize abbreviations
        replacements = {
            'Intl': 'International',
            'Dept': 'Department',
            'Gov': 'Government',
        }

        for abbr, full in replacements.items():
            name = re.sub(rf'\b{abbr}\b', full, name, flags=re.IGNORECASE)

        return name.strip()

    def find_duplicate_entities(self, entities: List[Dict]) -> List[Tuple[str, str]]:
        """
        Find potential duplicate entities in the graph.

        Args:
            entities: List of entity dicts with 'name' and 'type'

        Returns:
            List of (entity1, entity2) tuples that might be duplicates
        """
        duplicates = []
        normalized_map = {}

        for entity in entities:
            normalized = self.normalize_entity_name(entity['name'], entity['type'])

            if normalized in normalized_map:
                duplicates.append((normalized_map[normalized], entity['name']))
            else:
                normalized_map[normalized] = entity['name']

        return duplicates

    def merge_entities_in_graph(self, neo4j_client, entity1: str, entity2: str):
        """
        Merge two entity nodes in Neo4j, consolidating all relationships.

        Args:
            neo4j_client: Neo4j client instance
            entity1: First entity name (will be kept)
            entity2: Second entity name (will be merged into entity1)
        """
        merge_query = """
        // Find both entities
        MATCH (e1:Entity {name: $entity1})
        MATCH (e2:Entity {name: $entity2})

        // Copy all relationships from e2 to e1
        OPTIONAL MATCH (e2)-[r]->(other)
        WHERE NOT (e1)-[:TYPE(r)]->(other)
        WITH e1, e2, r, other
        CALL apoc.create.relationship(e1, type(r), properties(r), other) YIELD rel
        WITH e1, e2

        // Copy all relationships to e2 to e1
        OPTIONAL MATCH (other)-[r]->(e2)
        WHERE NOT (other)-[:TYPE(r)]->(e1)
        WITH e1, e2, r, other
        CALL apoc.create.relationship(other, type(r), properties(r), e1) YIELD rel
        WITH e1, e2

        // Merge properties (prefer e1, but add missing from e2)
        SET e1 += e2

        // Delete e2
        DETACH DELETE e2

        RETURN e1.name as merged_entity
        """

        neo4j_client.execute_query(merge_query, {
            'entity1': entity1,
            'entity2': entity2
        })
```

**Integration:**

Update `mirage/src/core/graph_builder/llm_entity_extractor.py`:

```python
from .entity_normalizer import EntityNormalizer

class LLMEntityExtractor:
    def __init__(self, llm_client, neo4j_client):
        self.llm_client = llm_client
        self.neo4j_client = neo4j_client
        self.normalizer = EntityNormalizer(llm_client)  # Add this

    def extract_entities(self, text: str) -> dict:
        # ... existing extraction logic ...

        # Normalize entity names
        for entity in result['entities']:
            entity['name'] = self.normalizer.normalize_entity_name(
                entity['name'],
                entity['type']
            )

        return result
```

**Testing:**
```python
# Test cases
test_entities = [
    {"name": "Officer Johnson", "type": "Person"},
    {"name": "Inspector Johnson", "type": "Person"},
    {"name": "Dr. Sarah Smith", "type": "Person"},
    {"name": "Sarah Smith, PhD", "type": "Person"},
]

normalizer = EntityNormalizer(llm_client)
for entity in test_entities:
    normalized = normalizer.normalize_entity_name(entity['name'], entity['type'])
    print(f"{entity['name']} → {normalized}")

# Expected output:
# Officer Johnson → Johnson
# Inspector Johnson → Johnson
# Dr. Sarah Smith → Sarah Smith
# Sarah Smith, PhD → Sarah Smith
```

### Task 1.2: Enhanced Graph Traversal

**Priority:** High
**Estimated Time:** 2-3 days

**Implementation:**

Create `mirage/src/core/graph_builder/graph_traversal.py`:

```python
from typing import List, Dict, Set
from dataclasses import dataclass

@dataclass
class TraversalResult:
    """Result from graph traversal."""
    entities: List[Dict]
    relationships: List[Dict]
    paths: List[List[str]]
    depth: int

class GraphTraversal:
    """Multi-hop graph traversal for entity-centric retrieval."""

    def __init__(self, neo4j_client):
        self.neo4j_client = neo4j_client

    def traverse_from_entities(
        self,
        entity_names: List[str],
        max_hops: int = 2,
        relationship_types: List[str] = None
    ) -> TraversalResult:
        """
        Traverse graph starting from seed entities.

        Args:
            entity_names: Starting entities
            max_hops: Maximum traversal depth (1-3 recommended)
            relationship_types: Optional filter for relationship types

        Returns:
            TraversalResult with all discovered entities and relationships
        """
        relationship_filter = ""
        if relationship_types:
            types_str = "|".join(relationship_types)
            relationship_filter = f":{types_str}"

        query = f"""
        // Start from seed entities
        MATCH (start:Entity)
        WHERE start.name IN $entity_names

        // Traverse up to max_hops
        CALL apoc.path.subgraphAll(start, {{
            relationshipFilter: '{relationship_filter}',
            minLevel: 0,
            maxLevel: $max_hops
        }})
        YIELD nodes, relationships

        // Return all discovered entities and relationships
        RETURN
            [node IN nodes | {{
                name: node.name,
                type: node.type,
                properties: properties(node)
            }}] as entities,
            [rel IN relationships | {{
                source: startNode(rel).name,
                target: endNode(rel).name,
                type: type(rel),
                properties: properties(rel)
            }}] as relationships
        """

        result = self.neo4j_client.execute_query(query, {
            'entity_names': entity_names,
            'max_hops': max_hops
        })

        # Process results
        if not result:
            return TraversalResult([], [], [], 0)

        entities = result[0]['entities']
        relationships = result[0]['relationships']

        # Find paths between entities
        paths = self._find_paths(entity_names, relationships)

        return TraversalResult(
            entities=entities,
            relationships=relationships,
            paths=paths,
            depth=max_hops
        )

    def _find_paths(self, entity_names: List[str], relationships: List[Dict]) -> List[List[str]]:
        """Find all paths between entities in the traversal result."""
        # Build adjacency list
        graph = {}
        for rel in relationships:
            source = rel['source']
            target = rel['target']

            if source not in graph:
                graph[source] = []
            graph[source].append(target)

        # Find paths using DFS
        paths = []
        for start in entity_names:
            for end in entity_names:
                if start != end:
                    path = self._dfs_path(graph, start, end, set(), [])
                    if path:
                        paths.append(path)

        return paths

    def _dfs_path(
        self,
        graph: Dict[str, List[str]],
        start: str,
        end: str,
        visited: Set[str],
        path: List[str]
    ) -> List[str]:
        """DFS to find path between two nodes."""
        visited.add(start)
        path.append(start)

        if start == end:
            return path.copy()

        if start in graph:
            for neighbor in graph[start]:
                if neighbor not in visited:
                    result = self._dfs_path(graph, neighbor, end, visited, path)
                    if result:
                        return result

        path.pop()
        visited.remove(start)
        return None

    def get_entity_neighbors(
        self,
        entity_name: str,
        hops: int = 1
    ) -> List[Dict]:
        """
        Get direct neighbors of an entity.

        Args:
            entity_name: Entity to start from
            hops: Number of hops (default 1 for direct neighbors)

        Returns:
            List of neighbor entities
        """
        query = """
        MATCH (start:Entity {name: $entity_name})-[*1..%d]-(neighbor:Entity)
        WHERE start <> neighbor
        RETURN DISTINCT
            neighbor.name as name,
            neighbor.type as type,
            properties(neighbor) as properties
        """ % hops

        results = self.neo4j_client.execute_query(query, {
            'entity_name': entity_name
        })

        return [dict(r) for r in results]
```

**Testing:**
```python
# Test multi-hop traversal
traversal = GraphTraversal(neo4j_client)

result = traversal.traverse_from_entities(
    entity_names=["Johnson", "Smith"],
    max_hops=2
)

print(f"Found {len(result.entities)} entities")
print(f"Found {len(result.relationships)} relationships")
print(f"Found {len(result.paths)} paths between entities")
```

### Task 1.3: Evaluation Framework

**Priority:** Critical
**Estimated Time:** 3-4 days

**Implementation:**

Create `mirage/src/evaluation/metrics.py`:

```python
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from anthropic import Anthropic

@dataclass
class EvaluationMetrics:
    """Metrics for RAG evaluation."""
    faithfulness: float  # 0-1, how factually accurate
    answer_relevancy: float  # 0-1, how relevant to question
    context_recall: float  # 0-1, how much ground truth retrieved
    context_precision: float  # 0-1, precision of retrieved context

class RAGEvaluator:
    """
    Evaluator for RAG systems using RAGAS-inspired metrics.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> EvaluationMetrics:
        """
        Evaluate a RAG answer.

        Args:
            question: The user question
            answer: Generated answer
            contexts: Retrieved contexts used
            ground_truth: Optional ground truth answer

        Returns:
            EvaluationMetrics with scores
        """
        faithfulness = self._evaluate_faithfulness(answer, contexts)
        answer_relevancy = self._evaluate_answer_relevancy(question, answer)

        context_recall = None
        context_precision = None
        if ground_truth:
            context_recall = self._evaluate_context_recall(contexts, ground_truth)
            context_precision = self._evaluate_context_precision(contexts, question)

        return EvaluationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall or 0.0,
            context_precision=context_precision or 0.0
        )

    def _evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if answer is faithful to contexts (not hallucinated).
        """
        prompt = f"""
        Evaluate if the answer is factually supported by the contexts.

        Answer: {answer}

        Contexts:
        {chr(10).join(f"{i+1}. {ctx}" for i, ctx in enumerate(contexts))}

        For each statement in the answer, check if it's supported by contexts.
        Return a score from 0.0 to 1.0 where:
        - 1.0 = All statements are supported
        - 0.5 = Half the statements are supported
        - 0.0 = No statements are supported

        Return only the numeric score.
        """

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            score = float(response.content[0].text.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def _evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate if answer is relevant to the question.
        """
        prompt = f"""
        Evaluate if this answer is relevant to the question.

        Question: {question}

        Answer: {answer}

        Rate relevancy from 0.0 to 1.0 where:
        - 1.0 = Perfectly relevant, answers the question directly
        - 0.5 = Partially relevant
        - 0.0 = Not relevant at all

        Return only the numeric score.
        """

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            score = float(response.content[0].text.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def _evaluate_context_recall(self, contexts: List[str], ground_truth: str) -> float:
        """
        Evaluate how much of the ground truth can be derived from contexts.
        """
        prompt = f"""
        Evaluate if the contexts contain information needed to answer this ground truth.

        Ground Truth: {ground_truth}

        Contexts:
        {chr(10).join(f"{i+1}. {ctx}" for i, ctx in enumerate(contexts))}

        What fraction of the ground truth facts can be found in the contexts?
        Return a score from 0.0 to 1.0.

        Return only the numeric score.
        """

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            score = float(response.content[0].text.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    def _evaluate_context_precision(self, contexts: List[str], question: str) -> float:
        """
        Evaluate precision of retrieved contexts.
        """
        if not contexts:
            return 0.0

        relevant_count = 0
        for context in contexts:
            prompt = f"""
            Is this context relevant to answering the question?

            Question: {question}

            Context: {context}

            Answer: YES or NO
            """

            response = self.llm_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )

            if "YES" in response.content[0].text.upper():
                relevant_count += 1

        return relevant_count / len(contexts)
```

**Testing:**
```python
# Test evaluation
evaluator = RAGEvaluator(llm_client)

metrics = evaluator.evaluate_answer(
    question="What is GraphRAG?",
    answer="GraphRAG is a graph-based retrieval augmented generation system that uses knowledge graphs.",
    contexts=[
        "GraphRAG uses knowledge graphs for retrieval.",
        "It was developed by Microsoft Research."
    ]
)

print(f"Faithfulness: {metrics.faithfulness:.2f}")
print(f"Answer Relevancy: {metrics.answer_relevancy:.2f}")
```

### Phase 1 Deliverables

- ✅ Entity normalization preventing duplicates
- ✅ Multi-hop graph traversal (2-3 hops)
- ✅ Evaluation framework with RAGAS metrics
- ✅ Test suite for all components
- ✅ Documentation updates

---

## Phase 2: Community Detection (Weeks 3-4)

### Goals
- Integrate Neo4j Graph Data Science library
- Implement Leiden algorithm for community detection
- Build hierarchical community structure

### Task 2.1: Neo4j GDS Setup

**Priority:** Critical
**Estimated Time:** 1-2 days

**Docker Setup:**

Update `docker-compose.yml`:

```yaml
services:
  neo4j:
    image: neo4j:5.15-enterprise  # Note: enterprise for GDS
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["graph-data-science"]
      - NEO4J_dbms_security_procedures_unrestricted=gds.*
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
```

**Note:** For community edition, we'll implement custom Leiden using `python-louvain` library.

Alternative for community edition in `requirements.txt`:
```
python-louvain==0.16
networkx==3.2
```

### Task 2.2: Community Detection Implementation

**Priority:** Critical
**Estimated Time:** 4-5 days

**Implementation:**

Create `mirage/src/core/graph_builder/community_detector.py`:

```python
import community as community_louvain
import networkx as nx
from typing import Dict, List, Set
from collections import defaultdict

class CommunityDetector:
    """
    Detect hierarchical communities in entity graph using Leiden algorithm.
    Falls back to Louvain if Leiden not available.
    """

    def __init__(self, neo4j_client, use_neo4j_gds: bool = False):
        self.neo4j_client = neo4j_client
        self.use_neo4j_gds = use_neo4j_gds

    def detect_communities(
        self,
        resolution: float = 1.0,
        max_levels: int = 5
    ) -> Dict[int, Dict[str, int]]:
        """
        Detect hierarchical communities in the entity graph.

        Args:
            resolution: Resolution parameter (higher = more communities)
            max_levels: Maximum hierarchy levels

        Returns:
            Dictionary mapping level -> {entity_name: community_id}
        """
        if self.use_neo4j_gds:
            return self._detect_with_neo4j_gds(resolution)
        else:
            return self._detect_with_louvain(resolution, max_levels)

    def _detect_with_neo4j_gds(self, resolution: float) -> Dict[int, Dict[str, int]]:
        """Use Neo4j GDS Leiden algorithm."""

        # Create graph projection
        self.neo4j_client.execute_query("""
            CALL gds.graph.drop('entity-graph', false)
        """)

        self.neo4j_client.execute_query("""
            CALL gds.graph.project(
                'entity-graph',
                'Entity',
                {
                    RELATED_TO: {
                        orientation: 'UNDIRECTED',
                        properties: 'weight'
                    }
                }
            )
        """)

        # Run Leiden
        result = self.neo4j_client.execute_query("""
            CALL gds.leiden.stream('entity-graph', {
                includeIntermediateCommunities: true,
                relationshipWeightProperty: 'weight',
                resolution: $resolution
            })
            YIELD nodeId, communityId, intermediateCommunityIds
            WITH gds.util.asNode(nodeId) AS entity, communityId, intermediateCommunityIds
            RETURN
                entity.name AS name,
                communityId,
                intermediateCommunityIds
        """, {'resolution': resolution})

        # Organize by hierarchy level
        hierarchical_communities = defaultdict(dict)

        for record in result:
            name = record['name']
            intermediate = record.get('intermediateCommunityIds', [])

            # Level 0 = finest granularity
            hierarchical_communities[0][name] = record['communityId']

            # Intermediate levels
            for level, comm_id in enumerate(intermediate, start=1):
                hierarchical_communities[level][name] = comm_id

        return dict(hierarchical_communities)

    def _detect_with_louvain(
        self,
        resolution: float,
        max_levels: int
    ) -> Dict[int, Dict[str, int]]:
        """Use NetworkX + python-louvain as fallback."""

        # Build NetworkX graph from Neo4j
        query = """
        MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
        RETURN e1.name as source, e2.name as target, r.weight as weight
        """

        results = self.neo4j_client.execute_query(query)

        # Create NetworkX graph
        G = nx.Graph()
        for record in results:
            G.add_edge(
                record['source'],
                record['target'],
                weight=record.get('weight', 1.0)
            )

        # Run Louvain algorithm iteratively for hierarchy
        hierarchical_communities = {}

        current_graph = G
        for level in range(max_levels):
            # Detect communities at this level
            partition = community_louvain.best_partition(
                current_graph,
                resolution=resolution
            )

            # Store this level
            hierarchical_communities[level] = partition

            # Check if we have more than one community
            num_communities = len(set(partition.values()))
            if num_communities <= 1:
                break

            # Build next level graph (communities become nodes)
            next_graph = nx.Graph()
            community_weights = defaultdict(float)

            for (node1, node2, data) in current_graph.edges(data=True):
                comm1 = partition[node1]
                comm2 = partition[node2]

                if comm1 != comm2:
                    edge = tuple(sorted([comm1, comm2]))
                    community_weights[edge] += data.get('weight', 1.0)

            for (comm1, comm2), weight in community_weights.items():
                next_graph.add_edge(comm1, comm2, weight=weight)

            if len(next_graph) <= 1:
                break

            current_graph = next_graph

        return hierarchical_communities

    def store_communities_in_graph(
        self,
        hierarchical_communities: Dict[int, Dict[str, int]]
    ):
        """
        Store community assignments in Neo4j.

        Creates Community nodes and BELONGS_TO relationships.
        """
        # Clear existing communities
        self.neo4j_client.execute_query("""
            MATCH (c:Community)
            DETACH DELETE c
        """)

        # Create community nodes and relationships for each level
        for level, communities in hierarchical_communities.items():
            # Group entities by community
            comm_entities = defaultdict(list)
            for entity, comm_id in communities.items():
                comm_entities[comm_id].append(entity)

            # Create Community nodes
            for comm_id, entity_list in comm_entities.items():
                create_query = """
                // Create community node
                MERGE (c:Community {
                    id: $comm_id,
                    level: $level
                })
                SET c.entity_count = $entity_count

                // Connect entities to community
                WITH c
                UNWIND $entities AS entity_name
                MATCH (e:Entity {name: entity_name})
                MERGE (e)-[:BELONGS_TO {level: $level}]->(c)
                """

                self.neo4j_client.execute_query(create_query, {
                    'comm_id': f"L{level}_C{comm_id}",
                    'level': level,
                    'entity_count': len(entity_list),
                    'entities': entity_list
                })

            # Create parent-child relationships between levels
            if level > 0:
                parent_level = level - 1
                parent_communities = hierarchical_communities[parent_level]

                parent_child_map = defaultdict(set)
                for entity, parent_comm in parent_communities.items():
                    if entity in communities:
                        child_comm = communities[entity]
                        parent_child_map[parent_comm].add(child_comm)

                for parent_comm, child_comms in parent_child_map.items():
                    for child_comm in child_comms:
                        hierarchy_query = """
                        MATCH (parent:Community {id: $parent_id})
                        MATCH (child:Community {id: $child_id})
                        MERGE (child)-[:CHILD_OF]->(parent)
                        """

                        self.neo4j_client.execute_query(hierarchy_query, {
                            'parent_id': f"L{level}_C{parent_comm}",
                            'child_id': f"L{parent_level}_C{child_comm}"
                        })

    def get_community_entities(
        self,
        community_id: str,
        level: int
    ) -> List[Dict]:
        """
        Get all entities in a community.

        Args:
            community_id: Community identifier
            level: Hierarchy level

        Returns:
            List of entity dictionaries
        """
        query = """
        MATCH (e:Entity)-[:BELONGS_TO {level: $level}]->(c:Community {id: $comm_id})
        RETURN
            e.name as name,
            e.type as type,
            properties(e) as properties
        """

        results = self.neo4j_client.execute_query(query, {
            'comm_id': community_id,
            'level': level
        })

        return [dict(r) for r in results]

    def get_community_relationships(
        self,
        community_id: str,
        level: int
    ) -> List[Dict]:
        """Get all relationships within a community."""
        query = """
        MATCH (e1:Entity)-[:BELONGS_TO {level: $level}]->(c:Community {id: $comm_id})
        MATCH (e2:Entity)-[:BELONGS_TO {level: $level}]->(c)
        MATCH (e1)-[r:RELATED_TO]->(e2)
        RETURN
            e1.name as source,
            e2.name as target,
            type(r) as relationship_type,
            properties(r) as properties
        """

        results = self.neo4j_client.execute_query(query, {
            'comm_id': community_id,
            'level': level
        })

        return [dict(r) for r in results]
```

**Testing:**
```python
# Test community detection
detector = CommunityDetector(neo4j_client, use_neo4j_gds=False)

# Detect communities
hierarchical_communities = detector.detect_communities(
    resolution=1.0,
    max_levels=5
)

print(f"Detected {len(hierarchical_communities)} hierarchy levels")
for level, communities in hierarchical_communities.items():
    num_communities = len(set(communities.values()))
    print(f"Level {level}: {num_communities} communities")

# Store in graph
detector.store_communities_in_graph(hierarchical_communities)

# Get community entities
entities = detector.get_community_entities("L0_C1", level=0)
print(f"Community L0_C1 has {len(entities)} entities")
```

### Phase 2 Deliverables

- ✅ Community detection algorithm (Leiden/Louvain)
- ✅ Hierarchical community structure (3-5 levels)
- ✅ Community storage in Neo4j
- ✅ Community retrieval methods
- ✅ Tests and benchmarks

---

## Phase 3: Community Summaries (Weeks 5-6)

### Goals
- Generate LLM summaries for each community
- Build hierarchical summaries (bottom-up)
- Store summaries efficiently

### Task 3.1: Community Summary Generator

**Priority:** Critical
**Estimated Time:** 5-6 days

**Implementation:**

Create `mirage/src/core/graph_builder/community_summarizer.py`:

```python
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CommunitySummary:
    """Community summary with metadata."""
    community_id: str
    level: int
    summary: str
    entity_count: int
    relationship_count: int
    child_summaries: List[str]
    keywords: List[str]

class CommunitySummarizer:
    """
    Generate hierarchical summaries for communities.
    """

    def __init__(self, neo4j_client, llm_client, community_detector):
        self.neo4j_client = neo4j_client
        self.llm_client = llm_client
        self.community_detector = community_detector

    def generate_all_summaries(
        self,
        hierarchical_communities: Dict[int, Dict[str, int]],
        batch_size: int = 10
    ) -> Dict[str, CommunitySummary]:
        """
        Generate summaries for all communities (bottom-up).

        Args:
            hierarchical_communities: Community structure
            batch_size: Number of summaries to generate in parallel

        Returns:
            Dictionary mapping community_id to CommunitySummary
        """
        all_summaries = {}

        # Get max level
        max_level = max(hierarchical_communities.keys())

        # Generate summaries bottom-up
        for level in range(max_level + 1):
            print(f"Generating summaries for level {level}...")

            # Get unique communities at this level
            communities_at_level = set(hierarchical_communities[level].values())

            for comm_id in communities_at_level:
                full_comm_id = f"L{level}_C{comm_id}"

                # Generate summary
                summary = self.generate_community_summary(
                    community_id=full_comm_id,
                    level=level,
                    child_summaries=all_summaries  # Pass existing summaries
                )

                all_summaries[full_comm_id] = summary

                # Store in Neo4j
                self._store_summary(summary)

        return all_summaries

    def generate_community_summary(
        self,
        community_id: str,
        level: int,
        child_summaries: Dict[str, CommunitySummary] = None
    ) -> CommunitySummary:
        """
        Generate summary for a single community.

        Args:
            community_id: Community identifier
            level: Hierarchy level
            child_summaries: Summaries from lower levels

        Returns:
            CommunitySummary
        """
        # Get entities in community
        entities = self.community_detector.get_community_entities(
            community_id, level
        )

        # Get relationships
        relationships = self.community_detector.get_community_relationships(
            community_id, level
        )

        # Get child summaries if not bottom level
        child_summary_texts = []
        if level > 0 and child_summaries:
            child_comms = self._get_child_communities(community_id, level)
            for child_id in child_comms:
                if child_id in child_summaries:
                    child_summary_texts.append(child_summaries[child_id].summary)

        # Generate summary using LLM
        summary_text = self._generate_summary_text(
            entities=entities,
            relationships=relationships,
            child_summaries=child_summary_texts
        )

        # Extract keywords
        keywords = self._extract_keywords(summary_text, entities)

        return CommunitySummary(
            community_id=community_id,
            level=level,
            summary=summary_text,
            entity_count=len(entities),
            relationship_count=len(relationships),
            child_summaries=child_summary_texts,
            keywords=keywords
        )

    def _generate_summary_text(
        self,
        entities: List[Dict],
        relationships: List[Dict],
        child_summaries: List[str]
    ) -> str:
        """Generate summary text using LLM."""

        # Format entities
        entity_text = "\n".join([
            f"- {e['name']} ({e['type']}): {e.get('properties', {}).get('description', 'No description')}"
            for e in entities[:50]  # Limit to avoid token overflow
        ])

        # Format relationships
        rel_text = "\n".join([
            f"- {r['source']} → {r['relationship_type']} → {r['target']}"
            for r in relationships[:50]
        ])

        # Format child summaries
        child_text = ""
        if child_summaries:
            child_text = "\n\nSub-community summaries:\n" + "\n".join([
                f"{i+1}. {summary[:200]}..."
                for i, summary in enumerate(child_summaries)
            ])

        prompt = f"""
        You are a knowledge graph analyst. Summarize this community of related entities.

        **Entities in this community:**
        {entity_text}

        **Relationships:**
        {rel_text}

        {child_text}

        **Instructions:**
        1. Identify the main theme or topic of this community
        2. Describe the key entities and their roles
        3. Explain the most important relationships and patterns
        4. If sub-community summaries are provided, incorporate higher-level themes
        5. Keep the summary concise but comprehensive (3-5 paragraphs)

        **Format:**
        - Start with a clear theme statement
        - Use specific entity names
        - Highlight important relationships
        - Be factual and precise

        Generate the community summary:
        """

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()

    def _extract_keywords(
        self,
        summary: str,
        entities: List[Dict]
    ) -> List[str]:
        """Extract keywords from summary."""
        # Simple approach: use entity names + LLM extraction
        entity_names = [e['name'] for e in entities]

        prompt = f"""
        Extract 5-10 key topics/keywords from this summary.

        Summary: {summary}

        Return only a comma-separated list of keywords.
        """

        response = self.llm_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        keywords = [k.strip() for k in response.content[0].text.split(',')]
        keywords.extend(entity_names[:5])  # Add top entities

        return list(set(keywords))[:10]

    def _get_child_communities(
        self,
        community_id: str,
        level: int
    ) -> List[str]:
        """Get child community IDs."""
        query = """
        MATCH (parent:Community {id: $comm_id, level: $level})
        MATCH (child:Community)-[:CHILD_OF]->(parent)
        RETURN child.id as id
        """

        results = self.neo4j_client.execute_query(query, {
            'comm_id': community_id,
            'level': level
        })

        return [r['id'] for r in results]

    def _store_summary(self, summary: CommunitySummary):
        """Store summary in Neo4j."""
        query = """
        MATCH (c:Community {id: $comm_id})
        SET c.summary = $summary,
            c.keywords = $keywords,
            c.entity_count = $entity_count,
            c.relationship_count = $relationship_count
        """

        self.neo4j_client.execute_query(query, {
            'comm_id': summary.community_id,
            'summary': summary.summary,
            'keywords': summary.keywords,
            'entity_count': summary.entity_count,
            'relationship_count': summary.relationship_count
        })

    def get_community_summary(
        self,
        community_id: str
    ) -> CommunitySummary:
        """Retrieve stored community summary."""
        query = """
        MATCH (c:Community {id: $comm_id})
        RETURN
            c.id as id,
            c.level as level,
            c.summary as summary,
            c.entity_count as entity_count,
            c.relationship_count as relationship_count,
            c.keywords as keywords
        """

        result = self.neo4j_client.execute_query(query, {
            'comm_id': community_id
        })

        if not result:
            return None

        r = result[0]
        return CommunitySummary(
            community_id=r['id'],
            level=r['level'],
            summary=r['summary'],
            entity_count=r['entity_count'],
            relationship_count=r['relationship_count'],
            child_summaries=[],  # Not stored
            keywords=r.get('keywords', [])
        )

    def get_all_summaries_at_level(
        self,
        level: int
    ) -> List[CommunitySummary]:
        """Get all community summaries at a specific level."""
        query = """
        MATCH (c:Community {level: $level})
        WHERE c.summary IS NOT NULL
        RETURN
            c.id as id,
            c.level as level,
            c.summary as summary,
            c.entity_count as entity_count,
            c.relationship_count as relationship_count,
            c.keywords as keywords
        ORDER BY c.entity_count DESC
        """

        results = self.neo4j_client.execute_query(query, {'level': level})

        return [
            CommunitySummary(
                community_id=r['id'],
                level=r['level'],
                summary=r['summary'],
                entity_count=r['entity_count'],
                relationship_count=r['relationship_count'],
                child_summaries=[],
                keywords=r.get('keywords', [])
            )
            for r in results
        ]
```

**Testing:**
```python
# Test summary generation
summarizer = CommunitySummarizer(neo4j_client, llm_client, community_detector)

# Generate all summaries
hierarchical_communities = detector.detect_communities()
all_summaries = summarizer.generate_all_summaries(hierarchical_communities)

print(f"Generated {len(all_summaries)} summaries")

# Check a summary
summary = summarizer.get_community_summary("L0_C1")
print(f"\nCommunity: {summary.community_id}")
print(f"Entities: {summary.entity_count}")
print(f"Keywords: {', '.join(summary.keywords)}")
print(f"\nSummary:\n{summary.summary}")
```

### Phase 3 Deliverables

- ✅ Community summary generation (bottom-up)
- ✅ Hierarchical summary integration
- ✅ Summary storage in Neo4j
- ✅ Keyword extraction
- ✅ Summary retrieval API

---

*(Due to length, I'll create the rest of the plan in the next section)*

## Phases 4-5 Overview

**Phase 4: Global Search (Weeks 7-8)**
- Implement map-reduce query strategy
- Community summary ranking
- Parallel partial answer generation
- Answer synthesis

**Phase 5: Local & Hybrid Search (Weeks 9-10)**
- Local search with graph traversal
- Hybrid retrieval combining vector + graph
- Query classifier and router
- Integration with existing chat service

---

## Summary of Changes Needed

### New Files to Create
1. `mirage/src/core/graph_builder/entity_normalizer.py`
2. `mirage/src/core/graph_builder/graph_traversal.py`
3. `mirage/src/core/graph_builder/community_detector.py`
4. `mirage/src/core/graph_builder/community_summarizer.py`
5. `mirage/src/core/search/global_search.py`
6. `mirage/src/core/search/local_search.py`
7. `mirage/src/core/search/hybrid_search.py`
8. `mirage/src/core/search/query_router.py`
9. `mirage/src/evaluation/metrics.py`

### Files to Modify
1. `mirage/src/api/chat_service.py` - Integrate new search strategies
2. `mirage/src/api/db_service.py` - Add community endpoints
3. `mirage/src/core/graph_builder/llm_entity_extractor.py` - Add normalization
4. `docker-compose.yml` - Update Neo4j configuration

### Configuration Changes
1. Add community detection settings to `mirage/src/config/processing.yaml`
2. Add search strategy settings to `mirage/src/config/settings.yaml`

---

## Cost Estimation

### LLM Costs (per 1M tokens indexed)

**Entity Extraction:** ~$50 (existing)
**Community Summaries:** ~$100-200 (new)
- Depends on number of communities
- Typical: 100-500 communities for 1M tokens
- ~500-1000 tokens per summary

**Query Costs:**
- Global Search: $0.01-0.05 per query (map-reduce overhead)
- Local Search: $0.005-0.01 per query
- Hybrid Search: $0.01-0.02 per query

**Total Additional Cost:** ~$150-250 per 1M tokens indexed (one-time)

### Compute Costs

**Community Detection:** Negligible (< 1 minute for 10K entities)
**Storage:** +20-30% for community nodes and summaries

---

## Success Metrics

### Quantitative
- Answer relevancy: > 0.90 (target)
- Faithfulness: > 0.85
- Global query support: 100% (currently 0%)
- Query latency: < 5s p95

### Qualitative
- Can answer "What are the main themes?"
- Can explain relationships between entities
- Provides comprehensive context
- Reduces hallucinations

---

## Next Steps

1. Review and approve this plan
2. Set up development environment
3. Begin Phase 1 implementation
4. Regular checkpoints (bi-weekly)
5. Continuous evaluation and iteration

---

**Document Version:** 1.0
**Last Updated:** January 2025
**Status:** Ready for Review
