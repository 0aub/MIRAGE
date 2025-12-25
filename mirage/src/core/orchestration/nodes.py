"""
Workflow Nodes
Individual operations in the LangGraph RAG pipeline
"""

import re
import time
from typing import Dict, Any, Optional
from loguru import logger

from ..graph_builder import Neo4jClient
from ..graph_builder.enhanced_neo4j_client import EnhancedNeo4jClient
from ..graph_builder import GraphTraversal, HybridSearchEngine
from ..embeddings import JinaEmbedder
from .claude_client import ClaudeClient
from .workflow_state import WorkflowState
from ...config.settings import settings


class WorkflowNodes:
    """Collection of nodes for the RAG workflow"""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        claude_client: ClaudeClient,
    ):
        """
        Initialize workflow nodes

        Args:
            neo4j_client: Neo4j client for graph retrieval
            claude_client: Claude API client
        """
        self.neo4j = neo4j_client
        self.claude = claude_client

        # Initialize enhanced Neo4j client with bilingual search and vector embeddings
        try:
            self.embedder = JinaEmbedder()
            self.enhanced_neo4j = EnhancedNeo4jClient(
                embedder=self.embedder,
                llm_client=claude_client
            )
            logger.info("Initialized EnhancedNeo4jClient with hybrid search capabilities")
        except Exception as e:
            logger.warning(f"Could not initialize EnhancedNeo4jClient: {e}. Falling back to standard client.")
            self.enhanced_neo4j = None
            self.embedder = None

        # Initialize GraphRAG components (Phase 1-5)
        try:
            self.graph_traversal = GraphTraversal(neo4j_client)
            self.hybrid_search = HybridSearchEngine(
                neo4j_client=neo4j_client,
                graph_traversal=self.graph_traversal,
                llm_endpoint=settings.tgi_endpoint,
                auto_route=True  # Automatic query routing
            )
            logger.info("Initialized GraphRAG HybridSearchEngine with auto-routing")
        except Exception as e:
            logger.warning(f"Could not initialize GraphRAG components: {e}. Using standard retrieval.")
            self.graph_traversal = None
            self.hybrid_search = None

        logger.info("Initialized WorkflowNodes")
    
    def query_analysis_node(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze query and extract entities/intents using LLM

        Args:
            state: Current workflow state

        Returns:
            Updated state with query analysis
        """
        start_time = time.time()

        query = state["query"]
        logger.info(f"Analyzing query: {query}")

        # Use LLM to extract key entities and concepts from the query
        try:
            extraction_prompt = f"""Extract search keywords in BOTH English and Arabic from the question below. Return ONLY the keywords separated by commas. Do NOT include explanations, citations, or extra text.

Examples:
Question: Who won the best innovation award in 2023?
Keywords: innovation award, جائزة الابتكار, 2023, winner, فائز

Question: What are the ministry's digital transformation achievements?
Keywords: ministry, وزارة, digital transformation, التحول الرقمي, achievements, إنجازات

Question: {query}
Keywords:"""

            # Use Claude client to extract entities
            result = self.claude.generate_response(
                query=extraction_prompt,
                context="",
                system_prompt="You are a bilingual keyword extractor. Extract keywords in BOTH English and Arabic. Return ONLY keywords separated by commas.",
                max_tokens=150
            )

            # Parse and clean the response to get entity names
            entity_text = result.get("response", "").strip()

            # Remove common artifacts and clean up
            # Remove anything after newlines (hallucinated explanations)
            entity_text = entity_text.split('\n')[0]
            # Remove citation markers like [1], [2], etc.
            entity_text = re.sub(r'\[\d+\]', '', entity_text)
            # Remove parenthetical explanations
            entity_text = re.sub(r'\([^)]*\)', '', entity_text)

            # Split by comma and clean each entity
            potential_entities = []
            for e in entity_text.split(","):
                e = e.strip()
                # Skip empty, very short, or numeric-only terms
                if len(e) > 2 and not e.isdigit():
                    # Remove any remaining special characters except spaces and hyphens
                    e = re.sub(r'[^\w\s\-]', '', e)
                    if e:
                        potential_entities.append(e)

            logger.info(f"LLM extracted entities: {potential_entities}")

        except Exception as e:
            logger.warning(f"LLM entity extraction failed, falling back to keyword extraction: {e}")
            # Fallback to simple keyword extraction
            query_words = query.lower().split()
            # Extract meaningful keywords (longer than 3 chars, not common words)
            common_words = {"what", "who", "when", "where", "how", "which", "are", "the", "is", "in", "of", "and", "to", "for", "with"}
            potential_entities = [
                word for word in query_words
                if len(word) > 3 and word not in common_words
            ]

        # Search for entities in graph using enhanced hybrid search if available
        entities_found = []

        if self.enhanced_neo4j:
            # Use hybrid search (semantic + keyword) for better bilingual matching
            try:
                # Search with full query first
                entities_found = self.enhanced_neo4j.hybrid_search_entities(
                    query=query,
                    limit=10,
                    keyword_weight=0.3,  # 30% keyword matching
                    semantic_weight=0.7  # 70% semantic similarity
                )
                logger.info(f"Hybrid search found {len(entities_found)} entities")

                # If still no results, try individual keywords
                if not entities_found and potential_entities:
                    for entity_name in potential_entities[:5]:
                        results = self.enhanced_neo4j.hybrid_search_entities(
                            query=entity_name,
                            limit=3,
                            keyword_weight=0.3,
                            semantic_weight=0.7
                        )
                        entities_found.extend(results)
                    logger.info(f"Keyword-based hybrid search found {len(entities_found)} entities")
            except Exception as e:
                logger.warning(f"Enhanced search failed: {e}. Falling back to standard search.")
                # Fallback to standard search
                for entity_name in potential_entities[:10]:
                    results = self.neo4j.search_entities(entity_name, limit=5)
                    entities_found.extend(results)
        else:
            # Fallback to standard keyword search
            for entity_name in potential_entities[:10]:
                results = self.neo4j.search_entities(entity_name, limit=5)
                entities_found.extend(results)

            # If no entities found, try fuzzy search with query terms
            if not entities_found:
                logger.info("No exact matches, trying broader search")
                for word in potential_entities[:5]:
                    results = self.neo4j.search_entities(word, limit=3)
                    entities_found.extend(results)

        latency = (time.time() - start_time) * 1000

        state["entities_found"] = entities_found
        state["workflow_step"] = "query_analysis"
        if "latency_ms" not in state:
            state["latency_ms"] = {}
        state["latency_ms"]["query_analysis"] = latency

        logger.info(f"Found {len(entities_found)} entities (latency={latency:.1f}ms)")

        return state
    
    def graph_retrieval_node(self, state: WorkflowState) -> WorkflowState:
        """
        Retrieve relevant subgraph from Neo4j using GraphRAG HybridSearch

        GraphRAG Enhancement:
        - Auto-routes query to Global (community summaries) or Local (entity traversal)
        - Global: Best for holistic questions ("What are the main themes?")
        - Local: Best for specific questions ("Who works at IBM?")
        - Hybrid: Combines both for complex queries

        Args:
            state: Current workflow state

        Returns:
            Updated state with retrieved subgraph and GraphRAG metadata
        """
        start_time = time.time()

        query = state.get("query", "")
        entities = state.get("entities_found", [])

        # Try GraphRAG first if available
        if self.hybrid_search:
            try:
                logger.info(f"Using GraphRAG HybridSearch for query: {query}")

                # Execute GraphRAG search (auto-routes to Global/Local/Hybrid)
                graphrag_result = self.hybrid_search.search(query)

                logger.info(
                    f"GraphRAG search completed: mode={graphrag_result.search_mode}, "
                    f"confidence={graphrag_result.confidence:.2f}"
                )

                # Extract GraphRAG metadata for explainability
                graphrag_metadata = {
                    "search_mode": graphrag_result.search_mode,
                    "confidence": graphrag_result.confidence,
                }

                # Extract nodes, edges, communities based on search mode
                all_nodes = []
                all_edges = []
                communities = []

                if graphrag_result.search_mode == "local":
                    # Local search metadata
                    local_meta = graphrag_result.metadata
                    graphrag_metadata["seed_entities"] = local_meta.get("seed_entities", [])
                    graphrag_metadata["discovered_entities"] = local_meta.get("discovered_entities", 0)
                    graphrag_metadata["relationships"] = local_meta.get("relationships", 0)

                    # Retrieve actual subgraph for visualization
                    for entity_name in local_meta.get("seed_entities", [])[:3]:
                        subgraph = self.neo4j.query_subgraph(entity_name, depth=2)
                        all_nodes.extend(subgraph.get("nodes", []))
                        all_edges.extend(subgraph.get("edges", []))

                elif graphrag_result.search_mode == "global":
                    # Global search metadata
                    global_meta = graphrag_result.metadata
                    graphrag_metadata["communities_searched"] = global_meta.get("communities_searched", 0)
                    graphrag_metadata["themes"] = global_meta.get("themes", [])

                    # Retrieve community data for visualization
                    communities = self._get_community_data()

                elif graphrag_result.search_mode == "hybrid":
                    # Hybrid metadata combines both
                    hybrid_meta = graphrag_result.metadata
                    if "local" in hybrid_meta:
                        graphrag_metadata["seed_entities"] = hybrid_meta["local"].get("seed_entities", [])
                        graphrag_metadata["discovered_entities"] = hybrid_meta["local"].get("discovered_entities", 0)
                    if "global" in hybrid_meta:
                        graphrag_metadata["communities_searched"] = hybrid_meta["global"].get("communities_searched", 0)
                        graphrag_metadata["themes"] = hybrid_meta["global"].get("themes", [])

                    # Get both subgraph and communities
                    if "local" in hybrid_meta:
                        for entity_name in hybrid_meta["local"].get("seed_entities", [])[:3]:
                            subgraph = self.neo4j.query_subgraph(entity_name, depth=2)
                            all_nodes.extend(subgraph.get("nodes", []))
                            all_edges.extend(subgraph.get("edges", []))
                    communities = self._get_community_data()

                # Deduplicate nodes
                unique_nodes = self._deduplicate_nodes(all_nodes)

                # Create chunks from GraphRAG answer
                retrieved_chunks = [{
                    "text": graphrag_result.answer,
                    "source": "graphrag",
                    "search_mode": graphrag_result.search_mode,
                    "metadata": graphrag_metadata,
                }]

                latency = (time.time() - start_time) * 1000

                state["subgraph"] = {
                    "nodes": unique_nodes,
                    "edges": all_edges,
                    "communities": communities,
                    "node_count": len(unique_nodes),
                    "edge_count": len(all_edges),
                    "community_count": len(communities),
                }
                state["retrieved_chunks"] = retrieved_chunks
                state["graphrag_metadata"] = graphrag_metadata
                state["workflow_step"] = "graph_retrieval"
                state["latency_ms"]["graph_retrieval"] = latency

                logger.info(
                    f"GraphRAG retrieval complete: {len(unique_nodes)} nodes, "
                    f"{len(all_edges)} edges, {len(communities)} communities "
                    f"(mode={graphrag_result.search_mode}, latency={latency:.1f}ms)"
                )

                return state

            except Exception as e:
                logger.warning(f"GraphRAG search failed: {e}. Falling back to standard retrieval.")
                # Fall through to standard retrieval

        # Fallback: Standard subgraph retrieval (original implementation)
        logger.info(f"Using standard subgraph retrieval for {len(entities)} entities")

        all_nodes = []
        all_edges = []
        retrieved_chunks = []

        for entity in entities[:3]:  # Limit to top 3 entities
            entity_name = entity.get("name", "")
            if not entity_name:
                continue

            # Get subgraph around this entity
            subgraph = self.neo4j.query_subgraph(entity_name, depth=2)

            all_nodes.extend(subgraph.get("nodes", []))
            all_edges.extend(subgraph.get("edges", []))

            # Create text chunks from subgraph
            for node in subgraph.get("nodes", []):
                chunk_text = self._node_to_text(node)
                retrieved_chunks.append({
                    "text": chunk_text,
                    "source": "graph",
                    "entity": entity_name,
                    "node_data": node,
                })

        # Deduplicate nodes
        unique_nodes = self._deduplicate_nodes(all_nodes)

        latency = (time.time() - start_time) * 1000

        state["subgraph"] = {
            "nodes": unique_nodes,
            "edges": all_edges,
            "node_count": len(unique_nodes),
            "edge_count": len(all_edges),
        }
        state["retrieved_chunks"] = retrieved_chunks
        state["workflow_step"] = "graph_retrieval"
        state["latency_ms"]["graph_retrieval"] = latency

        logger.info(
            f"Retrieved subgraph: {len(unique_nodes)} nodes, "
            f"{len(all_edges)} edges, {len(retrieved_chunks)} chunks "
            f"(latency={latency:.1f}ms)"
        )

        return state
    
    def compression_node(self, state: WorkflowState) -> WorkflowState:
        """
        Passthrough node (compression removed).
        Kept for backward compatibility with existing workflow definitions.
        """
        # Simply pass through without compression
        chunks = state.get("retrieved_chunks", [])
        state["compressed_chunks"] = chunks
        state["compression_stats"] = {}
        state["workflow_step"] = "compression"
        return state
    
    def generation_node(self, state: WorkflowState) -> WorkflowState:
        """
        Generate response using LLM

        Args:
            state: Current workflow state

        Returns:
            Updated state with generated response
        """
        start_time = time.time()

        query = state.get("query", "")
        # Use retrieved chunks directly (compression bypassed)
        chunks = state.get("retrieved_chunks", [])

        # Build context from retrieved chunks
        context = self._build_context(chunks)

        logger.info(
            f"Generating response with {len(chunks)} chunks (uncompressed), "
            f"context length={len(context)}"
        )
        
        try:
            # Call Claude API
            result = self.claude.generate_response(
                query=query,
                context=context,
            )
            
            latency = (time.time() - start_time) * 1000
            
            state["context"] = context
            state["response"] = result["response"]
            state["citations"] = result["citations"]
            state["workflow_step"] = "generation"
            state["latency_ms"]["generation"] = latency
            state["latency_ms"]["total_tokens"] = result["total_tokens"]
            
            logger.info(
                f"Generated response: {len(result['response'])} chars, "
                f"{len(result['citations'])} citations "
                f"(latency={latency:.1f}ms, tokens={result['total_tokens']})"
            )
            
        except Exception as e:
            logger.error(f"Error in generation node: {e}")
            state["error"] = str(e)
            state["response"] = "Error generating response. Please try again."
            state["citations"] = []
        
        return state
    
    def _node_to_text(self, node: Dict[str, Any]) -> str:
        """Convert graph node to text representation with bilingual support"""
        name = node.get("name", "Unknown")
        entity_type = node.get("type", "Entity")
        confidence = node.get("confidence", 0.5)

        # Get bilingual names if available
        name_en = node.get("name_en", "")
        name_ar = node.get("name_ar", "")
        description = node.get("description", "")

        # Build bilingual text representation
        text_parts = [f"{name} ({entity_type})"]

        # Add translations if different from original name
        if name_en and name_en != name:
            text_parts.append(f"English: {name_en}")
        if name_ar and name_ar != name:
            text_parts.append(f"Arabic: {name_ar}")

        # Add description if available
        if description:
            text_parts.append(f"Description: {description}")

        # Add confidence if high
        if confidence > 0.7:
            text_parts.append(f"confidence: {confidence:.2f}")

        # Join with separators
        text = " | ".join(text_parts)

        return text
    
    def _deduplicate_nodes(self, nodes: list) -> list:
        """Remove duplicate nodes based on name"""
        seen = set()
        unique = []
        
        for node in nodes:
            name = node.get("name", "")
            if name and name not in seen:
                seen.add(name)
                unique.append(node)
        
        return unique
    
    def _build_context(self, compressed_chunks: list) -> str:
        """Build context string from compressed chunks"""
        if not compressed_chunks:
            return "No relevant information found."

        context_parts = []

        for i, chunk in enumerate(compressed_chunks, 1):
            compressed_text = chunk.get("compressed_text", chunk.get("text", ""))
            entity = chunk.get("entity", "")

            # Format as numbered section
            section = f"[{i}] {compressed_text}"
            if entity:
                section += f" (source: {entity})"

            context_parts.append(section)

        return "\n\n".join(context_parts)

    def _get_community_data(self) -> list:
        """
        Retrieve community data from Neo4j for visualization

        Returns:
            List of community dictionaries with metadata
        """
        try:
            query = """
            MATCH (c:Community)
            OPTIONAL MATCH (c)<-[:BELONGS_TO]-(e:Entity)
            WITH c, COUNT(e) as member_count, COLLECT(e.name)[0..5] as sample_members
            RETURN
                c.id as id,
                c.level as level,
                c.summary as summary,
                c.themes as themes,
                member_count,
                sample_members
            ORDER BY c.level, c.id
            LIMIT 50
            """

            results = self.neo4j.execute_query(query)

            communities = []
            for record in results:
                communities.append({
                    "id": record.get("id"),
                    "level": record.get("level", 0),
                    "summary": record.get("summary", ""),
                    "themes": record.get("themes", []),
                    "member_count": record.get("member_count", 0),
                    "sample_members": record.get("sample_members", []),
                })

            return communities

        except Exception as e:
            logger.error(f"Error retrieving community data: {e}")
            return []
