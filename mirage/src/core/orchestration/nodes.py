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
from ..embeddings import JinaEmbedder
from ..refrag import REFRAGCompressor
from .claude_client import ClaudeClient
from .workflow_state import WorkflowState


class WorkflowNodes:
    """Collection of nodes for the RAG workflow"""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        refrag_compressor: REFRAGCompressor,
        claude_client: ClaudeClient,
    ):
        """
        Initialize workflow nodes

        Args:
            neo4j_client: Neo4j client for graph retrieval
            refrag_compressor: REFRAG compressor
            claude_client: Claude API client
        """
        self.neo4j = neo4j_client
        self.compressor = refrag_compressor
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
        Retrieve relevant subgraph from Neo4j
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with retrieved subgraph
        """
        start_time = time.time()
        
        entities = state.get("entities_found", [])
        logger.info(f"Retrieving subgraphs for {len(entities)} entities")
        
        # Collect all relevant nodes and edges
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
        Apply REFRAG compression to retrieved chunks
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with compressed chunks
        """
        start_time = time.time()
        
        chunks = state.get("retrieved_chunks", [])
        query = state.get("query", "")
        
        if not chunks:
            logger.warning("No chunks to compress")
            state["compressed_chunks"] = []
            state["compression_stats"] = {}
            return state
        
        logger.info(f"Compressing {len(chunks)} chunks")
        
        # Apply REFRAG compression with query context
        compression_result = self.compressor.compress(
            chunks,
            query_context=query,
        )
        
        latency = (time.time() - start_time) * 1000
        
        state["compressed_chunks"] = compression_result["compressed_chunks"]
        state["compression_stats"] = {
            "original_length": compression_result["original_length"],
            "compressed_length": compression_result["compressed_length"],
            "compression_ratio": compression_result["compression_ratio"],
            "speedup_factor": compression_result["original_length"] / 
                            max(1, compression_result["compressed_length"]),
        }
        state["workflow_step"] = "compression"
        state["latency_ms"]["compression"] = latency
        
        logger.info(
            f"Compressed {len(chunks)} chunks: "
            f"ratio={compression_result['compression_ratio']:.2f} "
            f"(latency={latency:.1f}ms)"
        )
        
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
