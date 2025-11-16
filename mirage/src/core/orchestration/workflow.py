"""
RAG Workflow
LangGraph orchestration for the complete RAG pipeline
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from loguru import logger

from ..graph_builder import Neo4jClient
from ..refrag import REFRAGCompressor, CompressionPolicy, CompressionCache
from .workflow_state import WorkflowState
from .nodes import WorkflowNodes
from .claude_client import ClaudeClient


class RAGWorkflow:
    """
    LangGraph-based RAG workflow
    
    Flow:
    Query → Query Analysis → Graph Retrieval → Generation → Response
    (Compression disabled for now - focusing on GraphRAG)
    """
    
    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        refrag_compressor: Optional[REFRAGCompressor] = None,
        claude_client: Optional[ClaudeClient] = None,
    ):
        """
        Initialize RAG workflow
        
        Args:
            neo4j_client: Neo4j client (creates one if None)
            refrag_compressor: REFRAG compressor (creates one if None)
            claude_client: Claude client (creates one if None)
        """
        # Initialize components
        self.neo4j = neo4j_client or Neo4jClient()
        self.compressor = refrag_compressor or REFRAGCompressor(
            policy=CompressionPolicy(),
            cache=CompressionCache(),
        )
        self.claude = claude_client or ClaudeClient()
        
        # Initialize nodes
        self.nodes = WorkflowNodes(
            neo4j_client=self.neo4j,
            refrag_compressor=self.compressor,
            claude_client=self.claude,
        )
        
        # Build workflow graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        logger.info("Initialized RAGWorkflow with LangGraph")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow
        
        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("query_analysis", self.nodes.query_analysis_node)
        workflow.add_node("graph_retrieval", self.nodes.graph_retrieval_node)
        workflow.add_node("compression", self.nodes.compression_node)
        workflow.add_node("generation", self.nodes.generation_node)
        
        # Define edges (flow)
        workflow.set_entry_point("query_analysis")

        workflow.add_edge("query_analysis", "graph_retrieval")
        # Compression disabled - bypassing to focus on GraphRAG
        # workflow.add_edge("graph_retrieval", "compression")
        # workflow.add_edge("compression", "generation")
        workflow.add_edge("graph_retrieval", "generation")
        workflow.add_edge("generation", END)

        logger.info("Built LangGraph workflow with 3 active nodes (compression bypassed)")
        
        return workflow
    
    def invoke(
        self,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the RAG workflow
        
        Args:
            query: User query
            session_id: Optional session ID for tracking
            
        Returns:
            Workflow result with response and metadata
        """
        logger.info(f"Invoking RAG workflow for query: {query}")
        
        # Initialize state
        initial_state: WorkflowState = {
            "query": query,
            "session_id": session_id,
            "entities_found": [],
            "subgraph": {},
            "retrieved_chunks": [],
            "compressed_chunks": [],
            "compression_stats": {},
            "context": "",
            "response": "",
            "citations": [],
            "workflow_step": "start",
            "error": None,
            "latency_ms": {},
            "messages": [],
        }
        
        try:
            # Run workflow
            final_state = self.app.invoke(initial_state)
            
            # Calculate total latency
            total_latency = sum(final_state.get("latency_ms", {}).values())
            
            # Prepare graph visualization data
            subgraph = final_state.get("subgraph", {})
            graph_viz = {
                "nodes": [
                    {
                        "id": node.get("name", ""),
                        "label": node.get("name", ""),
                        "type": node.get("type", "Entity"),
                        "confidence": node.get("confidence", 0.5),
                    }
                    for node in subgraph.get("nodes", [])
                ],
                "edges": [
                    {
                        "source": edge.get("start", {}).get("name", ""),
                        "target": edge.get("end", {}).get("name", ""),
                        "relationship": edge.get("type", "RELATED_TO"),
                    }
                    for edge in subgraph.get("edges", [])
                ],
                "node_count": subgraph.get("node_count", 0),
                "edge_count": subgraph.get("edge_count", 0),
            }

            # Prepare compression comparison data
            retrieved_chunks = final_state.get("retrieved_chunks", [])
            compressed_chunks = final_state.get("compressed_chunks", [])

            compression_comparison = {
                "original_chunks": [
                    {
                        "text": chunk.get("text", ""),
                        "length": len(chunk.get("text", "")),
                    }
                    for chunk in retrieved_chunks[:5]  # Limit to 5 for UI
                ],
                "compressed_chunks": [
                    {
                        "text": chunk.get("compressed_text", chunk.get("text", "")),
                        "length": len(chunk.get("compressed_text", chunk.get("text", ""))),
                        "importance": chunk.get("importance_score", 0.5),
                        "compression_ratio": chunk.get("compression_ratio", 1.0),
                    }
                    for chunk in compressed_chunks[:5]  # Limit to 5 for UI
                ],
                "stats": final_state.get("compression_stats", {}),
            }

            # Build result
            result = {
                "query": query,
                "response": final_state.get("response", ""),
                "citations": final_state.get("citations", []),
                "metadata": {
                    "entities_found": len(final_state.get("entities_found", [])),
                    "nodes_retrieved": subgraph.get("node_count", 0),
                    "edges_retrieved": subgraph.get("edge_count", 0),
                    "chunks_compressed": len(compressed_chunks),
                    "compression_stats": final_state.get("compression_stats", {}),
                    "latency_ms": final_state.get("latency_ms", {}),
                    "total_latency_ms": total_latency,
                },
                "graph_visualization": graph_viz,
                "compression_comparison": compression_comparison,
                "success": final_state.get("error") is None,
                "error": final_state.get("error"),
            }
            
            logger.info(
                f"Workflow completed: {len(result['response'])} chars, "
                f"{len(result['citations'])} citations, "
                f"latency={total_latency:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG workflow: {e}")
            return {
                "query": query,
                "response": f"Error processing query: {str(e)}",
                "citations": [],
                "metadata": {},
                "success": False,
                "error": str(e),
            }
    
    async def astream(
        self,
        query: str,
        session_id: Optional[str] = None,
    ):
        """
        Stream the RAG workflow execution
        
        Args:
            query: User query
            session_id: Optional session ID
            
        Yields:
            State updates as workflow progresses
        """
        logger.info(f"Streaming RAG workflow for query: {query}")
        
        # Initialize state
        initial_state: WorkflowState = {
            "query": query,
            "session_id": session_id,
            "entities_found": [],
            "subgraph": {},
            "retrieved_chunks": [],
            "compressed_chunks": [],
            "compression_stats": {},
            "context": "",
            "response": "",
            "citations": [],
            "workflow_step": "start",
            "error": None,
            "latency_ms": {},
            "messages": [],
        }
        
        try:
            # Stream workflow execution
            async for state_update in self.app.astream(initial_state):
                yield state_update
                
        except Exception as e:
            logger.error(f"Error streaming RAG workflow: {e}")
            yield {
                "error": str(e),
                "workflow_step": "error",
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        return {
            "neo4j_connected": self.neo4j._connected,
            "compressor_stats": self.compressor.get_stats(),
            "workflow_nodes": 4,
        }
