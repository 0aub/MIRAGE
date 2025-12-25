import { useState, useRef, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger, SheetDescription } from "@/components/ui/sheet";
import { Label } from "@/components/ui/label";
import {
  Send, Bot, User, ChevronDown, Loader2, Network as NetworkIcon, GitBranch,
  Layers, Info, Settings2, Clock, Database, Minimize2,
  ArrowRight, Target, PanelRightClose, PanelRight,
  Activity, AlertCircle
} from "lucide-react";
import { Network } from "vis-network";
import { chatApi, documentsApi, RetrievalMode, ChunkInfo, CompressionStats } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

// Helper to fix double-encoding issues (Windows-1252/Latin1 interpreted as UTF-8)
import { fixEncoding } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  timestamp: Date;
  retrievedNodes?: number;
  responseTime?: number;
  retrievalMode?: string;
  retrievalTimeMs?: number;
  generationTimeMs?: number;
  entitiesFound?: string[];
  compressionStats?: {
    compression_ratio: number;
    speedup_factor: number;
    chunks_compressed: number;
    original_tokens?: number;
    compressed_tokens?: number;
  };
  modelUsed?: string;
  chunks?: ChunkInfo[];
}

// Mode descriptions
const MODE_INFO: Record<RetrievalMode, { label: string; description: string; color: string }> = {
  vector: { label: "Vector", description: "Vector similarity search with keyword fallback", color: "bg-blue-500" },
  local: { label: "Local", description: "Entity-focused graph traversal with semantic entity matching", color: "bg-green-500" },
  global: { label: "Global", description: "Map-reduce over community summaries (GraphRAG)", color: "bg-purple-500" },
  hybrid: { label: "Hybrid", description: "Fusion of vector + local + global modes", color: "bg-orange-500" },
  semantic: { label: "Semantic", description: "Cross-encoder re-ranking for precision", color: "bg-pink-500" },
  mix: { label: "Mix", description: "All modes combined with RRF fusion", color: "bg-red-500" },
  global_search: { label: "Global Search", description: "Community-based thematic search", color: "bg-teal-500" },
  drift: { label: "DRIFT", description: "Dynamic reasoning with iterative follow-up questions", color: "bg-cyan-500" },
};

// Config option explanations
const CONFIG_EXPLANATIONS = {
  hyde: {
    title: "HyDE (Hypothetical Document Embeddings)",
    description: "Generates a hypothetical answer to your query, then searches using that answer's embedding. Improves retrieval for complex questions by creating a 'bridge' between query and document space."
  },
  ppr: {
    title: "PPR (Personalized PageRank)",
    description: "Uses graph algorithms to find important nodes connected to query entities. Ranks chunks by their 'importance' in the knowledge graph relative to the query context."
  },
  community: {
    title: "Community Selection",
    description: "First identifies relevant community clusters in the graph, then retrieves chunks from those communities. Useful for focused retrieval when you know which topic areas are relevant."
  }
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "مرحباً! أنا مساعد MIRAGE. كيف يمكنني مساعدتك اليوم؟\n\nHello! I'm MIRAGE assistant. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isRTL, setIsRTL] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();
  const [contextGraph, setContextGraph] = useState<{ nodes: any[]; edges: any[] }>({
    nodes: [],
    edges: [],
  });
  const [retrievalMetadata, setRetrievalMetadata] = useState<any>(null);
  const [retrievedChunks, setRetrievedChunks] = useState<ChunkInfo[]>([]);

  // Mode and options state
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("local");
  const [topK, setTopK] = useState<number>(5);

  // V5 Advanced options
  const [useHyde, setUseHyde] = useState(false);
  const [usePpr, setUsePpr] = useState(false);
  const [useCommunitySelection, setUseCommunitySelection] = useState(false);

  // UI State
  const [showPanel, setShowPanel] = useState(true);
  const [showMetrics, setShowMetrics] = useState(false);

  // Last response metadata
  const [lastResponseMeta, setLastResponseMeta] = useState<{
    retrievalMode: string;
    retrievalTimeMs: number;
    generationTimeMs: number;
    totalTimeMs: number;
    chunksCount: number;
    entitiesFound?: string[];
    compressionStats?: CompressionStats;
    retrievalStats?: {
      total_chunks: number;
      vector_chunks: number;
      graph_1hop_chunks: number;
      graph_2hop_chunks: number;
      entities_used?: string[];
    };
    modelUsed?: string;
  } | null>(null);

  const networkRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Update graph visualization when context changes
  useEffect(() => {
    if (!networkRef.current || contextGraph.nodes.length === 0) return;

    const data = {
      nodes: contextGraph.nodes,
      edges: contextGraph.edges,
    };

    const options: any = {
      nodes: {
        shape: "dot",
        size: 20,
        font: {
          size: 11,
          face: "Noto Naskh Arabic, Inter",
          color: "#fff",
        },
        borderWidth: 2,
      },
      edges: {
        width: 1,
        smooth: true,
        arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      },
      groups: {
        chunk: {
          color: { background: "hsl(210, 100%, 55%)", border: "hsl(210, 100%, 40%)" },
        },
        entity: {
          color: { background: "hsl(140, 65%, 45%)", border: "hsl(140, 65%, 35%)" },
        },
        relationship: {
          color: { background: "hsl(25, 95%, 55%)", border: "hsl(25, 95%, 40%)" },
        },
      },
      physics: { enabled: true, stabilization: { iterations: 50 } },
      interaction: {
        dragNodes: true,
        dragView: true,
        zoomView: true,
      },
    };

    const network = new Network(networkRef.current, data, options);
    network.fit();

    return () => {
      network.destroy();
    };
  }, [contextGraph]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await chatApi.ask({
        message: input,
        retrieval_mode: retrievalMode,
        top_k: topK,
        conversation_id: conversationId,
        use_hyde: useHyde,
        use_ppr: usePpr,
        use_community_selection: useCommunitySelection,
      });

      // Store chunks for detailed view
      setRetrievedChunks(response.chunks || []);

      // Build graph from chunks - extract entities from chunk data
      const nodes: any[] = [];
      const edges: any[] = [];
      const seenNodes = new Set<string>();
      const entityNodes = new Map<string, number>();

      // Extract entities from chunks (via_entity field)
      const chunkEntities = new Set<string>();
      response.chunks?.forEach((chunk) => {
        if (chunk.via_entity) {
          chunkEntities.add(chunk.via_entity);
        }
      });

      // Add entity nodes from chunks
      chunkEntities.forEach((entity) => {
        if (!seenNodes.has(`entity_${entity}`)) {
          const entityId = nodes.length + 1;
          nodes.push({
            id: entityId,
            label: entity,
            group: "entity",
            title: `Entity: ${entity}`,
          });
          seenNodes.add(`entity_${entity}`);
          entityNodes.set(entity, entityId);
        }
      });

      // Add chunk nodes and connect to entities
      response.chunks?.forEach((chunk, idx) => {
        const chunkId = nodes.length + 1;
        const sourceType = chunk.source_type || (chunk.via_entity ? 'graph' : 'vector');
        nodes.push({
          id: chunkId,
          label: `Chunk ${idx + 1}`,
          group: "chunk",
          title: `[${sourceType}] ${chunk.text.substring(0, 200)}...`,
          raw: chunk,
        });

        // Connect chunk to entity if available
        if (chunk.via_entity && entityNodes.has(chunk.via_entity)) {
          const entityId = entityNodes.get(chunk.via_entity)!;
          edges.push({
            from: entityId,
            to: chunkId,
            label: chunk.hop_distance ? `${chunk.hop_distance}-hop` : "linked",
            raw: { relationship: chunk.via_relationship || "contains", hop: chunk.hop_distance },
          });
        }
      });

      setContextGraph({ nodes, edges });

      // Set metadata with full details
      setRetrievalMetadata({
        search_mode: response.retrieval_mode,
        query_entities: response.metadata?.query_entities,
        modes_used: response.metadata?.modes_used,
        fusion_method: response.metadata?.fusion_method,
        hyde_used: response.metadata?.hyde_used,
        ppr_used: response.metadata?.ppr_used,
        community_selection_used: response.metadata?.community_selection_used,
        compression_stats: response.metadata?.compression_stats,
        model_used: response.metadata?.model_used,
        entities_found: response.entities_found,
        entity_count: response.entity_count,
        enhanced_query: response.enhanced_query,
      });

      // Set last response metadata for performance display
      setLastResponseMeta({
        retrievalMode: response.retrieval_mode,
        retrievalTimeMs: response.retrieval_time_ms,
        generationTimeMs: response.generation_time_ms,
        totalTimeMs: response.total_time_ms,
        chunksCount: response.chunks?.length || 0,
        entitiesFound: response.entities_found || response.retrieval_stats?.entities_used,
        retrievalStats: response.retrieval_stats,
        compressionStats: response.compression_stats,
        modelUsed: response.metadata?.model_used,
      });

      // Create assistant message with full metadata
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer || "No response received",
        sources: response.chunks?.map((c) => c.document_id) || [],
        timestamp: new Date(),
        retrievedNodes: response.chunks?.length || 0,
        responseTime: response.total_time_ms,
        retrievalMode: response.retrieval_mode,
        retrievalTimeMs: response.retrieval_time_ms,
        generationTimeMs: response.generation_time_ms,
        entitiesFound: response.entities_found,
        compressionStats: response.metadata?.compression_stats,
        modelUsed: response.metadata?.model_used,
        chunks: response.chunks,
      };

      setMessages((prev) => [...prev, aiMessage]);

    } catch (error: any) {
      console.error('Error sending message:', error);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Error: ${error.message || "Failed to get response from server. Please try again."}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);

      toast({
        title: "Error",
        description: "Failed to send message to server",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Format time nicely
  const formatTime = (ms: number) => {
    if (ms >= 1000) {
      return `${(ms / 1000).toFixed(2)}s`;
    }
    return `${Math.round(ms)}ms`;
  };

  // Get entities from chunks (actual entities used in retrieval)
  const getChunkEntities = () => {
    const entities = new Set<string>();
    retrievedChunks.forEach(chunk => {
      if (chunk.via_entity) entities.add(chunk.via_entity);
    });
    return Array.from(entities);
  };

  // Calculate timing percentages for visual bar
  const getTimingPercentages = () => {
    if (!lastResponseMeta) return { retrieval: 50, generation: 50 };
    const total = lastResponseMeta.retrievalTimeMs + lastResponseMeta.generationTimeMs;
    if (total === 0) return { retrieval: 50, generation: 50 };
    return {
      retrieval: (lastResponseMeta.retrievalTimeMs / total) * 100,
      generation: (lastResponseMeta.generationTimeMs / total) * 100,
    };
  };

  const timingPcts = getTimingPercentages();
  const chunkEntities = getChunkEntities();

  const handleModeSelect = (mode: RetrievalMode) => {
    setRetrievalMode(mode);
    toast({
      title: "Mode Changed",
      description: `Switched to ${mode} mode`,
    });
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] animate-fade-in relative overflow-hidden">
      {/* Main Chat Area */}
      <div className={`flex flex-col flex-1 transition-all duration-300`}>
        
        {/* Chat Header */}
        <div className="flex items-center justify-between p-4 border-b bg-background/95 backdrop-blur z-10">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold">Chat</h1>
            <div className="flex gap-2">
                <Badge variant={retrievalMode === 'vector' ? 'default' : 'outline'}
                       className="cursor-pointer hover:bg-primary/90"
                       onClick={() => handleModeSelect('vector')}>
                    Vector
                </Badge>
                <Badge variant={retrievalMode === 'local' ? 'default' : 'outline'} 
                       className="cursor-pointer hover:bg-primary/90"
                       onClick={() => handleModeSelect('local')}>
                    Local
                </Badge>
                <Badge variant={retrievalMode === 'global' ? 'default' : 'outline'}
                       className="cursor-pointer hover:bg-primary/90"
                       onClick={() => handleModeSelect('global')}>
                    Global
                </Badge>
                <Badge variant={retrievalMode === 'hybrid' ? 'default' : 'outline'}
                       className="cursor-pointer hover:bg-primary/90"
                       onClick={() => handleModeSelect('hybrid')}>
                    Hybrid
                </Badge>
            </div>
          </div>
          
          <div className="flex gap-2">
            <Sheet>
                <SheetTrigger asChild>
                    <Button variant="outline" size="sm" className="gap-2">
                        <Settings2 className="w-4 h-4" />
                        Advanced Setup
                    </Button>
                </SheetTrigger>
                <SheetContent>
                    <SheetHeader>
                        <SheetTitle>Retrieval Configuration</SheetTitle>
                        <SheetDescription>
                            Fine-tune how the system retrieves and processes information.
                        </SheetDescription>
                    </SheetHeader>
                    
                    <div className="space-y-6 py-6">
                        {/* HyDE Toggle */}
                        <div className="flex items-center justify-between">
                            <div className="space-y-0.5">
                                <Label className="text-base flex items-center gap-2">
                                    HyDE
                                    <TooltipProvider>
                                    <Tooltip>
                                        <TooltipTrigger>
                                            <Info className="w-4 h-4 text-muted-foreground cursor-help" />
                                        </TooltipTrigger>
                                        <TooltipContent className="max-w-[300px]">
                                            <p><strong>Hypothetical Document Embeddings</strong></p>
                                            <p className="text-xs text-muted-foreground mt-1">
                                                Generates a hypothetical answer to improve retrieval by matching semantic meaning rather than just keywords. Best for complex questions.
                                            </p>
                                        </TooltipContent>
                                    </Tooltip>
                                    </TooltipProvider>
                                </Label>
                            </div>
                            <Switch checked={useHyde} onCheckedChange={setUseHyde} />
                        </div>

                        {/* PPR Toggle */}
                        <div className="flex items-center justify-between">
                            <div className="space-y-0.5">
                                <Label className="text-base flex items-center gap-2">
                                    PPR Re-ranking
                                    <TooltipProvider>
                                    <Tooltip>
                                        <TooltipTrigger>
                                            <Info className="w-4 h-4 text-muted-foreground cursor-help" />
                                        </TooltipTrigger>
                                        <TooltipContent className="max-w-[300px]">
                                            <p><strong>Personalized PageRank</strong></p>
                                            <p className="text-xs text-muted-foreground mt-1">
                                                Prioritizes nodes in the graph that are structurally relevant to the retrieved entities. Improves precision in graph mode.
                                            </p>
                                        </TooltipContent>
                                    </Tooltip>
                                    </TooltipProvider>
                                </Label>
                            </div>
                            <Switch checked={usePpr} onCheckedChange={setUsePpr} />
                        </div>

                        {/* Community Selection */}
                        <div className="flex items-center justify-between">
                            <div className="space-y-0.5">
                                <Label className="text-base flex items-center gap-2">
                                    Community Guidance
                                    <TooltipProvider>
                                    <Tooltip>
                                        <TooltipTrigger>
                                            <Info className="w-4 h-4 text-muted-foreground cursor-help" />
                                        </TooltipTrigger>
                                        <TooltipContent className="max-w-[300px]">
                                            <p><strong>Community Detection</strong></p>
                                            <p className="text-xs text-muted-foreground mt-1">
                                                Uses detected communities to guide the search towards relevant clusters of information. Helpful for broad queries.
                                            </p>
                                        </TooltipContent>
                                    </Tooltip>
                                    </TooltipProvider>
                                </Label>
                            </div>
                            <Switch checked={useCommunitySelection} onCheckedChange={setUseCommunitySelection} />
                        </div>

                        {/* Top K */}
                        <div className="flex items-center justify-between">
                            <div className="space-y-0.5">
                                <Label className="text-base flex items-center gap-2">
                                    Top K Retrieved
                                    <TooltipProvider>
                                    <Tooltip>
                                        <TooltipTrigger>
                                            <Info className="w-4 h-4 text-muted-foreground cursor-help" />
                                        </TooltipTrigger>
                                        <TooltipContent className="max-w-[300px]">
                                            <p><strong>Retrieval Count</strong></p>
                                            <p className="text-xs text-muted-foreground mt-1">
                                                Number of chunks to retrieve from the database. Higher means more context but slower processing.
                                            </p>
                                        </TooltipContent>
                                    </Tooltip>
                                    </TooltipProvider>
                                </Label>
                            </div>
                            <div className="w-[80px]">
                                <Input 
                                    type="number" 
                                    min={1} 
                                    max={50} 
                                    value={topK} 
                                    onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                                    className="h-8 text-right"
                                />
                            </div>
                        </div>
                        
                    </div>
                </SheetContent>
            </Sheet>

            <Button
              variant={showPanel ? "default" : "outline"}
              size="icon"
              onClick={() => setShowPanel(!showPanel)}
              title="Toggle Graph Panel"
            >
              <NetworkIcon className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth" ref={scrollRef}>
          {/* Expandable Metrics Panel */}
          <Collapsible open={showMetrics} onOpenChange={setShowMetrics}>
            <CollapsibleContent>
              {lastResponseMeta && (
                <div className="mb-3 p-3 bg-secondary/20 rounded-lg space-y-2">
                  {/* Timing Bar */}
                  <div className="flex items-center gap-2 text-xs">
                    <div className="flex-1 flex h-2 rounded-full overflow-hidden">
                      <div className="bg-blue-500" style={{ width: `${timingPcts.retrieval}%` }} />
                      <div className="bg-purple-500" style={{ width: `${timingPcts.generation}%` }} />
                    </div>
                    <span className="text-muted-foreground w-36 text-right">
                      R: {formatTime(lastResponseMeta.retrievalTimeMs)} / G: {formatTime(lastResponseMeta.generationTimeMs)}
                    </span>
                  </div>

                  {/* Stats Row */}
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <Badge variant="secondary" className={`${MODE_INFO[lastResponseMeta.retrievalMode as RetrievalMode]?.color || 'bg-gray-500'} text-white`}>
                      {lastResponseMeta.retrievalMode}
                    </Badge>
                    {chunkEntities.length > 0 && (
                      <span className="flex items-center gap-1">
                        <Target className="w-3 h-3 text-green-500" />
                        {chunkEntities.slice(0, 3).join(", ")}
                        {chunkEntities.length > 3 && ` +${chunkEntities.length - 3}`}
                      </span>
                    )}
                    {lastResponseMeta.retrievalStats && (
                      <span className="flex items-center gap-1 text-muted-foreground">
                        <Database className="w-3 h-3" />
                        Vec:{lastResponseMeta.retrievalStats.vector_chunks} 1-hop:{lastResponseMeta.retrievalStats.graph_1hop_chunks} 2-hop:{lastResponseMeta.retrievalStats.graph_2hop_chunks}
                      </span>
                    )}
                    {lastResponseMeta.compressionStats?.enabled && (
                      <span className="flex items-center gap-1 text-green-600">
                        <Minimize2 className="w-3 h-3" />
                        {lastResponseMeta.compressionStats.original_length.toLocaleString()} → {lastResponseMeta.compressionStats.compressed_length.toLocaleString()}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </CollapsibleContent>
          </Collapsible>

          {/* Messages Area - Takes all available space */}
          <ScrollArea className="flex-1 pr-4 mb-3" ref={scrollRef}>
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  dir={isRTL ? "rtl" : "ltr"}
                >
                  {message.role === "assistant" && (
                    <div className="w-8 h-8 rounded-full gradient-primary flex items-center justify-center flex-shrink-0">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                  )}

                  <div className="flex flex-col gap-1.5 max-w-[80%]">
                    <div 
                    className={`p-3 rounded-lg ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground" 
                        : "bg-secondary/50 border"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap font-sans" dir="auto">{fixEncoding(message.content)}</p>
                    </div>
                    {/* Source citations */}
                    {message.role === "assistant" && message.chunks && message.chunks.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
                        <Badge variant="outline" className="h-5 text-xs py-0">
                          {message.retrievalMode}
                        </Badge>
                        <span>{formatTime(message.responseTime)}</span>
                        {message.compressionStats && (
                          <span className="text-green-600">{message.compressionStats.compression_ratio.toFixed(1)}x</span>
                        )}
                      </div>
                    )}

                    {/* Collapsible sources */}
                    {message.sources && message.sources.length > 0 && (
                      <details className="text-xs">
                        <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                          Sources ({[...new Set(message.sources)].length})
                        </summary>
                        <div className="mt-1 space-y-0.5 pl-2 text-muted-foreground">
                          {[...new Set(message.sources)].slice(0, 5).map((source, idx) => (
                            <div key={idx}>• {source}</div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>

                  {message.role === "user" && (
                    <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
                      <User className="w-5 h-5" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </ScrollArea>

          {/* Input Area - Fixed at bottom */}
          <div className="space-y-2 pt-3 border-t">
            {/* Input Row */}
            <div className="flex gap-2" dir={isRTL ? "rtl" : "ltr"}>
              <Input
                placeholder={isRTL ? "اكتب رسالتك هنا..." : "Type your message..."}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                className="flex-1 h-11"
                disabled={isLoading}
              />
              <Button onClick={handleSend} size="icon" className="h-11 w-11" disabled={isLoading}>
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </Button>
            </div>
          </div>
      </div>
      </div>

    
        {/* Explainability Side Panel - Collapsible */}
        {showPanel && (
          <Card className="w-[420px] flex-shrink-0 p-4 flex flex-col">
            <h2 className="text-lg font-semibold mb-3">Explainability</h2>

            {contextGraph.nodes.length > 0 || retrievalMetadata || retrievedChunks.length > 0 ? (
              <Tabs defaultValue="chunks" className="flex-1 flex flex-col">
                <TabsList className="grid w-full grid-cols-4 h-8">
                  <TabsTrigger value="chunks" className="text-xs py-1">
                    <Layers className="w-3 h-3 mr-1" />
                    Chunks
                  </TabsTrigger>
                  <TabsTrigger value="graph" className="text-xs py-1">
                    <NetworkIcon className="w-3 h-3 mr-1" />
                    Graph
                  </TabsTrigger>
                  <TabsTrigger value="entities" className="text-xs py-1">
                    <Target className="w-3 h-3 mr-1" />
                    Entities
                  </TabsTrigger>
                  <TabsTrigger value="meta" className="text-xs py-1">
                    <Info className="w-3 h-3 mr-1" />
                    Meta
                  </TabsTrigger>
                </TabsList>

                {/* Chunks Tab */}
                <TabsContent value="chunks" className="flex-1 mt-3">
                  <ScrollArea className="h-[calc(100vh-20rem)]">
                    <div className="space-y-2 pr-2">
                      {retrievedChunks.length === 0 ? (
                        <p className="text-sm text-muted-foreground text-center py-8">No chunks retrieved</p>
                      ) : (
                        retrievedChunks.map((chunk, idx) => {
                          const sourceType = chunk.source_type || (chunk.via_entity ? 'graph' : 'vector');
                          const isGraph = sourceType.includes('graph') || chunk.via_entity;

                          return (
                            <Card key={idx} className="p-2.5">
                              <div className="space-y-1.5">
                                {/* Header */}
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center gap-1.5">
                                    <span className="text-xs font-bold">#{idx + 1}</span>
                                    <Badge className={`text-xs h-5 ${
                                      sourceType === 'vector' ? 'bg-blue-500' :
                                      sourceType === 'graph_1hop' || chunk.hop_distance === 1 ? 'bg-green-500' :
                                      sourceType === 'graph_2hop' || chunk.hop_distance === 2 ? 'bg-purple-500' :
                                      isGraph ? 'bg-green-500' : 'bg-blue-500'
                                    }`}>
                                      {sourceType === 'graph_1hop' || chunk.hop_distance === 1 ? '1-hop' :
                                       sourceType === 'graph_2hop' || chunk.hop_distance === 2 ? '2-hop' :
                                       sourceType}
                                    </Badge>
                                  </div>
                                  <span className="text-xs text-muted-foreground">
                                    {chunk.score.toFixed(3)}
                                  </span>
                                </div>

                                {/* Entity info - CLEAR display */}
                                {chunk.via_entity && (
                                  <div className="flex items-center gap-1.5 text-xs bg-green-500/10 px-2 py-1 rounded">
                                    <Target className="w-3 h-3 text-green-600" />
                                    <span className="font-medium text-green-700 dark:text-green-400">
                                      {chunk.via_entity}
                                    </span>
                                    {chunk.via_relationship && (
                                      <>
                                        <ArrowRight className="w-3 h-3 text-muted-foreground" />
                                        <span className="text-muted-foreground">{chunk.via_relationship}</span>
                                      </>
                                    )}
                                  </div>
                                )}

                                {/* Text */}
                                <p className="text-xs text-muted-foreground line-clamp-4">
                                  {fixEncoding(chunk.text)}
                                </p>
                              </div>
                            </Card>
                          );
                        })
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>

                {/* Graph Tab */}
                <TabsContent value="graph" className="flex-1 mt-3">
                  {contextGraph.nodes.length > 0 ? (
                    <>
                      <div
                        ref={networkRef}
                        className="w-full h-[calc(100vh-24rem)] rounded-lg bg-secondary/20"
                        style={{ border: "1px solid hsl(var(--border))" }}
                      />
                      <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <div className="w-3 h-3 rounded-full bg-green-500" />
                          Entities ({contextGraph.nodes.filter(n => n.group === 'entity').length})
                        </span>
                        <span className="flex items-center gap-1">
                          <div className="w-3 h-3 rounded-full bg-blue-500" />
                          Chunks ({contextGraph.nodes.filter(n => n.group === 'chunk').length})
                        </span>
                      </div>
                    </>
                  ) : (
                    <div className="flex-1 flex items-center justify-center h-[calc(100vh-24rem)]">
                      <div className="text-center text-muted-foreground">
                        <NetworkIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                        <p className="text-sm">No graph data</p>
                        <p className="text-xs">Use LOCAL or GRAPH modes to see entity connections</p>
                      </div>
                    </div>
                  )}
                </TabsContent>

                {/* Entities Tab - Shows CHUNK entities */}
                <TabsContent value="entities" className="flex-1 mt-3">
                  <ScrollArea className="h-[calc(100vh-20rem)]">
                    <div className="space-y-3 pr-2">
                      {/* Chunk entities - the actual entities used */}
                      <div>
                        <div className="text-xs font-medium mb-2 flex items-center gap-2">
                          <Target className="w-3 h-3" />
                          Entities from Retrieved Chunks
                        </div>
                        {chunkEntities.length > 0 ? (
                          <div className="flex flex-wrap gap-1.5">
                            {chunkEntities.map((entity, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs bg-green-500/10">
                                {entity}
                              </Badge>
                            ))}
                          </div>
                        ) : (
                          <p className="text-xs text-muted-foreground">No entities in retrieved chunks (try LOCAL or GRAPH mode)</p>
                        )}
                      </div>

                      {/* Query entities - what was detected in query */}
                      {retrievalMetadata?.entities_found && retrievalMetadata.entities_found.length > 0 && (
                        <div className="mt-4 pt-4 border-t">
                          <div className="text-xs font-medium mb-2 flex items-center gap-2">
                            <Info className="w-3 h-3" />
                            Entities Detected in Query
                          </div>
                          <div className="flex flex-wrap gap-1.5">
                            {retrievalMetadata.entities_found.map((entity: string, idx: number) => (
                              <Badge key={idx} variant="outline" className="text-xs">
                                {entity}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Relationships */}
                      {contextGraph.edges.filter(e => e.label !== 'contains' && e.label !== 'linked').length > 0 && (
                        <div className="mt-4 pt-4 border-t">
                          <div className="text-xs font-medium mb-2">Relationships</div>
                          <div className="space-y-1">
                            {contextGraph.edges
                              .filter(e => e.label !== 'contains' && e.label !== 'linked')
                              .slice(0, 8)
                              .map((edge, idx) => {
                                const from = contextGraph.nodes.find(n => n.id === edge.from);
                                const to = contextGraph.nodes.find(n => n.id === edge.to);
                                if (!from || !to) return null;
                                return (
                                  <div key={idx} className="flex items-center gap-1.5 text-xs bg-secondary/30 rounded p-1.5">
                                    <span>{from.label}</span>
                                    <ArrowRight className="w-3 h-3" />
                                    <Badge variant="outline" className="text-xs h-4 px-1">{edge.label}</Badge>
                                    <ArrowRight className="w-3 h-3" />
                                    <span>{to.label}</span>
                                  </div>
                                );
                              })}
                          </div>
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>

                {/* Meta Tab */}
                <TabsContent value="meta" className="flex-1 mt-3">
                  <ScrollArea className="h-[calc(100vh-20rem)]">
                    <div className="space-y-3 pr-2">
                      {retrievalMetadata || lastResponseMeta ? (
                        <>
                          {/* Summary */}
                          <div className="grid grid-cols-3 gap-2 text-xs">
                            <div className="bg-secondary/30 rounded p-2 text-center">
                              <div className="text-lg font-bold text-blue-500">{lastResponseMeta?.chunksCount || 0}</div>
                              <div className="text-muted-foreground">Chunks</div>
                            </div>
                            <div className="bg-secondary/30 rounded p-2 text-center">
                              <div className="text-lg font-bold text-green-500">{chunkEntities.length}</div>
                              <div className="text-muted-foreground">Entities</div>
                            </div>
                            <div className="bg-secondary/30 rounded p-2 text-center">
                              <div className="text-lg font-bold text-purple-500">{formatTime(lastResponseMeta?.totalTimeMs || 0)}</div>
                              <div className="text-muted-foreground">Time</div>
                            </div>
                          </div>

                          {/* Source breakdown - CLEAR labels */}
                          {lastResponseMeta?.retrievalStats && (
                            <div className="bg-secondary/30 rounded-lg p-2.5">
                              <div className="text-xs font-medium mb-2">Source Breakdown</div>
                              <div className="flex h-2 rounded-full overflow-hidden mb-2">
                                {lastResponseMeta.retrievalStats.vector_chunks > 0 && (
                                  <div
                                    className="bg-blue-500"
                                    style={{ width: `${(lastResponseMeta.retrievalStats.vector_chunks / lastResponseMeta.retrievalStats.total_chunks) * 100}%` }}
                                  />
                                )}
                                {lastResponseMeta.retrievalStats.graph_1hop_chunks > 0 && (
                                  <div
                                    className="bg-green-500"
                                    style={{ width: `${(lastResponseMeta.retrievalStats.graph_1hop_chunks / lastResponseMeta.retrievalStats.total_chunks) * 100}%` }}
                                  />
                                )}
                                {lastResponseMeta.retrievalStats.graph_2hop_chunks > 0 && (
                                  <div
                                    className="bg-purple-500"
                                    style={{ width: `${(lastResponseMeta.retrievalStats.graph_2hop_chunks / lastResponseMeta.retrievalStats.total_chunks) * 100}%` }}
                                  />
                                )}
                              </div>
                              <div className="grid grid-cols-3 gap-1 text-xs">
                                <div className="flex items-center gap-1">
                                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                                  <span>Vector: {lastResponseMeta.retrievalStats.vector_chunks}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                  <div className="w-2 h-2 rounded-full bg-green-500" />
                                  <span>1-hop: {lastResponseMeta.retrievalStats.graph_1hop_chunks}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                  <div className="w-2 h-2 rounded-full bg-purple-500" />
                                  <span>2-hop: {lastResponseMeta.retrievalStats.graph_2hop_chunks}</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Compression stats */}
                          {lastResponseMeta?.compressionStats?.enabled && (
                            <div className="bg-green-500/10 rounded-lg p-2.5">
                              <div className="text-xs font-medium mb-2 text-green-700 dark:text-green-400">
                                Context Compression
                              </div>
                              <div className="text-xs space-y-1">
                                <div className="flex justify-between">
                                  <span>Strategy:</span>
                                  <Badge variant="outline" className="text-xs h-5">{lastResponseMeta.compressionStats.strategy}</Badge>
                                </div>
                                <div className="flex justify-between">
                                  <span>Ratio:</span>
                                  <span className="text-green-600 font-bold">
                                    {(lastResponseMeta.compressionStats.compression_ratio * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span>Original:</span>
                                  <span>{lastResponseMeta.compressionStats.original_length.toLocaleString()} chars</span>
                                </div>
                                <div className="flex justify-between">
                                  <span>Compressed:</span>
                                  <span className="text-green-600">{lastResponseMeta.compressionStats.compressed_length.toLocaleString()} chars</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Config */}
                          <div className="bg-secondary/30 rounded-lg p-2.5">
                            <div className="text-xs font-medium mb-2">Configuration Used</div>
                            <div className="flex flex-wrap gap-1">
                              <Badge variant={retrievalMetadata?.hyde_used ? "default" : "outline"} className="text-xs h-5">
                                HyDE {retrievalMetadata?.hyde_used ? "ON" : "off"}
                              </Badge>
                              <Badge variant={retrievalMetadata?.ppr_used ? "default" : "outline"} className="text-xs h-5">
                                PPR {retrievalMetadata?.ppr_used ? "ON" : "off"}
                              </Badge>
                              <Badge variant={retrievalMetadata?.community_selection_used ? "default" : "outline"} className="text-xs h-5">
                                Community {retrievalMetadata?.community_selection_used ? "ON" : "off"}
                              </Badge>
                            </div>
                          </div>

                          {/* Fusion pipeline */}
                          {retrievalMetadata?.modes_used && retrievalMetadata.modes_used.length > 0 && (
                            <div className="bg-secondary/30 rounded-lg p-2.5">
                              <div className="text-xs font-medium mb-2">Fusion Pipeline</div>
                              <div className="flex flex-wrap items-center gap-1 text-xs">
                                {retrievalMetadata.modes_used.map((mode: string, idx: number) => (
                                  <span key={idx} className="flex items-center">
                                    <Badge variant="outline" className="text-xs h-5">{mode}</Badge>
                                    {idx < retrievalMetadata.modes_used.length - 1 && (
                                      <ArrowRight className="w-3 h-3 mx-0.5 text-muted-foreground" />
                                    )}
                                  </span>
                                ))}
                                {retrievalMetadata.fusion_method && (
                                  <>
                                    <ArrowRight className="w-3 h-3 mx-0.5 text-muted-foreground" />
                                    <Badge className="bg-orange-500 text-xs h-5">{retrievalMetadata.fusion_method}</Badge>
                                  </>
                                )}
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        <p className="text-sm text-muted-foreground text-center py-8">No metadata available</p>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>
              </Tabs>
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <div className="text-center p-4">
                  <Activity className="w-12 h-12 mx-auto text-muted-foreground mb-3" />
                  <p className="text-sm text-muted-foreground">
                    {isLoading ? "Retrieving..." : "Send a message to see details"}
                  </p>
                </div>
              </div>
            )}
          </Card>
        )}
      </div>
  );
}
