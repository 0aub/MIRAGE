import { useEffect, useRef, useState } from "react";
import { Network } from "vis-network";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Search, Download, Filter, ZoomIn, ZoomOut, Maximize2, Loader2, Network as NetworkIcon, Layers, Info, Image as ImageIcon, FileJson } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { graphApi, graphragApi } from "@/lib/api";
import { fixEncoding } from "@/lib/utils";

interface Community {
  id: string;
  level: number;
  summary: string;
  themes: string[];
  member_count: number;
  sample_members: string[];
  parent_id?: string;
}

export default function GraphPage() {
  const networkRef = useRef<HTMLDivElement>(null);
  const [network, setNetwork] = useState<Network | null>(null);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [graphData, setGraphData] = useState<{ nodes: any[]; edges: any[] }>({ nodes: [], edges: [] });
  const [isDarkMode, setIsDarkMode] = useState(document.documentElement.classList.contains('dark'));
  const [communities, setCommunities] = useState<Community[]>([]);
  const [isLoadingCommunities, setIsLoadingCommunities] = useState(false);
  const [selectedLevel, setSelectedLevel] = useState<number | undefined>(undefined);
  const { toast } = useToast();

  // Listen for theme changes
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDarkMode(document.documentElement.classList.contains('dark'));
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => observer.disconnect();
  }, []);

  const [graphStats, setGraphStats] = useState<{ nodes: number; edges: number } | null>(null);
  
  // Load graph data from backend
  useEffect(() => {
    loadGraph();
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const stats = await graphApi.stats();
      setGraphStats({ nodes: stats.total_nodes, edges: stats.total_edges });
    } catch (error) {
      console.error('Error loading graph stats:', error);
    }
  };

  const loadGraph = async () => {
    try {
      setIsLoading(true);
      const response = await graphApi.visualizeFull(500);  // Cap at 500 for performance

      // Convert API response to vis-network format
      const nodes = response.nodes.map((node, idx) => ({
        id: idx + 1,
        label: node.label || node.id,
        group: node.type.toLowerCase(),
        title: `${node.type}: ${node.label}`,
        confidence: node.confidence,
      }));

      // Create node ID mapping
      const nodeIdMap = new Map(response.nodes.map((node, idx) => [node.id, idx + 1]));

      const edges = response.edges.map(edge => ({
        from: nodeIdMap.get(edge.source) || 1,
        to: nodeIdMap.get(edge.target) || 1,
        label: edge.relationship,
        title: `${edge.relationship} (${edge.confidence.toFixed(2)})`,
      }));

      setGraphData({ nodes, edges });
    } catch (error) {
      console.error('Error loading graph:', error);
      toast({
        title: "Error",
        description: "Failed to load graph data from server",
        variant: "destructive",
      });
      // Use empty data on error
      setGraphData({ nodes: [], edges: [] });
    } finally {
      setIsLoading(false);
    }
  };

  const loadCommunities = async (level?: number) => {
    try {
      setIsLoadingCommunities(true);
      const response = await graphragApi.getCommunities(level);
      setCommunities(response.communities);
    } catch (error) {
      console.error('Error loading communities:', error);
      toast({
        title: "Error",
        description: "Failed to load communities from server",
        variant: "destructive",
      });
      setCommunities([]);
    } finally {
      setIsLoadingCommunities(false);
    }
  };

  // Helper function to generate unique colors for node types
  const getColorForType = (typeName: string): { background: string; border: string } => {
    const colorMap: Record<string, { background: string; border: string }> = {
      document: { background: "hsl(160, 84%, 39%)", border: "hsl(160, 84%, 30%)" },
      person: { background: "hsl(217, 91%, 60%)", border: "hsl(217, 91%, 50%)" },
      organization: { background: "hsl(142, 71%, 45%)", border: "hsl(142, 71%, 35%)" },
      location: { background: "hsl(24, 95%, 53%)", border: "hsl(24, 95%, 43%)" },
      concept: { background: "hsl(173, 80%, 40%)", border: "hsl(173, 80%, 30%)" },
      event: { background: "hsl(280, 65%, 60%)", border: "hsl(280, 65%, 50%)" },
      technology: { background: "hsl(340, 82%, 52%)", border: "hsl(340, 82%, 42%)" },
      entity: { background: "hsl(200, 70%, 50%)", border: "hsl(200, 70%, 40%)" },
    };

    // Return predefined color if exists
    if (colorMap[typeName]) return colorMap[typeName];

    // Generate unique color from hash
    let hash = 0;
    for (let i = 0; i < typeName.length; i++) {
      hash = typeName.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash % 360);
    const saturation = 65 + (Math.abs(hash >> 8) % 25); // 65-90%
    const lightness = 45 + (Math.abs(hash >> 16) % 15); // 45-60%

    return {
      background: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
      border: `hsl(${hue}, ${saturation}%, ${lightness - 10}%)`,
    };
  };

  // Create network only when data changes
  useEffect(() => {
    if (!networkRef.current || graphData.nodes.length === 0) return;

    const data = graphData;

    // Generate groups dynamically for all node types
    const groups: Record<string, any> = {};
    Array.from(new Set(graphData.nodes.map(n => n.group))).forEach(type => {
      const colors = getColorForType(type);
      groups[type] = { color: colors };
    });

    const options: any = {
      nodes: {
        shape: "dot",
        size: 25,
        font: {
          size: 14,
          face: "Noto Naskh Arabic, Inter",
          color: isDarkMode ? "#f9fafb" : "#111827",
        },
        borderWidth: 2,
        shadow: true,
      },
      edges: {
        width: 2,
        shadow: true,
        smooth: {
          enabled: true,
          type: "continuous",
          roundness: 0.5,
        },
        font: {
          size: 12,
          face: "Inter",
          color: isDarkMode ? "#d1d5db" : "#374151",
          strokeWidth: 0,
        },
      },
      groups,
      physics: {
        solver: 'barnesHut',
        adaptiveTimestep: true,
        stabilization: {
          enabled: true,
          iterations: 1000,
          updateInterval: 50,
          onlyDynamicEdges: false,
          fit: true
        },
        barnesHut: {
          gravitationalConstant: -3000,
          centralGravity: 0.3,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.2,
          avoidOverlap: 0.5
        },
        maxVelocity: 30,
        minVelocity: 0.75,
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        navigationButtons: false,
        keyboard: true,
      },
    };

    const networkInstance = new Network(networkRef.current, data, options);

    networkInstance.on("selectNode", (params) => {
      const nodeId = params.nodes[0];
      const node = graphData.nodes.find((n) => n.id === nodeId);
      setSelectedNode(node);
    });

    networkInstance.on("deselectNode", () => {
      setSelectedNode(null);
    });

    setNetwork(networkInstance);

    return () => {
      networkInstance.destroy();
    };
  }, [graphData]);

  // Update colors when theme changes without recreating the network
  useEffect(() => {
    if (!network) return;

    network.setOptions({
      nodes: {
        font: {
          color: isDarkMode ? "#f9fafb" : "#111827",
        },
      },
      edges: {
        font: {
          color: isDarkMode ? "#d1d5db" : "#374151",
        },
      },
    });
  }, [isDarkMode, network]);

  const handleZoomIn = () => {
    network?.moveTo({ scale: (network.getScale() || 1) * 1.2 });
  };

  const handleZoomOut = () => {
    network?.moveTo({ scale: (network.getScale() || 1) * 0.8 });
  };

  const handleFit = () => {
    network?.fit();
  };

  // Search implementation
  useEffect(() => {
    if (!network || !searchQuery) return;

    const query = searchQuery.toLowerCase();
    const matchingNodes = graphData.nodes.filter(n => 
      n.label.toLowerCase().includes(query) || 
      n.group.toLowerCase().includes(query)
    );

    if (matchingNodes.length > 0) {
      // Highlight matching nodes
      network.selectNodes(matchingNodes.map(n => n.id));
      
      // Focus on the first match if it's a specific search
      if (matchingNodes.length === 1 || (searchQuery.length > 3 && matchingNodes.length < 5)) {
        network.focus(matchingNodes[0].id, {
          scale: 1.2,
          animation: { duration: 1000, easingFunction: "easeInOutQuad" }
        });
        setSelectedNode(matchingNodes[0]);
      }
    } else {
        network.unselectAll();
    }
  }, [searchQuery, network, graphData]);

  const handleExportHtml = async () => {
    try {
        toast({ title: "Generating Export", description: "Fetching full graph data..." });
        
        // Fetch ALL data for export
        const response = await graphApi.visualizeFull(2000); 
        
        const nodes = response.nodes.map((node, idx) => ({
            id: idx + 1,
            label: fixEncoding(node.label || node.id),
            group: node.type.toLowerCase(),
            title: `${node.type}: ${node.label}`
        }));

        const nodeIdMap = new Map(response.nodes.map((node, idx) => [node.id, idx + 1]));

        const edges = response.edges.map(edge => ({
            from: nodeIdMap.get(edge.source) || 1,
            to: nodeIdMap.get(edge.target) || 1,
            label: edge.relationship
        }));
        
        const htmlContent = `<!DOCTYPE html>
            <html>
            <head>
            <title>MIRAGE Graph Full Export</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style type="text/css">
                #mynetwork { width: 100vw; height: 100vh; border: 1px solid lightgray; }
            </style>
            </head>
            <body>
            <div id="mynetwork"></div>
            <script type="text/javascript">
                var nodes = new vis.DataSet(${JSON.stringify(nodes)});
                var edges = new vis.DataSet(${JSON.stringify(edges)});
                var container = document.getElementById('mynetwork');
                var data = { nodes: nodes, edges: edges };
                var options = {
                    nodes: { shape: 'dot', size: 16 },
                    physics: { 
                        stabilization: false,
                        barnesHut: { gravitationalConstant: -2000, springLength: 200 } 
                    } 
                };
                var network = new vis.Network(container, data, options);
            </script>
            </body>
            </html>`;
            
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mirage-graph-full-${new Date().toISOString().slice(0, 10)}.html`;
        a.click();
        URL.revokeObjectURL(url);
        
        toast({ title: "Export Complete", description: "Full graph exported to HTML" });
        
    } catch (e) {
        toast({ title: "Export Failed", description: "Could not fetch full graph", variant: "destructive" });
    }
  };

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Knowledge Graph</h1>
          <p className="text-muted-foreground">Explore relationships, entities, and communities</p>
        </div>
      </div>

      <Tabs defaultValue="graph" className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="graph" className="gap-2">
            <NetworkIcon className="w-4 h-4" />
            Graph View
          </TabsTrigger>
          <TabsTrigger value="communities" className="gap-2" onClick={() => communities.length === 0 && loadCommunities()}>
            <Layers className="w-4 h-4" />
            Communities
          </TabsTrigger>
        </TabsList>

        <TabsContent value="graph" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Control Panel */}
        <Card className="p-6 lg:col-span-1 h-fit">
          <h2 className="text-lg font-semibold mb-4">Controls</h2>

          {/* Graph Statistics */}
          <div className="mb-4 pb-4 border-b">
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Nodes:</span>
                <span className="font-semibold">{graphStats?.nodes?.toLocaleString() || graphData.nodes.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Edges:</span>
                <span className="font-semibold">{graphStats?.edges?.toLocaleString() || graphData.edges.length}</span>
              </div>
            </div>
            {/* Value Cap Warning */}
             <div className="mt-3 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-md text-xs text-yellow-600 dark:text-yellow-400">
              <p className="font-medium flex items-center gap-1">
                <Info className="w-3 h-3" />
                Performance Cap Active
              </p>
              <p className="mt-1 opacity-90">
                Showing <strong>Top 500</strong> nodes by connectivity. Use Search to explore specific subgraphs.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Search Entities</label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Node Types</label>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-2">
                {/* Dynamically generate legend from actual node types */}
                {Array.from(new Set(graphData.nodes.map(n => n.group)))
                  .sort()
                  .map(type => {
                    // Generate unique color for each type using hash function
                    const getColorForType = (typeName: string): string => {
                      const colorMap: Record<string, string> = {
                        document: "hsl(160, 60%, 45%)",
                        person: "hsl(217, 70%, 58%)",
                        organization: "hsl(142, 55%, 50%)",
                        location: "hsl(24, 75%, 55%)",
                        concept: "hsl(173, 55%, 48%)",
                        event: "hsl(280, 50%, 58%)",
                        technology: "hsl(340, 60%, 54%)",
                        entity: "hsl(200, 55%, 52%)",
                      };

                      // Return predefined color if exists
                      if (colorMap[typeName]) return colorMap[typeName];

                      // Generate unique color from hash
                      let hash = 0;
                      for (let i = 0; i < typeName.length; i++) {
                        hash = typeName.charCodeAt(i) + ((hash << 5) - hash);
                      }
                      const hue = Math.abs(hash % 360);
                      const saturation = 50 + (Math.abs(hash >> 8) % 20); // 50-70%
                      const lightness = 48 + (Math.abs(hash >> 16) % 12); // 48-60%
                      return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                    };

                    const color = getColorForType(type);
                    const count = graphData.nodes.filter(n => n.group === type).length;

                    return (
                      <div key={type} className="flex items-center gap-2 justify-between min-w-0">
                        <div className="flex items-center gap-2 flex-1 min-w-0">
                          <div
                            className="w-4 h-4 rounded-full border-2 flex-shrink-0"
                            style={{
                              backgroundColor: color,
                              borderColor: color.replace(/\d+%\)/, (m) => {
                                const pct = parseInt(m);
                                return `${pct - 10}%)`;
                              })
                            }}
                          />
                          <span className="text-sm capitalize truncate">{type}</span>
                        </div>
                        <span className="text-xs text-muted-foreground flex-shrink-0 ml-2">{count}</span>
                      </div>
                    );
                  })}
              </div>
            </div>

            <div className="space-y-2 pt-2">
              <Button variant="outline" size="sm" className="w-full gap-2" onClick={handleZoomIn}>
                <ZoomIn className="w-4 h-4" />
                Zoom In
              </Button>
              <Button variant="outline" size="sm" className="w-full gap-2" onClick={handleZoomOut}>
                <ZoomOut className="w-4 h-4" />
                Zoom Out
              </Button>
              <Button variant="outline" size="sm" className="w-full gap-2" onClick={handleFit}>
                <Maximize2 className="w-4 h-4" />
                Fit View
              </Button>
               <Button variant="outline" size="sm" className="w-full gap-2" onClick={handleExportHtml}>
                  <Download className="w-4 h-4" />
                  Export Full HTML
                </Button>
            </div>
          </div>
        </Card>

        {/* Graph Visualization */}
        <Card className="p-6 lg:col-span-3">
          {isLoading ? (
            <div className="w-full h-[600px] flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-12 h-12 mx-auto text-primary animate-spin mb-4" />
                <p className="text-muted-foreground">Loading graph...</p>
              </div>
            </div>
          ) : graphData.nodes.length === 0 ? (
            <div className="w-full h-[600px] flex items-center justify-center">
              <div className="text-center">
                <p className="text-lg font-semibold mb-2">No graph data available</p>
                <p className="text-muted-foreground">Upload and process documents to build the knowledge graph</p>
              </div>
            </div>
          ) : (
            <div
              ref={networkRef}
              className="w-full h-[600px] rounded-lg bg-secondary/20"
              style={{ border: "1px solid hsl(var(--border))" }}
            />
          )}

          {selectedNode && (
            <Card className="mt-4 p-4 bg-secondary/50">
                <h3 className="font-bold text-lg mb-2">{fixEncoding(selectedNode.label)}</h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline">{selectedNode.group}</Badge>
                  <span className="text-sm font-medium">{selectedNode.label}</span>
                </div>
                <p className="text-sm text-muted-foreground">{selectedNode.title}</p>
              </div>
            </Card>
          )}
        </Card>
      </div>
        </TabsContent>

        <TabsContent value="communities" className="mt-6">
          {isLoadingCommunities ? (
            <div className="text-center py-12">
              <Loader2 className="w-12 h-12 mx-auto text-primary animate-spin mb-4" />
              <p className="text-muted-foreground">Loading communities...</p>
            </div>
          ) : communities.length === 0 ? (
            <Card className="p-12 text-center">
              <Layers className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-xl font-semibold mb-2">No communities found</h3>
              <p className="text-muted-foreground mb-4">
                Communities are detected using the Louvain algorithm during graph processing
              </p>
              <Button onClick={() => loadCommunities()}>Load Communities</Button>
            </Card>
          ) : (
            <div className="space-y-6">
              {/* Community Level Filter */}
              <Card className="p-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-muted-foreground">
                    Showing {communities.length} communities across {Math.max(...communities.map(c => c.level)) + 1} levels
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant={selectedLevel === undefined ? "default" : "outline"}
                      size="sm"
                      onClick={() => {
                        setSelectedLevel(undefined);
                        loadCommunities();
                      }}
                    >
                      All Levels
                    </Button>
                    {Array.from(new Set(communities.map(c => c.level))).sort().map(level => (
                      <Button
                        key={level}
                        variant={selectedLevel === level ? "default" : "outline"}
                        size="sm"
                        onClick={() => {
                          setSelectedLevel(level);
                          loadCommunities(level);
                        }}
                      >
                        L{level}
                      </Button>
                    ))}
                  </div>
                </div>
              </Card>

              {/* Communities Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {communities.map((community) => (
                  <Card key={community.id} className="p-4 hover:shadow-lg transition-shadow">
                    <div className="space-y-3">
                      {/* Header */}
                      <div className="flex flex-col gap-2 mb-2">
                         <div className="flex items-start justify-between gap-2">
                            <h3 className="font-semibold text-sm leading-tight">
                                {community.themes && community.themes.length > 0 
                                    ? community.themes[0] 
                                    : "Community Group"}
                            </h3>
                            <Badge variant="outline" className="text-xs shrink-0">
                                Level {community.level}
                            </Badge>
                         </div>
                         <div className="flex items-center gap-2">
                            <Badge variant="secondary" className="text-xs">
                                {community.member_count} members
                            </Badge>
                         </div>
                      </div>

                      {/* Community ID */}
                      <div className="text-xs text-muted-foreground font-mono">
                        {community.id}
                      </div>

                      {/* Themes */}
                      {community.themes && community.themes.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {community.themes.slice(0, 3).map((theme, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              {theme}
                            </Badge>
                          ))}
                          {community.themes.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{community.themes.length - 3}
                            </Badge>
                          )}
                        </div>
                      )}

                      {/* Summary */}
                      {community.summary && (
                        <div className="bg-secondary/30 rounded-lg p-3">
                          <p className="text-xs leading-relaxed line-clamp-4">
                            {community.summary}
                          </p>
                        </div>
                      )}

                      {/* Sample Members */}
                      {community.sample_members && community.sample_members.length > 0 && (
                        <div className="pt-2 border-t">
                          <details className="text-xs">
                            <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                              Members ({community.member_count})
                            </summary>
                            <div className="mt-2 space-y-1">
                              {community.sample_members.map((member, idx) => (
                                <div key={idx} className="text-xs pl-2">
                                  • {member}
                                </div>
                              ))}
                              {community.member_count > community.sample_members.length && (
                                <div className="text-xs pl-2 text-muted-foreground">
                                  ... and {community.member_count - community.sample_members.length} more
                                </div>
                              )}
                            </div>
                          </details>
                        </div>
                      )}

                      {/* Parent Info */}
                      {community.parent_id && (
                        <div className="pt-2 border-t text-xs text-muted-foreground">
                          Parent: {community.parent_id}
                        </div>
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
