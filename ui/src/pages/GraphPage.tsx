import { useEffect, useRef, useState } from "react";
import { Network } from "vis-network";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Download, Filter, ZoomIn, ZoomOut, Maximize2, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { graphApi } from "@/lib/api";

export default function GraphPage() {
  const networkRef = useRef<HTMLDivElement>(null);
  const [network, setNetwork] = useState<Network | null>(null);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [graphData, setGraphData] = useState<{ nodes: any[]; edges: any[] }>({ nodes: [], edges: [] });
  const [isDarkMode, setIsDarkMode] = useState(document.documentElement.classList.contains('dark'));
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

  // Load graph data from backend
  useEffect(() => {
    loadGraph();
  }, []);

  const loadGraph = async () => {
    try {
      setIsLoading(true);
      const response = await graphApi.visualizeFull(1000);  // Increased from 100 to 1000

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
        stabilization: true,
        barnesHut: {
          gravitationalConstant: -2000,
          springConstant: 0.001,
          springLength: 200,
        },
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

  const handleExport = () => {
    if (!network || !networkRef.current) return;

    // Get current theme colors
    const bgColor = isDarkMode ? '#1a1a1a' : '#f5f5f0';
    const textColor = isDarkMode ? '#f3f4f6' : '#1f2937';
    const borderColor = isDarkMode ? '#383838' : '#e5e7eb';

    // Prepare graph data for export
    const graphDataJson = JSON.stringify({
      nodes: graphData.nodes,
      edges: graphData.edges
    });

    // Create HTML content with embedded vis-network
    const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MIRAGE Knowledge Graph Export</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Naskh+Arabic:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    body {
      font-family: 'Inter', 'Noto Naskh Arabic', sans-serif;
      background: ${bgColor};
      color: ${textColor};
      padding: 2rem;
    }
    .container {
      max-width: 1400px;
      margin: 0 auto;
    }
    .header {
      margin-bottom: 2rem;
      padding-bottom: 1rem;
      border-bottom: 2px solid ${borderColor};
    }
    h1 {
      font-size: 2rem;
      font-weight: bold;
      margin-bottom: 0.5rem;
    }
    .stats {
      display: flex;
      gap: 2rem;
      font-size: 0.875rem;
      color: ${isDarkMode ? '#9ca3af' : '#6b7280'};
    }
    .graph-container {
      position: relative;
      background: ${isDarkMode ? '#262626' : '#ffffff'};
      border: 1px solid ${borderColor};
      border-radius: 0.75rem;
      height: 600px;
      overflow: hidden;
    }
    #network {
      width: 100%;
      height: 100%;
    }
    .controls {
      position: absolute;
      top: 1rem;
      right: 1rem;
      display: flex;
      gap: 0.5rem;
      z-index: 100;
    }
    .control-btn {
      background: ${isDarkMode ? '#404040' : '#ffffff'};
      border: 1px solid ${borderColor};
      color: ${textColor};
      padding: 0.5rem;
      border-radius: 0.5rem;
      cursor: pointer;
      font-size: 0.875rem;
      transition: all 0.2s;
    }
    .control-btn:hover {
      background: ${isDarkMode ? '#505050' : '#f5f5f0'};
    }
    .footer {
      margin-top: 2rem;
      text-align: center;
      font-size: 0.875rem;
      color: ${isDarkMode ? '#9ca3af' : '#6b7280'};
    }
    .legend {
      margin-top: 2rem;
      padding: 1.5rem;
      background: ${isDarkMode ? '#262626' : '#ffffff'};
      border: 1px solid ${borderColor};
      border-radius: 0.75rem;
    }
    .legend h2 {
      font-size: 1.25rem;
      font-weight: 600;
      margin-bottom: 1rem;
    }
    .legend-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 0.75rem;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .legend-color {
      width: 1rem;
      height: 1rem;
      border-radius: 50%;
      border: 2px solid;
      flex-shrink: 0;
    }
    .legend-label {
      font-size: 0.875rem;
      text-transform: capitalize;
    }
    .legend-count {
      margin-left: auto;
      font-size: 0.75rem;
      color: ${isDarkMode ? '#9ca3af' : '#6b7280'};
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>MIRAGE Knowledge Graph</h1>
      <div class="stats">
        <span>Nodes: ${graphData.nodes.length}</span>
        <span>Edges: ${graphData.edges.length}</span>
        <span>Exported: ${new Date().toLocaleString()}</span>
      </div>
    </div>

    <div class="graph-container">
      <div class="controls">
        <button class="control-btn" onclick="zoomIn()">Zoom In</button>
        <button class="control-btn" onclick="zoomOut()">Zoom Out</button>
        <button class="control-btn" onclick="fitGraph()">Fit</button>
      </div>
      <div id="network"></div>
    </div>

    <div class="legend">
      <h2>Node Types</h2>
      <div class="legend-grid">
        ${Array.from(new Set(graphData.nodes.map(n => n.group)))
          .sort()
          .map(type => {
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
              if (colorMap[typeName]) return colorMap[typeName];
              let hash = 0;
              for (let i = 0; i < typeName.length; i++) {
                hash = typeName.charCodeAt(i) + ((hash << 5) - hash);
              }
              const hue = Math.abs(hash % 360);
              const saturation = 50 + (Math.abs(hash >> 8) % 20);
              const lightness = 48 + (Math.abs(hash >> 16) % 12);
              return 'hsl(' + hue + ', ' + saturation + '%, ' + lightness + '%)';
            };
            const color = getColorForType(type);
            const count = graphData.nodes.filter(n => n.group === type).length;
            return '<div class="legend-item">' +
              '<div class="legend-color" style="background-color: ' + color + '; border-color: ' + color + ';"></div>' +
              '<span class="legend-label">' + type + '</span>' +
              '<span class="legend-count">' + count + '</span>' +
              '</div>';
          })
          .join('')}
      </div>
    </div>

    <div class="footer">
      <p>Generated by MIRAGE - Multilingual Information Retrieval with Accelerated Graph Embeddings</p>
    </div>
  </div>

  <script>
    // Graph data
    const graphData = ${graphDataJson};

    // Color mapping function
    function getColorForType(typeName) {
      const colorMap = {
        document: { background: "hsl(160, 60%, 45%)", border: "hsl(160, 60%, 35%)" },
        person: { background: "hsl(217, 70%, 58%)", border: "hsl(217, 70%, 48%)" },
        organization: { background: "hsl(142, 55%, 50%)", border: "hsl(142, 55%, 40%)" },
        location: { background: "hsl(24, 75%, 55%)", border: "hsl(24, 75%, 45%)" },
        concept: { background: "hsl(173, 55%, 48%)", border: "hsl(173, 55%, 38%)" },
        event: { background: "hsl(280, 50%, 58%)", border: "hsl(280, 50%, 48%)" },
        technology: { background: "hsl(340, 60%, 54%)", border: "hsl(340, 60%, 44%)" },
        entity: { background: "hsl(200, 55%, 52%)", border: "hsl(200, 55%, 42%)" },
      };

      if (colorMap[typeName]) return colorMap[typeName];

      let hash = 0;
      for (let i = 0; i < typeName.length; i++) {
        hash = typeName.charCodeAt(i) + ((hash << 5) - hash);
      }
      const hue = Math.abs(hash % 360);
      const saturation = 50 + (Math.abs(hash >> 8) % 20);
      const lightness = 48 + (Math.abs(hash >> 16) % 12);

      return {
        background: 'hsl(' + hue + ', ' + saturation + '%, ' + lightness + '%)',
        border: 'hsl(' + hue + ', ' + saturation + '%, ' + (lightness - 10) + '%)'
      };
    }

    // Generate groups for all node types
    const groups = {};
    Array.from(new Set(graphData.nodes.map(n => n.group))).forEach(type => {
      groups[type] = { color: getColorForType(type) };
    });

    // Network options
    const options = {
      nodes: {
        shape: "dot",
        size: 25,
        font: {
          size: 14,
          face: "Noto Naskh Arabic, Inter",
          color: "${isDarkMode ? '#f9fafb' : '#111827'}",
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
          color: "${isDarkMode ? '#d1d5db' : '#374151'}",
          strokeWidth: 0,
        },
      },
      groups: groups,
      physics: {
        stabilization: true,
        barnesHut: {
          gravitationalConstant: -2000,
          springConstant: 0.001,
          springLength: 200,
        },
      },
      interaction: {
        hover: true,
        tooltipDelay: 100,
        navigationButtons: false,
        keyboard: true,
      },
    };

    // Initialize network
    const container = document.getElementById('network');
    const network = new vis.Network(container, graphData, options);

    // Control functions
    function zoomIn() {
      const scale = network.getScale();
      network.moveTo({ scale: scale * 1.2 });
    }

    function zoomOut() {
      const scale = network.getScale();
      network.moveTo({ scale: scale * 0.8 });
    }

    function fitGraph() {
      network.fit();
    }

    // Fit graph on load
    network.once('stabilizationIterationsDone', function() {
      network.fit();
    });
  </script>
</body>
</html>`;

    // Download HTML file
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mirage-graph.html';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 animate-fade-in pb-20 md:pb-0">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Knowledge Graph</h1>
          <p className="text-muted-foreground">Explore relationships and entities</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Control Panel */}
        <Card className="p-6 lg:col-span-1 h-fit">
          <h2 className="text-lg font-semibold mb-4">Controls</h2>

          {/* Graph Statistics */}
          <div className="mb-4 pb-4 border-b">
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Nodes:</span>
                <span className="font-semibold">{graphData.nodes.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Edges:</span>
                <span className="font-semibold">{graphData.edges.length}</span>
              </div>
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
              <Button variant="outline" size="sm" className="w-full gap-2" onClick={handleExport}>
                <Download className="w-4 h-4" />
                Export HTML
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
              <h3 className="font-semibold mb-2">Selected Node</h3>
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
    </div>
  );
}
