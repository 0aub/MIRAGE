# Implementation Roadmap for UI Enhancements

## Quick Start Checklist

### Phase 1: Vectorstores Tab Redesign (3 blocks per row)
**Effort**: 15 minutes | **Complexity**: Low

**File to modify**: `/home/aub/boo/MIRAGE/ui/src/pages/VectorsPage.tsx` (Line 145)

**Changes needed**:
```typescript
// BEFORE (line 145):
<div className="space-y-4">
  {filteredChunks.map((chunk) => (
    <Card key={chunk.id} className="p-6 hover:shadow-lg transition-shadow">

// AFTER:
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {filteredChunks.map((chunk) => (
    <Card key={chunk.id} className="p-4 hover:shadow-lg transition-shadow">
```

**Additional**:
- Reduce padding in Card from `p-6` to `p-4` for better fit
- Update text sizes in chunk display (optional)
- Pagination stays below grid

---

### Phase 2: Graph Visualization with Communities Tab
**Effort**: 45 minutes | **Complexity**: Medium

**File to modify**: `/home/aub/boo/MIRAGE/ui/src/pages/GraphPage.tsx`

**Step 1**: Add tab state at top with other state declarations (around line 16)
```typescript
const [graphTab, setGraphTab] = useState<"full" | "communities">("full")
```

**Step 2**: Import Tabs component (add to imports at top)
```typescript
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
```

**Step 3**: Wrap main graph section (around line 546) with tabs
```typescript
<div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
  {/* Control Panel - keep as is */}
  <Card className="p-6 lg:col-span-1 h-fit">
    {/* existing controls */}
  </Card>

  {/* Graph Section with Tabs */}
  <Card className="p-6 lg:col-span-3">
    <Tabs value={graphTab} onValueChange={(v) => setGraphTab(v as "full" | "communities")}>
      <TabsList>
        <TabsTrigger value="full">Full Graph</TabsTrigger>
        <TabsTrigger value="communities">Communities</TabsTrigger>
      </TabsList>
      
      <TabsContent value="full" className="mt-4">
        {/* Current network visualization code (lines 661-681) */}
      </TabsContent>
      
      <TabsContent value="communities" className="mt-4">
        {/* New communities visualization */}
        <div className="w-full h-[600px] flex items-center justify-center">
          <Card className="p-8">
            <h3 className="text-lg font-semibold mb-2">Communities</h3>
            <p className="text-muted-foreground">Community detection coming soon</p>
          </Card>
        </div>
      </TabsContent>
    </Tabs>
  </Card>
</div>
```

**Step 4**: Add API call for community data (if available)
```typescript
// Add to api.ts if endpoint exists:
communities: (limit: number = 50) => {
  return fetchApi<{
    communities: Array<{
      id: number
      name: string
      node_count: number
      edge_count: number
      nodes: Array<{ id: string; label: string }>;
      edges: Array<{ source: string; target: string }>;
    }>;
  }>(`/graph/communities?limit=${limit}`);
}
```

---

### Phase 3: Chat Page with Expanded Explainability
**Effort**: 60 minutes | **Complexity**: Medium-High

**File to modify**: `/home/aub/boo/MIRAGE/ui/src/pages/ChatPage.tsx`

**Step 1**: Update state variables (around line 34)
```typescript
const [contextTab, setContextTab] = useState<"graph" | "communities" | "context">("graph")
```

**Step 2**: Add interface for explainability data
```typescript
interface ExplainabilityData {
  nodes: Array<{ id: string; label: string; type: string; score: number }>;
  edges: Array<{ source: string; target: string; relationship: string }>;
  communities: Array<{ id: number; name: string; members: string[] }>;
  statistics: {
    retrieved_nodes: number;
    retrieved_edges: number;
    response_time_ms: number;
    confidence_score: number;
  };
}
```

**Step 3**: Add state for explainability
```typescript
const [explainability, setExplainability] = useState<ExplainabilityData | null>(null)
```

**Step 4**: Update handleSend to populate explainability (around line 146)
```typescript
// After setting context graph:
setExplainability({
  nodes: response.graph_visualization.nodes || [],
  edges: response.graph_visualization.edges || [],
  communities: response.communities || [],
  statistics: {
    retrieved_nodes: response.workflow_metadata?.retrieved_nodes_count || 0,
    retrieved_edges: response.graph_visualization?.edges?.length || 0,
    response_time_ms: response.workflow_metadata?.response_time_ms || 0,
    confidence_score: response.workflow_metadata?.confidence_score || 0.8,
  }
})
```

**Step 5**: Replace the right panel (around line 284)
```typescript
{/* Active Subgraph - Now with Tabs */}
<Card className="lg:w-96 p-6 flex flex-col">
  <h2 className="text-lg font-semibold mb-4">Explainability</h2>
  
  <Tabs value={contextTab} onValueChange={(v) => setContextTab(v as any)} className="flex-1 flex flex-col">
    <TabsList className="grid w-full grid-cols-3">
      <TabsTrigger value="graph">Graph</TabsTrigger>
      <TabsTrigger value="communities">Communities</TabsTrigger>
      <TabsTrigger value="context">Context</TabsTrigger>
    </TabsList>
    
    <TabsContent value="graph" className="flex-1 mt-2">
      {contextGraph.nodes.length > 0 ? (
        <>
          <div
            ref={networkRef}
            className="w-full h-64 rounded-lg bg-secondary/20"
            style={{ border: "1px solid hsl(var(--border))" }}
          />
          <div className="mt-4 space-y-2 text-sm">
            <p className="font-medium">Nodes: {contextGraph.nodes.length}</p>
            <p className="font-medium">Edges: {contextGraph.edges.length}</p>
          </div>
        </>
      ) : (
        <div className="h-64 flex items-center justify-center text-center">
          <p className="text-sm text-muted-foreground">
            {isLoading ? "Retrieving..." : "Send a message to see the context graph"}
          </p>
        </div>
      )}
    </TabsContent>
    
    <TabsContent value="communities" className="flex-1 mt-2 overflow-y-auto">
      {explainability?.communities && explainability.communities.length > 0 ? (
        <div className="space-y-3">
          {explainability.communities.map((community, idx) => (
            <Card key={idx} className="p-3 bg-secondary/30">
              <h4 className="font-semibold text-sm">{community.name}</h4>
              <p className="text-xs text-muted-foreground">{community.members.length} members</p>
              <div className="mt-1 flex flex-wrap gap-1">
                {community.members.slice(0, 3).map((member, i) => (
                  <span key={i} className="text-xs bg-primary/20 px-2 py-1 rounded">
                    {member}
                  </span>
                ))}
                {community.members.length > 3 && (
                  <span className="text-xs text-muted-foreground">+{community.members.length - 3}</span>
                )}
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">No community data available</p>
      )}
    </TabsContent>
    
    <TabsContent value="context" className="flex-1 mt-2 overflow-y-auto space-y-3">
      {explainability ? (
        <>
          <div className="space-y-2">
            <h4 className="font-semibold text-sm">Statistics</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-secondary/30 p-2 rounded">
                <p className="text-muted-foreground">Nodes Retrieved</p>
                <p className="font-semibold">{explainability.statistics.retrieved_nodes}</p>
              </div>
              <div className="bg-secondary/30 p-2 rounded">
                <p className="text-muted-foreground">Edges</p>
                <p className="font-semibold">{explainability.statistics.retrieved_edges}</p>
              </div>
              <div className="bg-secondary/30 p-2 rounded">
                <p className="text-muted-foreground">Response Time</p>
                <p className="font-semibold">{explainability.statistics.response_time_ms}ms</p>
              </div>
              <div className="bg-secondary/30 p-2 rounded">
                <p className="text-muted-foreground">Confidence</p>
                <p className="font-semibold">{(explainability.statistics.confidence_score * 100).toFixed(0)}%</p>
              </div>
            </div>
          </div>
          
          <div className="space-y-2">
            <h4 className="font-semibold text-sm">Top Nodes</h4>
            <div className="space-y-1">
              {explainability.nodes.slice(0, 5).map((node, idx) => (
                <div key={idx} className="text-xs p-2 bg-secondary/30 rounded">
                  <div className="font-medium flex justify-between">
                    <span>{node.label}</span>
                    <span className="text-muted-foreground">{(node.score * 100).toFixed(0)}%</span>
                  </div>
                  <p className="text-muted-foreground">{node.type}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">No context data available</p>
      )}
    </TabsContent>
  </Tabs>
</Card>
```

---

## File Locations Quick Reference

| Task | File | Lines |
|------|------|-------|
| 3-column grid | `/ui/src/pages/VectorsPage.tsx` | 145-197 |
| Graph tabs | `/ui/src/pages/GraphPage.tsx` | 546-696 |
| Chat explainability | `/ui/src/pages/ChatPage.tsx` | 284-311 |
| API methods | `/ui/src/lib/api.ts` | 198-273 (graphApi) |
| UI tabs component | `/ui/src/components/ui/tabs.tsx` | - |

---

## Dependencies Check

All required packages are already installed:
- `@radix-ui/react-tabs` - Tabs component
- `vis-network` - Graph visualization
- `lucide-react` - Icons
- `tailwindcss` - Styling

No new npm packages needed.

---

## Testing Checklist

After implementation:

### VectorsPage
- [ ] Grid displays 3 columns on large screens
- [ ] Grid displays 2 columns on medium screens
- [ ] Grid displays 1 column on mobile
- [ ] Pagination works correctly
- [ ] Search/filter still works
- [ ] Cards are properly aligned

### GraphPage
- [ ] Tabs switch between "Full Graph" and "Communities"
- [ ] Graph visualization renders in "Full Graph" tab
- [ ] Communities content shows in "Communities" tab
- [ ] Controls panel still works
- [ ] Selected node display still works
- [ ] Export functionality still works

### ChatPage
- [ ] Tabs switch between Graph, Communities, and Context
- [ ] Graph visualization shows in Graph tab
- [ ] Communities display in Communities tab
- [ ] Statistics display in Context tab
- [ ] All data updates when sending messages
- [ ] Mobile layout still responsive

---

## Performance Tips

1. **Graph Visualization**: Consider memoizing the Network component if it becomes slow
2. **Tabs**: Use `React.memo()` to prevent unnecessary re-renders of tab content
3. **State**: Keep state as local as possible (don't lift unnecessarily)
4. **API Calls**: The existing patterns already handle this well

---

## Dark Mode Support

All components automatically support dark mode via:
- TailwindCSS dark mode classes
- Theme toggle in Layout.tsx
- CSS variables for colors

No additional work needed.

---

## Next Steps for Backend Integration

When implementing community detection on the backend:

1. **Add API endpoint**: `/graph/communities`
2. **Response format**:
```json
{
  "communities": [
    {
      "id": 1,
      "name": "Community 1",
      "node_count": 15,
      "edge_count": 25,
      "nodes": [{"id": "n1", "label": "Node 1"}],
      "edges": [{"source": "n1", "target": "n2"}]
    }
  ]
}
```

3. **Add to api.ts**:
```typescript
communities: (limit: number = 50) => {
  return fetchApi<{
    communities: Array<any>
  }>(`/graph/communities?limit=${limit}`)
}
```

---

