# MIRAGE UI - Quick Reference Guide

## Essential File Paths

### Core Application Files
```
/home/aub/boo/MIRAGE/ui/src/
├── App.tsx                    # Root component with React Router setup
├── main.tsx                   # React entry point
├── pages/                     # Page components (what you need to modify)
│   ├── VectorsPage.tsx        # MODIFY: Add 3-column grid
│   ├── GraphPage.tsx          # MODIFY: Add communities tab
│   └── ChatPage.tsx           # MODIFY: Add explainability tabs
├── lib/
│   └── api.ts                 # REFERENCE: API endpoints
└── components/
    └── ui/                    # Pre-built shadcn components
```

## Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| Framework | React | 18.3.1 |
| Build Tool | Vite | 5.4.19 |
| Language | TypeScript | 5.8.3 |
| Styling | TailwindCSS | 3.4.17 |
| UI Components | shadcn/ui + Radix | Latest |
| Routing | React Router | 6.30.1 |
| Graph Viz | vis-network | 10.0.2 |
| Icons | lucide-react | 0.462.0 |

## What's Already Installed

All dependencies you need are already in package.json:
- Tabs component: `@radix-ui/react-tabs`
- Graph visualization: `vis-network`, `vis-data`
- Styling: TailwindCSS, shadcn/ui components
- Icons: lucide-react
- Forms: React Hook Form, Zod

**No new npm packages needed for your implementation.**

## Implementation Priority

### Quick Wins (15 min each)
1. VectorsPage - Change grid layout from vertical to 3 columns
2. Add Tabs import to GraphPage and ChatPage

### Medium Tasks (45-60 min each)
3. GraphPage - Wrap visualization in tabs, add communities placeholder
4. ChatPage - Expand right panel with 3 tabs for explainability

### Backend Integration (depends on API)
5. Community detection endpoint integration
6. Explainability data structure mapping

## Key Component Imports

```typescript
// Always use absolute imports with @/ alias
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useToast } from "@/hooks/use-toast"
import { cn } from "@/lib/utils"
```

## Common TailwindCSS Patterns

```typescript
// Responsive grid (used throughout project)
"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"

// Responsive flexbox
"flex flex-col md:flex-row gap-4"

// Mobile navigation spacing (accounts for bottom nav)
"pb-20 md:pb-0"

// Theme-aware colors
"bg-secondary/30 text-muted-foreground"
"dark:bg-slate-900 dark:text-slate-100"
```

## API Integration Pattern

All API calls follow this pattern:

```typescript
// In api.ts
export const myApi = {
  getList: (params?: { limit?: number }) => {
    const query = new URLSearchParams()
    if (params?.limit) query.set('limit', params.limit.toString())
    return fetchApi<ResponseType>(`/endpoint?${query}`)
  }
}

// In component
import { myApi } from "@/lib/api"

const [data, setData] = useState(null)
const [isLoading, setIsLoading] = useState(true)

useEffect(() => {
  myApi.getList({ limit: 50 })
    .then(data => setData(data))
    .catch(error => console.error(error))
    .finally(() => setIsLoading(false))
}, [])
```

## State Management Pattern

```typescript
// Local state with hooks (current pattern in MIRAGE)
const [data, setData] = useState<DataType[]>([])
const [isLoading, setIsLoading] = useState(false)
const [activeTab, setActiveTab] = useState("tab1")

// For complex interactions with DOM/libraries
const ref = useRef<HTMLDivElement>(null)

useEffect(() => {
  // Load data, setup, etc.
  return () => {
    // Cleanup
  }
}, [dependencies])
```

## Graph Visualization Pattern

```typescript
import { Network } from "vis-network"

const networkRef = useRef<HTMLDivElement>(null)
const [graphData, setGraphData] = useState({ nodes: [], edges: [] })

useEffect(() => {
  if (!networkRef.current || graphData.nodes.length === 0) return
  
  const network = new Network(networkRef.current, graphData, {
    nodes: { shape: "dot", size: 25 },
    edges: { width: 2 },
    physics: { stabilization: true }
  })
  
  return () => network.destroy()
}, [graphData])

// In JSX:
<div ref={networkRef} className="w-full h-[600px] bg-secondary/20" />
```

## Current Page Structures

### VectorsPage
```
Header with title
Search/Filter bar
Grid of chunks (currently: 1 column)
Pagination controls
```

### GraphPage
```
2-column layout (4:1 ratio)
├── Left: Control panel (20% width)
│   ├── Stats (nodes/edges)
│   ├── Search
│   ├── Node type legend
│   └── Zoom/Export buttons
└── Right: Main content (80% width)
    ├── Network visualization
    └── Selected node details
```

### ChatPage
```
2-column layout (50:50)
├── Left: Chat interface
│   ├── Message list
│   └── Input field
└── Right: Context (currently single visualization)
    ├── Context graph (vis-network)
    └── Retrieved nodes count
```

## Responsive Breakpoints (from TailwindCSS)

- `sm`: 640px
- `md`: 768px (main breakpoint in project)
- `lg`: 1024px (used for main content areas)
- `xl`: 1280px
- `2xl`: 1536px

Most components in MIRAGE use:
- `hidden md:flex` - hidden on mobile, visible on tablet+
- `grid-cols-1 md:grid-cols-2 lg:grid-cols-3` - responsive columns
- `w-full lg:w-96` - full width on mobile, fixed on desktop

## Dark Mode

Automatically handled by:
1. `document.documentElement.classList.add('dark')` in Layout.tsx
2. TailwindCSS dark mode classes
3. No additional CSS needed

Just use CSS classes like:
```typescript
className="bg-white dark:bg-slate-900"
// Or better: use semantic tokens
className="bg-background text-foreground"
```

## Development Workflow

```bash
# Start dev server on http://localhost:8080
npm run dev

# Build for production
npm run build

# Lint code
npm run lint

# Preview build locally
npm run preview
```

## Environment Configuration

```bash
# .env file in /ui/ directory
VITE_API_BASE_URL=http://localhost:8000
```

API will default to `http://localhost:8000` if not set.

## Common Component Properties

### Card
```typescript
<Card className="p-6 hover:shadow-lg transition-shadow">
  {children}
</Card>
```

### Button
```typescript
<Button variant="default|outline|ghost|secondary|destructive" size="sm|md|lg|icon">
  Click me
</Button>
```

### Input
```typescript
<Input 
  placeholder="Type..."
  value={value}
  onChange={(e) => setValue(e.target.value)}
  className="pl-9" // left padding for icon
/>
```

### Badge
```typescript
<Badge variant="default|outline|secondary">
  Label
</Badge>
```

## Error Handling Pattern

```typescript
const { toast } = useToast()

try {
  const data = await apiFunction()
  // success
} catch (error: any) {
  console.error('Error:', error)
  toast({
    title: "Error",
    description: error.message || "Something went wrong",
    variant: "destructive",
  })
}
```

## Files You'll Actually Modify

### Must Modify (3 files)
1. `/ui/src/pages/VectorsPage.tsx` - Add grid layout
2. `/ui/src/pages/GraphPage.tsx` - Add communities tab
3. `/ui/src/pages/ChatPage.tsx` - Add explainability tabs

### Might Add To (1 file)
4. `/ui/src/lib/api.ts` - Add communities endpoint if needed

### Reference Only (no changes needed)
- `/ui/src/components/ui/tabs.tsx` - Just import and use
- `/ui/src/components/Layout.tsx` - Main layout, already set up
- `/ui/src/App.tsx` - Routing is already configured

## Performance Optimization Notes

1. **Network recreations**: Always clean up in useEffect return
2. **Graph limits**: Current limit is 1000 nodes - consider pagination for larger graphs
3. **Tab content**: Render only visible tabs to save performance (already done in Radix UI Tabs)
4. **Theme changes**: Uses MutationObserver - minimal overhead

## Common Pitfalls to Avoid

- Don't forget `className` - TailwindCSS requires class names
- Always clean up in useEffect (especially vis-network)
- Use refs for DOM manipulation (vis-network), not state
- Import from `@/` not relative paths
- Don't modify response data directly - copy then modify
- Remember responsive classes when adding new layouts

## Getting Help

### Check Existing Code
- VectorsPage for pagination pattern
- GraphPage for vis-network setup
- ChatPage for state management with graph
- DataSourcesPage for Tabs usage

### Common Patterns Already Used
- API calls with error handling - check `/lib/api.ts`
- State management with hooks - check any page component
- Responsive design - check Layout.tsx
- Graph visualization - check GraphPage.tsx and ChatPage.tsx

