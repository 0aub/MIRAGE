# MIRAGE UI Architecture Analysis

## Executive Summary
The MIRAGE UI is a modern React TypeScript application built with Vite, using TailwindCSS for styling and shadcn/ui for component library. The architecture follows a page-based structure with clear API integration patterns and state management via React hooks and TanStack React Query.

---

## 1. UI Framework & Technology Stack

### Core Framework
- **Framework**: React 18.3.1
- **Build Tool**: Vite 5.4.19
- **Language**: TypeScript 5.8.3
- **Package Manager**: npm

### UI Library & Styling
- **Component Library**: shadcn/ui (Radix UI based)
- **Styling**: TailwindCSS 3.4.17
- **Icon Library**: lucide-react (0.462.0)
- **CSS Utilities**: class-variance-authority, clsx, tailwind-merge

### State Management & Data Fetching
- **State Management**: React hooks (useState, useRef, useEffect)
- **Data Fetching**: TanStack React Query 5.83.0
- **Routing**: React Router DOM 6.30.1

### Form Handling & Validation
- **Form Library**: React Hook Form 7.61.1
- **Validation**: Zod 3.25.76

### Additional Libraries
- **Charting**: Recharts 3.3.0
- **Graph Visualization**: vis-network 10.0.2, vis-data 8.0.3
- **Markdown**: marked 16.4.1
- **UI Utilities**: sonner (toast notifications), vaul (drawers)
- **Layout**: react-resizable-panels, react-dropzone

### Development Tools
- **Linting**: ESLint 9.32.0 with React plugin
- **TypeScript Config**: Multiple tsconfigs for app, node, and build

---

## 2. Directory Structure

```
/home/aub/boo/MIRAGE/ui/
├── src/
│   ├── components/
│   │   ├── ui/                    # shadcn/ui components (50+ files)
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── tabs.tsx           # Tab component
│   │   │   ├── form.tsx
│   │   │   └── ... (others)
│   │   ├── data-sources/          # Data source input components
│   │   │   ├── FileUploadTab.tsx
│   │   │   ├── WebPageTab.tsx
│   │   │   └── YouTubeTab.tsx
│   │   ├── Layout.tsx             # Main layout with navigation
│   │   └── NavLink.tsx
│   ├── pages/                     # Page components (routes)
│   │   ├── Dashboard.tsx
│   │   ├── DataSourcesPage.tsx    # Data sources with tabs
│   │   ├── GraphPage.tsx          # Knowledge graph visualization
│   │   ├── VectorsPage.tsx        # Vector store viewer
│   │   ├── ChatPage.tsx           # Chat with graph context
│   │   ├── SettingsPage.tsx
│   │   ├── UploadPage.tsx
│   │   ├── Index.tsx
│   │   └── NotFound.tsx
│   ├── hooks/
│   │   ├── use-mobile.tsx
│   │   └── use-toast.ts
│   ├── lib/
│   │   ├── api.ts                 # API client
│   │   └── utils.ts
│   ├── App.tsx                    # Root component with routing
│   ├── main.tsx                   # React entry point
│   ├── index.css                  # Global styles
│   └── vite-env.d.ts
├── vite.config.ts                 # Vite configuration
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── package.json
├── tailwind.config.js
├── postcss.config.js
└── eslint.config.js
```

---

## 3. Key Pages & Components for Your Implementation

### A. VectorsPage (Vector Store Tab Redesign)
**File**: `/home/aub/boo/MIRAGE/ui/src/pages/VectorsPage.tsx`
**Current Implementation**: 
- Single column list layout
- Pagination controls (limit=50)
- Search and filter by document ID
- Displays chunks with metadata

**What you need to modify**:
- Change from `<div className="space-y-4">` to a 3-column grid layout
- Use `grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4`
- Keep card components the same, just adjust grid

### B. GraphPage (Graph Visualization with Communities Tab)
**File**: `/home/aub/boo/MIRAGE/ui/src/pages/GraphPage.tsx`
**Current Implementation**:
- Uses vis-network (Network from "vis-network")
- Displays full knowledge graph
- Has a control panel on the left with legend
- Shows selected node details
- Export to HTML functionality

**What you need to modify**:
- Add a new tab container for "Graph" vs "Communities"
- Communities tab should show community detection visualization
- Current implementation can stay as "Full Graph" tab
- Add community statistics/metrics

### C. ChatPage (Conversation Tab with Explainability)
**File**: `/home/aub/boo/MIRAGE/ui/src/pages/ChatPage.tsx`
**Current Implementation**:
- Two-column layout: Chat + Active Context Graph
- Uses vis-network for context visualization
- Shows sources/citations
- Displays retrieved nodes count
- RTL/LTR language support

**What you need to modify**:
- Enhance right panel with tabs:
  1. "Active Context" (current graph view)
  2. "Communities" (communities in the context)
  3. "Retrieved Context" (detailed node/edge/citation info)
- Add explainability data display

---

## 4. API Integration

**File**: `/home/aub/boo/MIRAGE/ui/src/lib/api.ts`

### API Endpoints Used by Key Pages:

#### GraphPage Endpoints:
```typescript
graphApi.visualizeFull(limit: number)
  Returns: { nodes[], edges[], node_count, edge_count }
  
graphApi.stats()
  Returns: { total_nodes, total_edges, node_types, edge_types, ... }
```

#### VectorsPage Endpoints:
```typescript
dbApi.vector.getChunks(limit, offset, documentId?)
  Returns: { chunks[], total, limit, offset, has_more }
```

#### ChatPage Endpoints:
```typescript
chatApi.queryDetailed(message, conversationId?)
  Returns: { 
    response, 
    citations, 
    workflow_metadata: { retrieved_nodes_count, response_time_ms, conversation_id },
    graph_visualization: { nodes[], edges[] }
  }
```

### API Base URL Configuration:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
```

---

## 5. State Management Approach

### No Global State Library
The application uses **React hooks for local state**:
- `useState` for component state
- `useRef` for direct DOM access (graph containers)
- `useEffect` for side effects and data loading

### TanStack React Query Integration
- Imported but minimally used in current implementation
- Created in App.tsx: `const queryClient = new QueryClient()`
- Available for refactoring to better handle API caching

### Example from GraphPage:
```typescript
const [graphData, setGraphData] = useState<{ nodes: any[]; edges: any[] }>({ nodes: [], edges: [] })
const [isLoading, setIsLoading] = useState(true)
const networkRef = useRef<HTMLDivElement>(null)

useEffect(() => {
  loadGraph() // Fetch from API
}, [])

useEffect(() => {
  if (!networkRef.current || graphData.nodes.length === 0) return
  const network = new Network(networkRef.current, graphData, options)
  return () => network.destroy()
}, [graphData])
```

---

## 6. Graph Visualization Libraries

### Current Implementation
- **Library**: vis-network (v10.0.2)
- **Data Format**: vis-network compatible (nodes with {id, label, group, ...}, edges with {from, to, ...})

### Node Structure:
```typescript
{
  id: number,
  label: string,
  group: string,        // Entity type: document, person, organization, etc.
  title: string,        // Hover text
  confidence?: number,
  degree?: number
}
```

### Edge Structure:
```typescript
{
  from: number,
  to: number,
  label: string,        // Relationship type
  title: string
}
```

### Styling
- Uses dynamic color generation based on node types
- Color map for predefined types (document, person, organization, location, etc.)
- Hash-based color generation for unknown types
- Theme-aware colors (dark/light mode support)

### Example Configuration:
```typescript
const options = {
  nodes: {
    shape: "dot",
    size: 25,
    font: { size: 14, color: isDarkMode ? "#f9fafb" : "#111827" },
    borderWidth: 2,
    shadow: true,
  },
  edges: {
    width: 2,
    shadow: true,
    smooth: { enabled: true, type: "continuous", roundness: 0.5 }
  },
  physics: {
    stabilization: true,
    barnesHut: { gravitationalConstant: -2000, springLength: 200 }
  }
}
```

---

## 7. Tab Implementation Pattern

### Current Pattern (DataSourcesPage)
Uses `@radix-ui/react-tabs`:

```typescript
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

<Tabs value={activeTab} onValueChange={setActiveTab}>
  <TabsList>
    <TabsTrigger value="all">All Sources</TabsTrigger>
    <TabsTrigger value="file">Files</TabsTrigger>
    <TabsTrigger value="webpage">Web Pages</TabsTrigger>
  </TabsList>
  
  <TabsContent value={activeTab}>
    {/* Content based on activeTab */}
  </TabsContent>
</Tabs>
```

### Features:
- Built on Radix UI (headless, unstyled)
- Full keyboard navigation support
- Smooth transitions
- Full screen responsive

---

## 8. Component Import Patterns

### UI Components:
```typescript
// Always use absolute imports with @/
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
```

### Icons:
```typescript
import { ChevronDown, Loader2, Send, Bot, User, ... } from "lucide-react"
```

### Utilities:
```typescript
import { useToast } from "@/hooks/use-toast"
import { cn } from "@/lib/utils"
```

---

## 9. Responsive Design Pattern

All pages follow mobile-first responsive design:

```typescript
// Classes used throughout
"hidden md:flex"           // Hidden on mobile, flex on md+
"grid-cols-1 md:grid-cols-2 lg:grid-cols-3"  // Responsive columns
"pb-20 md:pb-0"           // Mobile navigation spacing
"lg:col-span-3"           // Column spanning
"flex-1 lg:w-96"          // Flexible widths
```

---

## 10. Theme Management

**Location**: `/home/aub/boo/MIRAGE/ui/src/components/Layout.tsx`

```typescript
const [isDark, setIsDark] = useState(() => {
  const saved = localStorage.getItem("theme")
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches
  return saved === "dark" || (!saved && prefersDark)
})

useEffect(() => {
  if (isDark) {
    document.documentElement.classList.add("dark")
    localStorage.setItem("theme", "dark")
  } else {
    document.documentElement.classList.remove("dark")
    localStorage.setItem("theme", "light")
  }
}, [isDark])
```

- Theme stored in localStorage
- Applied to document root element
- TailwindCSS dark mode support via `dark` class

---

## 11. Critical Files for Your Implementation

| Feature | File Path | Key Component |
|---------|-----------|---------------|
| 3-column Vectorstores | `/ui/src/pages/VectorsPage.tsx` | Grid layout change |
| Graph Visualization | `/ui/src/pages/GraphPage.tsx` | Network component |
| Communities Tab | `/ui/src/pages/GraphPage.tsx` | New tab structure |
| Chat Tab | `/ui/src/pages/ChatPage.tsx` | Existing, needs expansion |
| API Calls | `/ui/src/lib/api.ts` | graphApi, dbApi, chatApi |
| UI Components | `/ui/src/components/ui/` | Reusable shadcn components |
| Layout/Navigation | `/ui/src/components/Layout.tsx` | Main layout wrapper |

---

## 12. Quick Reference: Key Types & Interfaces

### GraphPage:
```typescript
interface Node {
  id: number
  label: string
  group: string
  title: string
  confidence?: number
}

interface Edge {
  from: number
  to: number
  label: string
  title: string
}
```

### VectorsPage:
```typescript
interface VectorChunk {
  id: string
  document_id: string
  chunk_index: number
  text: string
  char_count: number
  word_count: number
  sentence_count: number
  metadata: Record<string, any>
}
```

### ChatPage:
```typescript
interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: string[]
  timestamp: Date
  retrievedNodes?: number
  responseTime?: number
}
```

---

## 13. Development Server Configuration

**Vite Config**: `/home/aub/boo/MIRAGE/ui/vite.config.ts`

```typescript
server: {
  host: "::",
  port: 8080,
}

resolve: {
  alias: {
    "@": path.resolve(__dirname, "./src"),
  },
}
```

- Runs on `http://localhost:8080`
- Path alias `@` points to `src/`

---

## 14. Environment Variables

Required in `.env` file (or via VITE_ prefix):

```
VITE_API_BASE_URL=http://localhost:8000
```

Default fallback: `http://localhost:8000`

---

## 15. Performance Considerations

### vis-network Performance Notes:
- Current limit for graph visualization: 1000 nodes (in GraphPage)
- Physics simulation enabled but can be disabled for better performance
- Networks are destroyed on component unmount to prevent memory leaks

### Rendering Optimization:
- useRef used for DOM-heavy libraries (vis-network)
- useEffect dependencies carefully managed to prevent unnecessary recreations
- Card components used throughout for consistent styling

---

## 16. Accessibility Features

- ARIA attributes in shadcn/ui components (Radix UI based)
- Keyboard navigation support (tabs, buttons)
- Theme toggle for dark mode support
- RTL support in ChatPage (isRTL toggle)
- Proper semantic HTML structure

---

## Summary for Implementation

### For Vectorstores Tab Redesign (3 blocks per row):
1. Change grid from `space-y-4` to `grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4`
2. The Card components remain unchanged
3. Keep pagination as-is

### For Graph Visualization with Communities Tab:
1. Add Tabs container in GraphPage
2. Keep current Network component as first tab
3. Add Community detection visualization as second tab
4. Call new API endpoint for community data (if available)

### For Conversation Tab with Explainability:
1. Add tab structure to right panel in ChatPage
2. Keep current "Active Context" as first tab
3. Add "Communities" tab showing community breakdown
4. Add "Retrieved Context" tab for detailed node/edge/citation info
5. Enhance workflow_metadata display from API

---

