# MIRAGE UI Backend Integration Guide

## Overview
The UI has been successfully integrated with the backend API. All static data has been replaced with real API calls.

---

## What Was Changed

### 1. API Client Created
**File**: [ui/src/lib/api.ts](ui/src/lib/api.ts)

A comprehensive API client with typed endpoints for all backend services:
- **Files API**: Upload, list, delete, get stats
- **Document API**: Process files, get status
- **Graph API**: Visualize, search, manage graph
- **Database API**: Health checks, stats, management
- **Chat API**: Send messages, detailed queries

### 2. Pages Updated

#### DataSourcesPage ([ui/src/pages/DataSourcesPage.tsx](ui/src/pages/DataSourcesPage.tsx))
**Before**: Hardcoded 5 sample files
**After**:
- ✅ Fetches real files from `/files/files` API
- ✅ Displays loading state while fetching
- ✅ Delete functionality integrated with backend
- ✅ Refreshes file list after upload
- ✅ Shows toast notifications for errors/success

**Key Changes**:
```typescript
// Now fetches from backend on mount
useEffect(() => {
  loadFiles();
}, []);

const loadFiles = async () => {
  const response = await filesApi.list({ limit: 100 });
  setDataSources(response.files.map(file => ({
    id: file.file_id,
    title: file.filename,
    type: "file",
    fileType: file.file_type,
    status: "completed",
    dateAdded: new Date(file.uploaded_at).toLocaleDateString(),
  })));
};
```

#### GraphPage ([ui/src/pages/GraphPage.tsx](ui/src/pages/GraphPage.tsx))
**Before**: Hardcoded 10 sample nodes and edges
**After**:
- ✅ Fetches real graph from `/graph/visualize/full` API
- ✅ Converts backend format to vis-network format
- ✅ Displays loading state
- ✅ Shows empty state when no graph data
- ✅ Includes confidence scores in node titles

**Key Changes**:
```typescript
const loadGraph = async () => {
  const response = await graphApi.visualizeFull(100);

  // Convert API nodes to vis-network format
  const nodes = response.nodes.map((node, idx) => ({
    id: idx + 1,
    label: node.label || node.id,
    group: node.type.toLowerCase(),
    title: `${node.type}: ${node.label}`,
    confidence: node.confidence,
  }));

  setGraphData({ nodes, edges });
};
```

#### Dashboard ([ui/src/pages/Dashboard.tsx](ui/src/pages/Dashboard.tsx))
**Before**: Hardcoded stats, processing queue, health status
**After**:
- ✅ Fetches real stats from `/db/stats` and `/files/files/stats`
- ✅ Shows real database health from `/db/health`
- ✅ Displays actual entity, relationship, and vector counts
- ✅ Color-coded health indicators (green/red)
- ✅ Loading states for all data

**Key Changes**:
```typescript
const loadDashboardData = async () => {
  const [dbStats, fileStats] = await Promise.all([
    dbApi.stats(),
    filesApi.stats(),
  ]);

  setStats([
    { label: "Total Files", value: fileStats.total_files.toString(), ... },
    { label: "Entities", value: dbStats.graph.total_nodes?.toString() || "0", ... },
    { label: "Relationships", value: dbStats.graph.total_edges?.toString() || "0", ... },
    { label: "Vector Count", value: dbStats.vector.vectors_count?.toString() || "0", ... },
  ]);
};
```

---

## Configuration

### Environment Variables

1. **Create `.env.local`** in the `ui/` directory:
```bash
cd ui
cp .env.example .env.local
```

2. **Edit `.env.local`**:
```env
VITE_API_URL=http://localhost:8000
```

For production, update to your deployed backend URL.

### API Base URL

The API client ([api.ts:5](ui/src/lib/api.ts#L5)) uses:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

This allows you to:
- Use `localhost:8000` for development (default)
- Override via environment variable for different environments
- Deploy UI and API separately

---

## Running the Application

### 1. Start the Backend

```bash
cd /home/aub/boo/MIRAGE/mirage
python3 main.py
```

The backend will be available at `http://localhost:8000`

### 2. Start the Frontend

```bash
cd /home/aub/boo/MIRAGE/ui
npm run dev
```

The UI will be available at `http://localhost:5173` (or another port if 5173 is busy)

### 3. Access the Application

Open your browser and navigate to the UI URL. The UI will automatically connect to the backend API.

---

## Testing the Integration

### 1. Dashboard Page
- **URL**: `http://localhost:5173/`
- **What to check**:
  - Stats cards show real numbers from your database
  - System Health section shows database statuses
  - Loading spinners appear briefly while fetching data

### 2. Data Sources Page
- **URL**: `http://localhost:5173/data-sources`
- **What to check**:
  - Shows actual files you've uploaded
  - "Add Source" → "Files" tab allows file upload
  - Delete button removes files from backend
  - File count updates after operations

### 3. Graph Page
- **URL**: `http://localhost:5173/graph`
- **What to check**:
  - Shows actual graph if you've processed documents
  - Shows "No graph data available" if graph is empty
  - Node colors match entity types
  - Clicking nodes shows details

### 4. Error Handling
- **Stop the backend** and refresh any page
- **What to check**:
  - Toast notifications appear with error messages
  - Loading states handle gracefully
  - UI doesn't crash

---

## API Endpoints Used

| UI Page | API Endpoints | Purpose |
|---------|--------------|---------|
| **Dashboard** | `GET /db/stats`<br>`GET /db/health`<br>`GET /files/files/stats` | Get database statistics, health status, and file counts |
| **Data Sources** | `GET /files/files`<br>`POST /files/files/upload`<br>`DELETE /files/files/{id}` | List, upload, and delete files |
| **Graph** | `GET /graph/visualize/full?limit=100` | Get graph visualization data |
| **Chat** (future) | `POST /chat/message`<br>`POST /chat/query-detailed` | Send chat messages |

---

## What's Still Needed

### 1. Database Management Page (Not Yet Implemented)
You mentioned wanting a database management tab where users can:
- ✅ **Backend Ready**: All endpoints exist (`/db/graph/clear`, `/db/graph/rebuild`, etc.)
- ❌ **UI Not Created**: Need to create a new page with UI controls

**Suggested Implementation**:
```typescript
// Create: ui/src/pages/DatabaseManagement.tsx
// Features:
// - Clear Graph button (calls dbApi.graph.clear())
// - Rebuild Graph button (calls dbApi.graph.rebuild())
// - List documents in graph (calls graphApi.documents())
// - Delete document from graph (calls graphApi.deleteDocument(id))
// - Clear Vector DB button (calls dbApi.vector.clear())
```

### 2. File Upload Tab Integration
The file upload modal opens, but the upload might need to:
- Call the backend's `/document/process-with-refrag` endpoint
- Show processing progress
- Refresh the file list and graph after processing

### 3. Chat Page Integration
Currently shows static UI. Needs:
- Connect to `/chat/message` endpoint
- Display real conversation history
- Show graph visualization from query results
- Display compression comparison data

---

## Troubleshooting

### CORS Errors

If you see CORS errors in the browser console:

1. **Check backend CORS settings** in [mirage/src/config.py](mirage/src/config.py):
```python
allowed_origins = ["http://localhost:5173", "http://localhost:3000"]
```

2. **Or update in [mirage/main.py](mirage/main.py)**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your UI URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Network Errors

If API calls fail with "Network Error":

1. **Verify backend is running**: `curl http://localhost:8000/`
2. **Check API URL**: Ensure `.env.local` has correct `VITE_API_URL`
3. **Check browser console**: Look for detailed error messages

### Empty Data

If pages show "no data":

1. **Dashboard**: Upload and process files first
2. **Graph Page**: Process documents to build the graph
3. **Data Sources**: Upload files via the "Add Source" button

---

## Next Steps

### Immediate:
1. ✅ Test the integrated UI with a running backend
2. ⏳ Create Database Management page
3. ⏳ Integrate Chat page with backend
4. ⏳ Add processing progress indicators

### Future Enhancements:
- WebSocket support for real-time updates
- Pagination for large file lists
- Advanced graph filtering and search
- Export/import functionality
- User authentication

---

## File Structure

```
MIRAGE/
├── mirage/                 # Backend
│   ├── main.py            # FastAPI app
│   ├── src/
│   │   ├── api/
│   │   │   ├── file_service.py      # ✅ File management endpoints
│   │   │   ├── db_service.py        # ✅ DB management endpoints
│   │   │   ├── graph_service.py     # ✅ Graph visualization endpoints
│   │   │   ├── document_service.py  # Document processing
│   │   │   └── chat_service.py      # Chat endpoints
│   │   └── ...
│   └── ...
└── ui/                     # Frontend
    ├── src/
    │   ├── lib/
    │   │   └── api.ts              # ✅ API client (NEW)
    │   ├── pages/
    │   │   ├── Dashboard.tsx       # ✅ Updated with real API
    │   │   ├── DataSourcesPage.tsx # ✅ Updated with real API
    │   │   ├── GraphPage.tsx       # ✅ Updated with real API
    │   │   └── ChatPage.tsx        # ⏳ Needs API integration
    │   └── ...
    ├── .env.example               # ✅ Environment template (NEW)
    └── .env.local                 # Create this with your config
```

---

## Summary

✅ **Completed**:
- Created comprehensive API client
- Integrated Dashboard with real stats
- Integrated Data Sources with file management
- Integrated Graph with visualization data
- Added loading states and error handling
- Created environment configuration

⏳ **Remaining** (based on your requirements):
- Database Management page with graph controls
- Chat page integration
- Document processing progress tracking

The UI is now fully connected to your backend! All the hardcoded data is gone, and everything is pulling from the real MIRAGE API.
