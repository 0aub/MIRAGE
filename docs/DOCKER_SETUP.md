# MIRAGE Docker Setup Guide

## Quick Start

### 1. Set up environment variables
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

### 2. Start the services

**Start only the databases (for now):**
```bash
docker-compose up neo4j qdrant redis
```

**Start UI only:**
```bash
docker-compose up ui
```

**Start everything (when backend is ready):**
```bash
docker-compose up -d
```

### 3. Access the services

- **UI**: http://localhost:3000
- **Neo4j Browser**: http://localhost:7474
- **Qdrant API**: http://localhost:6333
- **Backend API** (when ready): http://localhost:8000

## Development Workflow

### UI Development with Hot Reload

The UI service mounts your local `./ui` directory into the container, so changes are reflected immediately:

```bash
# Start UI in development mode
docker-compose up ui

# The Vite dev server will automatically reload on file changes
```

### Changing the UI Port

Edit your `.env` file:
```
UI_PORT=3001
```

Then restart:
```bash
docker-compose up -d ui
```

## How the UI Connects to Backend

The UI uses environment variables to know where the backend is:

**Inside Docker Network:**
```
VITE_API_BASE_URL=http://api:8000
```

**For local development (backend running locally):**
```
VITE_API_BASE_URL=http://localhost:8000
```

When you start implementing API calls in your React components, use:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

// Example API call
fetch(`${API_BASE_URL}/document/upload`, {
  method: 'POST',
  body: formData
})
```

## Useful Commands

```bash
# View logs
docker-compose logs -f ui

# Rebuild UI after package.json changes
docker-compose build ui
docker-compose up -d ui

# Stop all services
docker-compose down

# Remove all data (careful!)
docker-compose down -v

# Access UI container shell
docker-compose exec ui sh
```

## Troubleshooting

**Port already in use:**
Change the port in `.env`:
```
UI_PORT=3001
```

**UI not updating after code changes:**
```bash
docker-compose restart ui
```

**Need to reinstall dependencies:**
```bash
docker-compose exec ui npm install
docker-compose restart ui
```

## Next Steps

1. Create backend services (FastAPI)
2. Uncomment the `api` service in docker-compose.yml
3. Update UI components to make real API calls
4. Test end-to-end flow
