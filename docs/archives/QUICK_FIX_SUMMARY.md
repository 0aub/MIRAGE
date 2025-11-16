# MIRAGE UI Docker Fix - Complete ✅

## Problem
The UI was trying to connect to `http://mirage:8000` instead of `http://localhost:8000`.

## Root Cause
The `docker-compose.yml` had `VITE_API_BASE_URL: http://mirage:${API_PORT:-8000}` as an environment variable, which works for Docker-to-Docker communication but not for browser-to-Docker communication.

## Solution Applied

### 1. Updated `docker-compose.yml`
**Removed** the hardcoded `VITE_API_BASE_URL` environment variable and now let Vite read from `.env.local`:

```yaml
# React UI
ui:
  build: ./ui
  container_name: mirage-ui
  environment:
    PORT: ${UI_PORT:-3000}
    # API connection - configured in .env.local file (mounted below)
  volumes:
    - ./ui:/app
    - /app/node_modules
    - ./ui/.env.local:/app/.env.local  # Mounted .env.local
```

### 2. Created `.env.local`
File: `/home/aub/boo/MIRAGE/ui/.env.local`
```env
VITE_API_BASE_URL=http://localhost:8000
```

### 3. Recreated Container
```bash
docker compose up -d --force-recreate ui
```

## Current Status

✅ **UI Container**: Running on port 3000
✅ **Environment Variable**: Removed from docker-compose, using .env.local
✅ **API URL**: Now correctly set to `http://localhost:8000`
✅ **Vite**: Running and ready

## Test It

1. **Open your browser**: http://localhost:3000
2. **Refresh the page** (Ctrl+Shift+R for hard refresh)
3. **Check browser console**: Should now see `GET http://localhost:8000/...` instead of `http://mirage:8000/...`

## Expected Behavior

- **Dashboard**: Shows real stats from backend
- **Data Sources**: Shows uploaded files
- **Graph**: Shows knowledge graph (if documents processed)
- **No errors**: About connecting to "mirage"

## If Still Not Working

**Hard refresh your browser**:
- Chrome/Edge: `Ctrl + Shift + R`
- Firefox: `Ctrl + F5`
- Safari: `Cmd + Shift + R`

**Clear browser cache** for localhost:3000

## Backend Must Be Running

Ensure your backend is also running:
```bash
# Check if running
docker ps | grep mirage-api

# Or if running directly:
curl http://localhost:8000/
```

Expected response:
```json
{
  "name": "MIRAGE API",
  "version": "0.1.0",
  "status": "operational"
}
```

## Files Changed

1. ✅ `/home/aub/boo/MIRAGE/docker-compose.yml` - Removed VITE_API_BASE_URL env var
2. ✅ `/home/aub/boo/MIRAGE/ui/.env.local` - Created with localhost:8000

---

**Status**: Ready to use! Refresh your browser and the UI should now connect to the backend successfully.
