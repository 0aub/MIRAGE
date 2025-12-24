# Ollama Integration for MIRAGE

## Overview

Successfully integrated Ollama as a local LLM provider for entity extraction in MIRAGE. This allows you to use local models like Gemma3:4B instead of cloud API providers.

## What Was Done

### 1. Updated Configuration ([settings.py](../mirage/src/config/settings.py))

Added Ollama configuration options:
- `use_ollama`: Enable/disable Ollama (default: False)
- `ollama_endpoint`: Ollama API endpoint (default: `http://ollama:11434`)
- `ollama_model`: Model to use (default: `gemma3:4b`)

### 2. Updated Entity Extractor ([llm_entity_extractor.py](../mirage/src/core/graph_builder/llm_entity_extractor.py))

Added Ollama support:
- Provider detection: Ollama has highest priority when enabled
- API integration: Uses OpenAI-compatible `/v1/chat/completions` endpoint
- Error handling: Proper timeout and retry logic
- Entity validation: LLM-based validation support

###  3. Updated Docker Configuration

**[.env](../.env)**:
```bash
USE_OLLAMA=true
OLLAMA_ENDPOINT=http://ollama:11434
OLLAMA_MODEL=gemma3:4b
USE_TGI=false  # Disabled to avoid loading multiple models
```

**[docker-compose.yml](../docker-compose.yml)**:
- Added environment variables for Ollama configuration
- Ollama container already configured and running

### 4. Verification

Confirmed via logs:
```
Using Ollama at http://ollama:11434 with model gemma3:4b for entity extraction
```

## Available Models

Models currently downloaded in Ollama:
- ✅ **gemma3:4b** (3.3 GB) - Currently configured
- gemma2:9b (5.4 GB)
- gemma2:2b (1.6 GB)
- qwen2.5:7b (4.7 GB)

## How to Use

### Option 1: Using the UI (Recommended)

1. **Access the UI**: Open [http://localhost:3000](http://localhost:3000)
2. **Upload a document**: Use the file upload feature
3. **Monitor extraction**: Ollama will extract entities automatically

### Option 2: Using the API

The file upload endpoint handles document processing with Ollama automatically.

Access API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 3: Run Benchmark Tests

Use the model comparison benchmark to test Ollama quality:

```bash
cd mirage/model_comparison
python3 benchmark_runner.py
```

## Switching Models

To use a different Ollama model:

1. **Update .env**:
   ```bash
   OLLAMA_MODEL=qwen2.5:7b  # Or gemma2:9b, etc.
   ```

2. **Restart the backend**:
   ```bash
   docker compose restart mirage
   ```

3. **Verify in logs**:
   ```bash
   docker logs mirage-api | grep "Using Ollama"
   ```

## Switching Back to Cloud APIs

To use OpenAI/Anthropic/Google instead of Ollama:

1. **Disable Ollama in .env**:
   ```bash
   USE_OLLAMA=false
   ```

2. **Enable your preferred provider**:
   ```bash
   OPENAI_API_KEY=your_key  # Or ANTHROPIC_API_KEY, GOOGLE_API_KEY
   ```

3. **Restart**:
   ```bash
   docker compose restart mirage
   ```

## Provider Priority

The system checks providers in this order:
1. **Ollama** (if `USE_OLLAMA=true`)
2. **TGI** (if `USE_TGI=true`)
3. **OpenAI** (if `OPENAI_API_KEY` set)
4. **Anthropic Claude** (if `ANTHROPIC_API_KEY` set)
5. **Google Gemini** (if `GOOGLE_API_KEY` set)

## Benefits of Ollama

✅ **No API Costs**: Run models locally for free
✅ **No Rate Limits**: Process as many documents as you want
✅ **Privacy**: All data stays on your machine
✅ **Fast**: Local inference is often faster than API calls
✅ **Offline**: Works without internet connection

## Performance Notes

- **First Request**: May take 5-10 seconds (model loading)
- **Subsequent Requests**: Much faster (~1-2 seconds per chunk)
- **Memory**: gemma3:4b requires ~4GB RAM when loaded
- **GPU**: Ollama will use GPU if available (much faster)

## Troubleshooting

### Ollama Not Detected

Check logs:
```bash
docker logs mirage-api | grep -i ollama
```

Should see:
```
Using Ollama at http://ollama:11434 with model gemma3:4b for entity extraction
```

### Connection Errors

Verify Ollama is running:
```bash
docker ps | grep ollama
curl http://localhost:11434/api/tags
```

### Model Not Found

Pull the model:
```bash
docker exec mirage-ollama ollama pull gemma3:4b
```

List available models:
```bash
docker exec mirage-ollama ollama list
```

## Next Steps

Now that Ollama is integrated, you can:

1. **Test with real documents**: Upload PDFs through the UI
2. **Compare models**: Run benchmark tests on different models
3. **Optimize performance**: Try different models for quality vs. speed
4. **Scale up**: Use larger models like gemma2:9b for better quality

## Status

✅ **Integration Complete**
✅ **gemma3:4b Model Ready**
✅ **System Configured**
📊 **Ready for Testing**

---

*Generated: 2025-12-24*
*MIRAGE with Ollama - Local, Private, Unlimited RAG*
