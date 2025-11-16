# Local LLM Setup with Text Generation Inference (TGI)

Run your own LLM on your 24GB GPU - **NO rate limits, NO API costs!**

## Quick Start

### 1. Start TGI Container

```bash
# Start TGI with Qwen 2.5 14B (recommended for Arabic)
docker compose up -d tgi
```

The first time will download the model (~28GB). This takes 10-20 minutes depending on your internet speed.

### 2. Wait for Model to Load

Check the logs to see when it's ready:
```bash
docker logs -f mirage-tgi
```

Look for: `Connected` or `Ready to accept requests`

### 3. Restart Backend

```bash
docker compose restart mirage
```

Check logs to confirm TGI is being used:
```bash
docker logs -f mirage-api | grep TGI
```

You should see: `Using local TGI endpoint at http://tgi:80 for entity extraction (NO RATE LIMITS!)`

## Recommended Models for 24GB GPU

### Option 1: Qwen 2.5 14B Instruct (Default - Best Choice)
- **Why**: Excellent Arabic support, great instruction following
- **VRAM**: ~16GB
- **Model**: `Qwen/Qwen2.5-14B-Instruct`
- Already configured in docker-compose.yml

### Option 2: Qwen 2.5 32B (4-bit quantized)
- **Why**: Even better performance, fits in 24GB
- **VRAM**: ~20GB
- **Model**: `Qwen/Qwen2.5-32B-Instruct-AWQ`

To use, edit docker-compose.yml:
```yaml
environment:
  MODEL_ID: Qwen/Qwen2.5-32B-Instruct-AWQ
```

### Option 3: Llama 3.1 70B (4-bit quantized)
- **Why**: Top-tier performance
- **VRAM**: ~24GB (tight fit)
- **Model**: `neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w4a16`

## Performance Comparison

| Model | VRAM | Speed | Arabic Quality | Entities/sec |
|-------|------|-------|----------------|--------------|
| Gemini 2.0 Flash (API) | N/A | Fast | Excellent | 50 req/day limit |
| Qwen 2.5 14B (Local) | 16GB | Medium | Excellent | **Unlimited** |
| Qwen 2.5 32B (Local) | 20GB | Slower | Superior | **Unlimited** |

## Testing Your Setup

1. Go to Data Sources in UI
2. Add a new YouTube video
3. Click "Load Video" then "Get Transcript & Process"
4. Watch backend logs: `docker logs -f mirage-api`

You should see:
```
Using local TGI endpoint at http://tgi:80 for entity extraction (NO RATE LIMITS!)
Processing chunk 1/27
Processing chunk 2/27
...
```

## Troubleshooting

### TGI container exits immediately
**Issue**: GPU not available or NVIDIA runtime not installed

**Fix**: Ensure you have:
- NVIDIA drivers installed
- Docker with NVIDIA runtime: `nvidia-docker` or `nvidia-container-toolkit`

Test with:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Out of memory errors
**Issue**: Model too large for your GPU

**Fix**: Use a smaller model or quantized version. Edit docker-compose.yml:
```yaml
MODEL_ID: Qwen/Qwen2.5-7B-Instruct  # Smaller, uses ~10GB
```

### Slow generation
**Issue**: Normal for large models

**Optimization**:
1. Use Flash Attention (requires newer GPU):
   ```yaml
   USE_FLASH_ATTENTION: "true"
   ```

2. Reduce max tokens:
   ```yaml
   MAX_TOTAL_TOKENS: 4096
   ```

### Connection errors
**Issue**: TGI not started or still loading

**Fix**: Check TGI is running:
```bash
docker ps | grep tgi
curl http://localhost:8080/health
```

## Switching Between Local and Cloud

### Use Local TGI
```bash
# In .env file
USE_TGI=true
TGI_ENDPOINT=http://localhost:8080
```

### Use Cloud API (Gemini/Claude/OpenAI)
```bash
# In .env file
USE_TGI=false
# Keep your API keys
GOOGLE_API_KEY=your_key_here
```

Restart backend:
```bash
docker compose restart mirage
```

## Advanced Configuration

### Increase Batch Size (Faster Processing)
Edit docker-compose.yml:
```yaml
environment:
  MAX_BATCH_PREFILL_TOKENS: 16384  # Process more tokens at once
```

### Use Multiple GPUs
```yaml
environment:
  NUM_SHARD: 2  # Split model across 2 GPUs
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Use 2 GPUs
```

### Custom System Prompt
Edit `mirage/src/core/graph_builder/llm_entity_extractor.py` to customize extraction behavior.

## Cost Comparison

| Provider | Cost | Limit | Your 21k word video |
|----------|------|-------|---------------------|
| Gemini Flash | Free tier | 50 req/day | $0 (rate limited) |
| Claude Haiku | $0.25/M tokens | Pay as you go | ~$0.08 per video |
| GPT-4o-mini | $0.15/M tokens | Pay as you go | ~$0.05 per video |
| **TGI Local** | **Electricity** | **Unlimited** | **~$0.002 per video** |

## Getting Help

Check logs:
```bash
# TGI logs
docker logs mirage-tgi

# Backend logs
docker logs mirage-api

# All services
docker compose logs -f
```

Stop TGI to save GPU:
```bash
docker stop mirage-tgi
```

Start again:
```bash
docker start mirage-tgi
```
