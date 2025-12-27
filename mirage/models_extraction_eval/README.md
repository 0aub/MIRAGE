# Graph Extraction Model Evaluation

Evaluation of different LLM models for entity and relationship extraction from Arabic and English policy documents.

## Objective

Compare models on:
1. **Entity Extraction Quality** - Are entities correct and meaningful?
2. **Relationship Extraction Accuracy** - Are relationships accurate and complete?
3. **Cross-Language Linking** (bonus) - Do Arabic/English entities get linked?
4. **Speed** - Tokens/second, total extraction time

## Source Documents

| Document | Language | Pages | Description |
|----------|----------|-------|-------------|
| PoliciesEn.pdf | English | 125 | Saudi data governance policies |
| PoliciesAr.pdf | Arabic | 57 | Arabic version of policies |

## Models Under Evaluation

### Text-based Models
| Model | Provider | Size | Notes |
|-------|----------|------|-------|
| humain-ai/ALLaM-7B-Instruct-preview | HuggingFace/Ollama | 7B | Arabic-focused |
| Qwen/Qwen3-4B-Thinking-2507 | HuggingFace/Ollama | 4B | Reasoning-enhanced |
| Qwen/Qwen2.5-7B-Instruct | HuggingFace/Ollama | 7B | Strong multilingual |
| Qwen/Qwen2.5-Coder-7B-Instruct | HuggingFace/Ollama | 7B | JSON output focused |
| mistralai/Mistral-7B-Instruct-v0.2 | HuggingFace/Ollama | 7B | Fast inference |
| llama3:8b | Ollama | 8B | Meta's latest |

### Vision-capable Models (Text + Image tests)
| Model | Provider | Size | Notes |
|-------|----------|------|-------|
| google/gemma-3n-E4B-it | Ollama | 4B | Multimodal |
| google/t5gemma-2-4b-4b | Ollama | 4B | Multimodal |

## Evaluation Metrics

### Quantitative Metrics
- **Entity Count**: Total entities extracted
- **Relationship Count**: Total relationships extracted
- **Entity Type Distribution**: Breakdown by type (Organization, Person, Location, etc.)
- **Relationship Type Distribution**: Breakdown by type (PART_OF, RELATED_TO, etc.)
- **Extraction Speed**: Tokens/second, pages/minute

### Quality Metrics (Manual Review)
- **Entity Precision**: % of extracted entities that are correct
- **Entity Recall (sampled)**: % of expected entities that were found
- **Entity Quality Score**: 1-5 rating on naming accuracy, type correctness
- **Relationship Precision**: % of relationships that are accurate
- **Relationship Quality Score**: 1-5 rating on meaningfulness

### Cross-Language Metrics (Bonus)
- **Linked Entity Count**: Entities with both Arabic and English variants
- **Linking Accuracy**: % of correct cross-language links

## Folder Structure

```
models_extraction_eval/
├── source_docs/                    # Source PDF files
│   ├── PoliciesEn.pdf
│   └── PoliciesAr.pdf
├── extractions/                    # Model outputs
│   ├── allam-7b/
│   │   ├── entities.json
│   │   ├── relationships.json
│   │   ├── extraction_log.json
│   │   └── raw_outputs/
│   ├── qwen3-4b-thinking/
│   ├── qwen2.5-7b-instruct/
│   ├── qwen2.5-coder-7b/
│   ├── mistral-7b-v0.2/
│   ├── llama3-8b/
│   ├── gemma-3n-e4b-text/
│   ├── gemma-3n-e4b-vision/
│   ├── t5gemma-2-4b-text/
│   └── t5gemma-2-4b-vision/
├── evaluation/                     # Evaluation tools
│   ├── metrics.py                  # Automated metrics
│   └── manual_review_template.md   # Manual review form
├── reports/                        # Final reports
│   └── comparison_report.md
├── scripts/                        # Extraction scripts
│   └── run_extraction.py
└── README.md
```

## Running Extractions

### Step 1: Start the model
```bash
# For Ollama models
ollama pull llama3:8b
ollama run llama3:8b

# For HuggingFace models via TGI
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id Qwen/Qwen2.5-7B-Instruct
```

### Step 2: Run extraction
```bash
python scripts/run_extraction.py \
  --model llama3:8b \
  --provider ollama \
  --docs source_docs/PoliciesEn.pdf source_docs/PoliciesAr.pdf \
  --output extractions/llama3-8b/
```

### Step 3: Review results
```bash
python evaluation/metrics.py --model-dir extractions/llama3-8b/
```

## Manual Review Process

For each model, I will:

1. **Sample 20 entities** (10 English, 10 Arabic) and verify:
   - Is this a real entity from the document?
   - Is the entity name accurate?
   - Is the entity type correct?

2. **Sample 20 relationships** and verify:
   - Does this relationship exist in the document?
   - Is the relationship type appropriate?
   - Are source/target entities correct?

3. **Check cross-language linking** (if applicable):
   - Are equivalent entities linked?
   - Are any false links present?

## Expected Output Format

### entities.json
```json
[
  {
    "text": "National Data Management Office",
    "type": "Organization",
    "importance": "high",
    "confidence": 0.9,
    "source_doc": "PoliciesEn.pdf",
    "page": 5
  }
]
```

### relationships.json
```json
[
  {
    "source": "National Data Management Office",
    "target": "SDAIA",
    "type": "PART_OF",
    "weight": 0.8,
    "source_doc": "PoliciesEn.pdf"
  }
]
```

### extraction_log.json
```json
{
  "model": "llama3:8b",
  "provider": "ollama",
  "start_time": "2024-12-26T10:00:00",
  "end_time": "2024-12-26T10:30:00",
  "total_pages": 182,
  "total_tokens": 50000,
  "tokens_per_second": 28.5,
  "errors": []
}
```

## Timeline

1. **Setup**: Create scripts and prepare models
2. **Extraction**: Run each model (one at a time)
3. **Review**: Manual quality assessment
4. **Report**: Generate comparison report
