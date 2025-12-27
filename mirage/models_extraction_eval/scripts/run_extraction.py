#!/usr/bin/env python3
"""
Graph Extraction Model Evaluation Script

Runs entity and relationship extraction on source documents using specified models.
Supports Ollama and TGI providers.

Usage:
    python run_extraction.py --model llama3:8b --provider ollama
    python run_extraction.py --model Qwen/Qwen2.5-7B-Instruct --provider tgi --endpoint http://localhost:8080
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import pymupdf
import tiktoken
import unicodedata


# =============================================================================
# Configuration
# =============================================================================

MODELS_CONFIG = {
    # Text-based models (using Ollama model names)
    "allam-7b": {
        "full_name": "iKhalid/ALLaM:7b",  # Community version from Ollama
        "provider": "ollama",
        "vision": False,
        "hf_name": "ALLaM-AI/ALLaM-7B-Instruct-preview"
    },
    "qwen3-4b-thinking": {
        "full_name": "qwen3:4b",  # Available as qwen3 or need specific tag
        "provider": "ollama",
        "vision": False,
        "hf_name": "Qwen/Qwen3-4B-Thinking-2507"
    },
    "qwen2.5-7b-instruct": {
        "full_name": "qwen2.5:7b",  # Already available
        "provider": "ollama",
        "vision": False,
        "hf_name": "Qwen/Qwen2.5-7B-Instruct"
    },
    "qwen2.5-coder-7b": {
        "full_name": "qwen2.5-coder:7b",  # Need to pull
        "provider": "ollama",
        "vision": False,
        "hf_name": "Qwen/Qwen2.5-Coder-7B-Instruct"
    },
    "mistral-7b-v0.2": {
        "full_name": "mistral:7b",  # Need to pull
        "provider": "ollama",
        "vision": False,
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "llama3-8b": {
        "full_name": "llama3:8b",  # Need to pull
        "provider": "ollama",
        "vision": False,
        "hf_name": "meta-llama/Meta-Llama-3-8B-Instruct"
    },
    # Vision models
    "gemma-3n-e4b-text": {
        "full_name": "gemma3:4b",  # Available - text mode test
        "provider": "ollama",
        "vision": False,
        "hf_name": "google/gemma-3n-E4B-it"
    },
    "gemma-3n-e4b-vision": {
        "full_name": "gemma3:4b",  # Vision mode test
        "provider": "ollama",
        "vision": True,
        "hf_name": "google/gemma-3n-E4B-it"
    },
    "gemma2-9b-text": {
        "full_name": "gemma2:9b",  # Available - additional test
        "provider": "ollama",
        "vision": False,
        "hf_name": "google/gemma-2-9b-it"
    },
    "gemma2-9b-vision": {
        "full_name": "gemma2:9b",  # Vision mode test
        "provider": "ollama",
        "vision": True,
        "hf_name": "google/gemma-2-9b-it"
    },
}

MAX_TOKENS_PER_CHUNK = 600  # GraphRAG recommended size


# =============================================================================
# Prompts
# =============================================================================

def get_extraction_prompt(language: str) -> Tuple[str, str]:
    """Get system and user prompt templates for entity extraction."""
    if language == "ar":
        system_prompt = """أنت مستخرج معرفة متخصص لبناء رسم بياني معرفي (Knowledge Graph).
مهمتك: استخراج الكيانات المسماة والعلاقات بينها. العلاقات مهمة جداً!

**أنواع الكيانات:**
- Organization: منظمات، شركات، مؤسسات، جهات حكومية
- Person: أشخاص، مناصب، أدوار
- Location: أماكن، مدن، دول، مناطق
- Concept: مفاهيم، مصطلحات، تعريفات
- Event: أحداث، مناسبات، تواريخ مهمة
- Product: منتجات، خدمات، أنظمة
- Document: وثائق، سياسات، قوانين، تقارير

**أنواع العلاقات (مهم جداً!):**
- RELATED_TO: علاقة عامة بين كيانين
- PART_OF: كيان جزء من كيان أكبر
- CONTAINS: كيان يحتوي على كيان آخر
- CAUSES: كيان يسبب شيء آخر
- USES: كيان يستخدم كيان آخر
- CREATED_BY: كيان أنشأه كيان آخر
- LOCATED_IN: كيان موجود في مكان
- BELONGS_TO: كيان ينتمي لكيان آخر
- DEPENDS_ON: كيان يعتمد على كيان آخر
- AFFECTS: كيان يؤثر على كيان آخر

**صيغة JSON:**
{
  "entities": [{"text": "اسم", "type": "النوع", "importance": "high/medium/low"}],
  "relationships": [{"source": "المصدر", "target": "الهدف", "type": "RELATIONSHIP_TYPE", "weight": 0.8}]
}

مهم: استخرج علاقات بين الكيانات! لا تستخرج كيانات بدون علاقات."""

        user_template = """استخرج الكيانات والعلاقات بينها:

{chunk}

JSON:"""

    else:  # English
        system_prompt = """You are a knowledge extractor for building a Knowledge Graph.
Extract named entities AND relationships between them. Relationships are critical!

**Entity Types:**
- Organization: Companies, institutions, agencies, departments
- Person: People, roles, positions, titles
- Location: Places, cities, countries, regions
- Concept: Ideas, terms, definitions, topics
- Event: Events, dates, milestones, occurrences
- Product: Products, services, systems, tools
- Document: Documents, policies, laws, reports, papers

**Relationship Types (CRITICAL!):**
- RELATED_TO: General association between entities
- PART_OF: Entity is component/member of another
- CONTAINS: Entity contains another entity
- CAUSES: Entity causes/leads to something
- USES: Entity uses/utilizes another
- CREATED_BY: Entity was created by another
- LOCATED_IN: Entity is located in a place
- BELONGS_TO: Entity belongs to another
- DEPENDS_ON: Entity depends on another
- AFFECTS: Entity affects/impacts another

**JSON format:**
{
  "entities": [{"text": "Name", "type": "Type", "importance": "high/medium/low"}],
  "relationships": [{"source": "Source", "target": "Target", "type": "RELATIONSHIP_TYPE", "weight": 0.8}]
}

IMPORTANT: Extract relationships between entities! Do not return entities without relationships."""

        user_template = """Extract entities AND relationships:

{chunk}

JSON:"""

    return system_prompt, user_template


# =============================================================================
# PDF Processing
# =============================================================================

def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    """Extract text from PDF with proper Arabic handling."""
    doc = pymupdf.open(file_path)

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks", sort=True)

        text_parts = []
        for block in blocks:
            if len(block) > 4 and block[4].strip():
                text_parts.append(block[4])

        text = "\n".join(text_parts)
        text = unicodedata.normalize('NFC', text)

        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text
            })

    doc.close()
    return {"pages": pages, "total_pages": len(pages)}


def extract_images_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract page images from PDF for vision models."""
    import base64
    from io import BytesIO

    doc = pymupdf.open(file_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to image
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode()

        images.append({
            "page_number": page_num + 1,
            "image_base64": img_b64
        })

    doc.close()
    return images


# =============================================================================
# Text Chunking
# =============================================================================

def count_tokens(text: str, encoding=None) -> int:
    """Count tokens in text."""
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def chunk_text(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> List[str]:
    """Split text into chunks that fit within token limits."""
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = count_tokens(text, encoding)

    if total_tokens <= max_tokens:
        return [text]

    # Split by sentences
    text_marked = text
    for sep in ['. ', '.\n', '! ', '? ', '، ', '؛ ']:
        text_marked = text_marked.replace(sep, sep.rstrip() + '|')
    sentences = [s.strip() for s in text_marked.split('|') if s.strip()]

    if len(sentences) <= 1:
        # No sentence boundaries - split by words
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0

        for word in words:
            word_tokens = count_tokens(word + ' ', encoding)
            if current_tokens + word_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-50:] if len(current_chunk) > 50 else []
                current_tokens = count_tokens(' '.join(current_chunk), encoding)
            current_chunk.append(word)
            current_tokens += word_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_tokens = 0
    overlap_sentences = 2

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, encoding)
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []
            current_tokens = sum(count_tokens(s, encoding) for s in current_chunk)

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# =============================================================================
# LLM Calling
# =============================================================================

def call_ollama(endpoint: str, model: str, system_prompt: str, user_prompt: str) -> str:
    """Call Ollama API."""
    response = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_ollama_vision(endpoint: str, model: str, system_prompt: str, user_prompt: str, image_b64: str) -> str:
    """Call Ollama API with image."""
    response = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        timeout=180
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_tgi(endpoint: str, system_prompt: str, user_prompt: str) -> str:
    """Call TGI API."""
    response = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": "tgi",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# =============================================================================
# JSON Parsing
# =============================================================================

def parse_json_response(content: str) -> Dict[str, Any]:
    """Parse JSON from LLM response with repair attempts."""
    # Try to find JSON in the response
    content = content.strip()

    # Remove markdown code blocks
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    # Find JSON object
    start = content.find("{")
    end = content.rfind("}") + 1

    if start >= 0 and end > start:
        json_str = content[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to repair common issues
            json_str = json_str.replace("'", '"')
            json_str = json_str.replace("True", "true")
            json_str = json_str.replace("False", "false")
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    return {"entities": [], "relationships": []}


# =============================================================================
# Main Extraction
# =============================================================================

def detect_language(text: str) -> str:
    """Detect if text is primarily Arabic or English."""
    arabic_chars = len([c for c in text if '\u0600' <= c <= '\u06FF'])
    latin_chars = len([c for c in text if 'a' <= c.lower() <= 'z'])
    return "ar" if arabic_chars > latin_chars else "en"


def extract_from_document(
    pdf_path: str,
    model_name: str,
    model_config: Dict[str, Any],
    endpoint: str,
    output_dir: str
) -> Dict[str, Any]:
    """Extract entities and relationships from a document."""

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"Model: {model_config['full_name']}")
    print(f"{'='*60}")

    start_time = time.time()

    # Extract text/images from PDF
    if model_config.get("vision"):
        print("Extracting page images...")
        pages_data = extract_images_from_pdf(pdf_path)
    else:
        print("Extracting text...")
        doc_data = extract_text_from_pdf(pdf_path)
        pages_data = doc_data["pages"]

    all_entities = []
    all_relationships = []
    errors = []
    raw_outputs = []
    total_tokens = 0

    # Detect language from first page
    if not model_config.get("vision"):
        first_text = pages_data[0]["text"] if pages_data else ""
        language = detect_language(first_text)
    else:
        language = "en"  # Default for vision

    system_prompt, user_template = get_extraction_prompt(language)

    print(f"Language detected: {language}")
    print(f"Total pages: {len(pages_data)}")

    for i, page_data in enumerate(pages_data):
        page_num = page_data.get("page_number", i + 1)
        print(f"\rProcessing page {page_num}/{len(pages_data)}...", end="", flush=True)

        try:
            if model_config.get("vision"):
                # Vision mode - process page image
                user_prompt = user_template.format(chunk="[See attached image]")
                content = call_ollama_vision(
                    endpoint,
                    model_config["full_name"],
                    system_prompt,
                    user_prompt,
                    page_data["image_base64"]
                )
                total_tokens += 1000  # Estimate for images
            else:
                # Text mode - chunk the text
                page_text = page_data["text"]
                chunks = chunk_text(page_text)

                for chunk in chunks:
                    user_prompt = user_template.format(chunk=chunk)
                    total_tokens += count_tokens(chunk)

                    if model_config["provider"] == "ollama":
                        content = call_ollama(
                            endpoint,
                            model_config["full_name"],
                            system_prompt,
                            user_prompt
                        )
                    else:
                        content = call_tgi(endpoint, system_prompt, user_prompt)

                    # Parse response
                    result = parse_json_response(content)

                    # Add source info
                    for entity in result.get("entities", []):
                        entity["source_doc"] = os.path.basename(pdf_path)
                        entity["page"] = page_num
                        all_entities.append(entity)

                    for rel in result.get("relationships", []):
                        rel["source_doc"] = os.path.basename(pdf_path)
                        rel["page"] = page_num
                        all_relationships.append(rel)

                    raw_outputs.append({
                        "page": page_num,
                        "chunk_preview": chunk[:200],
                        "response": content,
                        "parsed": result
                    })

        except Exception as e:
            errors.append({
                "page": page_num,
                "error": str(e)
            })
            print(f"\n  Error on page {page_num}: {e}")

    print()  # New line after progress

    end_time = time.time()
    elapsed = end_time - start_time

    # Calculate stats
    stats = {
        "model": model_name,
        "model_full_name": model_config["full_name"],
        "provider": model_config["provider"],
        "vision_mode": model_config.get("vision", False),
        "document": os.path.basename(pdf_path),
        "language": language,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_pages": len(pages_data),
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / elapsed, 2) if elapsed > 0 else 0,
        "pages_per_minute": round(len(pages_data) / (elapsed / 60), 2) if elapsed > 0 else 0,
        "total_entities": len(all_entities),
        "total_relationships": len(all_relationships),
        "errors_count": len(errors),
        "errors": errors
    }

    print(f"\nExtraction complete!")
    print(f"  Entities: {len(all_entities)}")
    print(f"  Relationships: {len(all_relationships)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Speed: {stats['tokens_per_second']} tokens/sec")

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "stats": stats,
        "raw_outputs": raw_outputs
    }


def save_results(output_dir: str, doc_name: str, results: Dict[str, Any]):
    """Save extraction results to files."""
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw_outputs")
    os.makedirs(raw_dir, exist_ok=True)

    # Determine file prefix based on document
    prefix = "en_" if "En" in doc_name else "ar_"

    # Save entities
    with open(os.path.join(output_dir, f"{prefix}entities.json"), "w", encoding="utf-8") as f:
        json.dump(results["entities"], f, ensure_ascii=False, indent=2)

    # Save relationships
    with open(os.path.join(output_dir, f"{prefix}relationships.json"), "w", encoding="utf-8") as f:
        json.dump(results["relationships"], f, ensure_ascii=False, indent=2)

    # Save stats
    with open(os.path.join(output_dir, f"{prefix}extraction_log.json"), "w", encoding="utf-8") as f:
        json.dump(results["stats"], f, ensure_ascii=False, indent=2)

    # Save raw outputs
    with open(os.path.join(raw_dir, f"{prefix}raw_outputs.json"), "w", encoding="utf-8") as f:
        json.dump(results["raw_outputs"], f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Graph Extraction Model Evaluation")
    parser.add_argument("--model", required=True, help="Model key from MODELS_CONFIG (e.g., llama3-8b)")
    parser.add_argument("--endpoint", default="http://localhost:11434", help="Ollama/TGI endpoint")
    parser.add_argument("--docs", nargs="+", help="PDF files to process (default: source_docs/*.pdf)")
    parser.add_argument("--output", help="Output directory (default: extractions/<model>/)")
    args = parser.parse_args()

    # Get model config
    if args.model not in MODELS_CONFIG:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(MODELS_CONFIG.keys())}")
        sys.exit(1)

    model_config = MODELS_CONFIG[args.model]

    # Set output directory
    script_dir = Path(__file__).parent.parent
    output_dir = args.output or str(script_dir / "extractions" / args.model)

    # Get documents
    if args.docs:
        docs = args.docs
    else:
        source_dir = script_dir / "source_docs"
        docs = list(source_dir.glob("*.pdf"))

    if not docs:
        print("No PDF documents found!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("GRAPH EXTRACTION MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {args.model} ({model_config['full_name']})")
    print(f"Provider: {model_config['provider']}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Vision Mode: {model_config.get('vision', False)}")
    print(f"Documents: {len(docs)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Process each document
    all_results = {}
    for doc_path in docs:
        doc_path = str(doc_path)
        doc_name = os.path.basename(doc_path)

        try:
            results = extract_from_document(
                doc_path,
                args.model,
                model_config,
                args.endpoint,
                output_dir
            )
            save_results(output_dir, doc_name, results)
            all_results[doc_name] = results["stats"]

        except Exception as e:
            print(f"\nFailed to process {doc_name}: {e}")
            all_results[doc_name] = {"error": str(e)}

    # Save combined summary
    summary_path = os.path.join(output_dir, "extraction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "model_full_name": model_config["full_name"],
            "timestamp": datetime.now().isoformat(),
            "documents": all_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
