#!/usr/bin/env python3
"""
Metrics Calculation for Graph Extraction Evaluation

Calculates quantitative metrics from extraction results.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_entity_metrics(entities: List[Dict]) -> Dict[str, Any]:
    """Calculate entity-related metrics."""
    if not entities:
        return {
            "total_count": 0,
            "type_distribution": {},
            "importance_distribution": {},
            "avg_per_page": 0
        }

    types = Counter(e.get("type", "Unknown") for e in entities)
    importance = Counter(e.get("importance", "unknown") for e in entities)
    pages = set(e.get("page", 0) for e in entities)

    return {
        "total_count": len(entities),
        "type_distribution": dict(types.most_common()),
        "importance_distribution": dict(importance.most_common()),
        "unique_pages": len(pages),
        "avg_per_page": round(len(entities) / len(pages), 2) if pages else 0
    }


def calculate_relationship_metrics(relationships: List[Dict]) -> Dict[str, Any]:
    """Calculate relationship-related metrics."""
    if not relationships:
        return {
            "total_count": 0,
            "type_distribution": {},
            "avg_weight": 0
        }

    types = Counter(r.get("type", "UNKNOWN") for r in relationships)
    weights = [r.get("weight", 0.5) for r in relationships if "weight" in r]
    pages = set(r.get("page", 0) for r in relationships)

    return {
        "total_count": len(relationships),
        "type_distribution": dict(types.most_common()),
        "unique_pages": len(pages),
        "avg_per_page": round(len(relationships) / len(pages), 2) if pages else 0,
        "avg_weight": round(sum(weights) / len(weights), 3) if weights else 0
    }


def calculate_cross_language_metrics(en_entities: List[Dict], ar_entities: List[Dict]) -> Dict[str, Any]:
    """Calculate cross-language linking potential."""
    # Simple heuristic: look for entities that might be translations
    # This is a rough estimate - actual linking would require more sophisticated methods

    en_names = set(e.get("text", "").lower() for e in en_entities)
    ar_names = set(e.get("text", "") for e in ar_entities)

    # Count organization entities (most likely to have cross-language equivalents)
    en_orgs = [e for e in en_entities if e.get("type") == "Organization"]
    ar_orgs = [e for e in ar_entities if e.get("type") == "Organization"]

    return {
        "english_entity_count": len(en_entities),
        "arabic_entity_count": len(ar_entities),
        "english_org_count": len(en_orgs),
        "arabic_org_count": len(ar_orgs),
        "potential_linking_candidates": min(len(en_orgs), len(ar_orgs))
    }


def analyze_model_results(model_dir: str) -> Dict[str, Any]:
    """Analyze results for a single model."""
    model_dir = Path(model_dir)

    results = {
        "model_dir": str(model_dir),
        "english": {},
        "arabic": {},
        "combined": {},
        "cross_language": {},
        "speed": {}
    }

    # Load English results
    en_entities = []
    en_relationships = []
    if (model_dir / "en_entities.json").exists():
        en_entities = load_json(model_dir / "en_entities.json")
    if (model_dir / "en_relationships.json").exists():
        en_relationships = load_json(model_dir / "en_relationships.json")

    results["english"] = {
        "entities": calculate_entity_metrics(en_entities),
        "relationships": calculate_relationship_metrics(en_relationships)
    }

    # Load Arabic results
    ar_entities = []
    ar_relationships = []
    if (model_dir / "ar_entities.json").exists():
        ar_entities = load_json(model_dir / "ar_entities.json")
    if (model_dir / "ar_relationships.json").exists():
        ar_relationships = load_json(model_dir / "ar_relationships.json")

    results["arabic"] = {
        "entities": calculate_entity_metrics(ar_entities),
        "relationships": calculate_relationship_metrics(ar_relationships)
    }

    # Combined metrics
    all_entities = en_entities + ar_entities
    all_relationships = en_relationships + ar_relationships
    results["combined"] = {
        "entities": calculate_entity_metrics(all_entities),
        "relationships": calculate_relationship_metrics(all_relationships)
    }

    # Cross-language metrics
    results["cross_language"] = calculate_cross_language_metrics(en_entities, ar_entities)

    # Load speed metrics from logs
    if (model_dir / "en_extraction_log.json").exists():
        en_log = load_json(model_dir / "en_extraction_log.json")
        results["speed"]["english"] = {
            "elapsed_seconds": en_log.get("elapsed_seconds", 0),
            "tokens_per_second": en_log.get("tokens_per_second", 0),
            "pages_per_minute": en_log.get("pages_per_minute", 0)
        }

    if (model_dir / "ar_extraction_log.json").exists():
        ar_log = load_json(model_dir / "ar_extraction_log.json")
        results["speed"]["arabic"] = {
            "elapsed_seconds": ar_log.get("elapsed_seconds", 0),
            "tokens_per_second": ar_log.get("tokens_per_second", 0),
            "pages_per_minute": ar_log.get("pages_per_minute", 0)
        }

    return results


def print_report(results: Dict[str, Any]):
    """Print a formatted report."""
    print("\n" + "="*70)
    print("MODEL EXTRACTION METRICS")
    print("="*70)

    print(f"\nModel: {Path(results['model_dir']).name}")

    # English metrics
    print("\n--- ENGLISH ---")
    en = results["english"]
    print(f"  Entities: {en['entities']['total_count']}")
    if en['entities']['type_distribution']:
        print(f"    Types: {en['entities']['type_distribution']}")
    print(f"  Relationships: {en['relationships']['total_count']}")
    if en['relationships']['type_distribution']:
        print(f"    Types: {en['relationships']['type_distribution']}")

    # Arabic metrics
    print("\n--- ARABIC ---")
    ar = results["arabic"]
    print(f"  Entities: {ar['entities']['total_count']}")
    if ar['entities']['type_distribution']:
        print(f"    Types: {ar['entities']['type_distribution']}")
    print(f"  Relationships: {ar['relationships']['total_count']}")
    if ar['relationships']['type_distribution']:
        print(f"    Types: {ar['relationships']['type_distribution']}")

    # Combined
    print("\n--- COMBINED ---")
    comb = results["combined"]
    print(f"  Total Entities: {comb['entities']['total_count']}")
    print(f"  Total Relationships: {comb['relationships']['total_count']}")

    # Speed
    if results["speed"]:
        print("\n--- SPEED ---")
        if "english" in results["speed"]:
            sp = results["speed"]["english"]
            print(f"  English: {sp['elapsed_seconds']}s, {sp['tokens_per_second']} tok/s, {sp['pages_per_minute']} pages/min")
        if "arabic" in results["speed"]:
            sp = results["speed"]["arabic"]
            print(f"  Arabic: {sp['elapsed_seconds']}s, {sp['tokens_per_second']} tok/s, {sp['pages_per_minute']} pages/min")

    # Cross-language
    print("\n--- CROSS-LANGUAGE ---")
    cl = results["cross_language"]
    print(f"  English orgs: {cl['english_org_count']}")
    print(f"  Arabic orgs: {cl['arabic_org_count']}")
    print(f"  Potential links: {cl['potential_linking_candidates']}")

    print("\n" + "="*70)


def compare_models(extractions_dir: str) -> Dict[str, Any]:
    """Compare all models in the extractions directory."""
    extractions_dir = Path(extractions_dir)
    comparison = {}

    for model_dir in extractions_dir.iterdir():
        if model_dir.is_dir():
            try:
                results = analyze_model_results(str(model_dir))
                comparison[model_dir.name] = {
                    "total_entities": results["combined"]["entities"]["total_count"],
                    "total_relationships": results["combined"]["relationships"]["total_count"],
                    "english_entities": results["english"]["entities"]["total_count"],
                    "arabic_entities": results["arabic"]["entities"]["total_count"],
                    "speed": results.get("speed", {})
                }
            except Exception as e:
                comparison[model_dir.name] = {"error": str(e)}

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Calculate extraction metrics")
    parser.add_argument("--model-dir", help="Single model directory to analyze")
    parser.add_argument("--compare-all", help="Compare all models in extractions directory")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    if args.model_dir:
        results = analyze_model_results(args.model_dir)
        print_report(results)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {args.output}")

    elif args.compare_all:
        comparison = compare_models(args.compare_all)

        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(f"\n{'Model':<25} {'Entities':<12} {'Relations':<12} {'EN Ent':<10} {'AR Ent':<10}")
        print("-"*70)

        for model, stats in sorted(comparison.items()):
            if "error" in stats:
                print(f"{model:<25} ERROR: {stats['error'][:40]}")
            else:
                print(f"{model:<25} {stats['total_entities']:<12} {stats['total_relationships']:<12} {stats['english_entities']:<10} {stats['arabic_entities']:<10}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
