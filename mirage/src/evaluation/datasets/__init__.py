"""
Evaluation Datasets Module
Provides test datasets for RAG evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..metrics.base_metrics import EvaluationSample


@dataclass
class DatasetInfo:
    """Dataset metadata"""
    name: str
    version: str
    description: str
    total_samples: int
    languages: Dict[str, int]
    query_types: Dict[str, int]


def load_dataset(name: str = "arabic_rag_test") -> List[EvaluationSample]:
    """
    Load evaluation dataset by name.

    Args:
        name: Dataset name (without .json extension)

    Returns:
        List of EvaluationSample objects
    """
    dataset_dir = Path(__file__).parent
    dataset_path = dataset_dir / f"{name}.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data.get("samples", []):
        sample = EvaluationSample(
            query=item["query"],
            ground_truth_answer=item["ground_truth_answer"],
            ground_truth_chunks=item.get("ground_truth_chunks", []),
            retrieved_chunks=[],  # To be filled during evaluation
            generated_answer="",  # To be filled during evaluation
            metadata={
                "id": item.get("id"),
                "query_type": item.get("query_type"),
                "language": item.get("language"),
                "difficulty": item.get("difficulty"),
                "expected_entities": item.get("expected_entities", [])
            }
        )
        samples.append(sample)

    return samples


def get_dataset_info(name: str = "arabic_rag_test") -> DatasetInfo:
    """Get dataset metadata without loading samples"""
    dataset_dir = Path(__file__).parent
    dataset_path = dataset_dir / f"{name}.json"

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    return DatasetInfo(
        name=data.get("name", name),
        version=data.get("version", "unknown"),
        description=data.get("description", ""),
        total_samples=meta.get("total_samples", 0),
        languages=meta.get("languages", {}),
        query_types=meta.get("query_types", {})
    )


def list_datasets() -> List[str]:
    """List available datasets"""
    dataset_dir = Path(__file__).parent
    return [p.stem for p in dataset_dir.glob("*.json")]


__all__ = [
    "load_dataset",
    "get_dataset_info",
    "list_datasets",
    "DatasetInfo"
]
