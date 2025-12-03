"""
MIRAGE V2 Experiment Runner
Runs and tracks RAG experiments with different configurations.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import time
from loguru import logger

from ..metrics.base_metrics import (
    MetricSuite,
    EvaluationSample,
    MetricResult,
    MetricType
)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    description: str = ""

    # Retrieval settings
    retrieval_mode: str = "hybrid"
    top_k: int = 10

    # Generation settings
    use_cot: bool = False
    max_tokens: int = 512
    temperature: float = 0.1

    # Chunking settings
    chunking_strategy: str = "semantic"

    # Other settings
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result from an experiment run"""
    config: ExperimentConfig
    metrics: Dict[str, Dict[str, float]]
    samples_evaluated: int
    total_time_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: List[str] = field(default_factory=list)
    sample_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "retrieval_mode": self.config.retrieval_mode,
                "top_k": self.config.top_k,
                "use_cot": self.config.use_cot,
                "chunking_strategy": self.config.chunking_strategy,
                "metadata": self.config.metadata
            },
            "metrics": self.metrics,
            "samples_evaluated": self.samples_evaluated,
            "total_time_seconds": self.total_time_seconds,
            "timestamp": self.timestamp,
            "errors": self.errors
        }

    def summary(self) -> str:
        """Get a summary string"""
        lines = [
            f"Experiment: {self.config.name}",
            f"Mode: {self.config.retrieval_mode}, Top-K: {self.config.top_k}",
            f"Samples: {self.samples_evaluated}, Time: {self.total_time_seconds:.1f}s",
            "",
            "Metrics:",
        ]

        for name, values in self.metrics.items():
            mean = values.get("mean", 0)
            std = values.get("std", 0)
            lines.append(f"  {name}: {mean:.4f} (+/- {std:.4f})")

        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")

        return "\n".join(lines)


class ExperimentRunner:
    """
    Runs RAG experiments with different configurations.

    Features:
    - Multiple experiment configurations
    - Automatic metric collection
    - Result persistence
    - Comparison across experiments
    """

    def __init__(
        self,
        metric_suite: Optional[MetricSuite] = None,
        results_dir: Optional[Path] = None
    ):
        """
        Initialize experiment runner.

        Args:
            metric_suite: Metrics to use for evaluation
            results_dir: Directory to save results
        """
        self.metric_suite = metric_suite or MetricSuite()
        self.results_dir = Path(results_dir) if results_dir else None

        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        self.experiments: List[ExperimentResult] = []

        logger.info("ExperimentRunner initialized")

    def run_experiment(
        self,
        config: ExperimentConfig,
        dataset: List[EvaluationSample],
        retrieval_fn: Optional[Callable] = None,
        generation_fn: Optional[Callable] = None,
        save_results: bool = True
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration
            dataset: Evaluation samples
            retrieval_fn: Function to perform retrieval (query -> chunks)
            generation_fn: Function to generate answer (query, chunks -> answer)
            save_results: Whether to save results to disk

        Returns:
            ExperimentResult with metrics
        """
        logger.info(f"Starting experiment: {config.name}")
        start_time = time.time()
        errors = []
        sample_results = []

        # Process each sample
        for i, sample in enumerate(dataset):
            try:
                # Run retrieval if function provided
                if retrieval_fn:
                    try:
                        sample.retrieved_chunks = retrieval_fn(
                            sample.query,
                            top_k=config.top_k,
                            mode=config.retrieval_mode
                        )
                    except Exception as e:
                        logger.warning(f"Retrieval failed for sample {i}: {e}")
                        errors.append(f"Retrieval error sample {i}: {str(e)}")

                # Run generation if function provided
                if generation_fn and sample.retrieved_chunks:
                    try:
                        sample.generated_answer = generation_fn(
                            sample.query,
                            sample.retrieved_chunks,
                            use_cot=config.use_cot
                        )
                    except Exception as e:
                        logger.warning(f"Generation failed for sample {i}: {e}")
                        errors.append(f"Generation error sample {i}: {str(e)}")

                # Evaluate sample
                metrics = self.metric_suite.evaluate(sample)
                sample_results.append({
                    "query": sample.query,
                    "metrics": {k: v.value for k, v in metrics.items()}
                })

            except Exception as e:
                logger.error(f"Sample {i} failed: {e}")
                errors.append(f"Sample {i} failed: {str(e)}")

        # Aggregate metrics
        metrics_summary = self.metric_suite.evaluate_batch(dataset)

        total_time = time.time() - start_time

        result = ExperimentResult(
            config=config,
            metrics=metrics_summary,
            samples_evaluated=len(dataset),
            total_time_seconds=total_time,
            errors=errors,
            sample_results=sample_results
        )

        self.experiments.append(result)

        # Save if requested
        if save_results and self.results_dir:
            self._save_result(result)

        logger.info(f"Experiment complete: {config.name} ({total_time:.1f}s)")
        return result

    def run_comparison(
        self,
        configs: List[ExperimentConfig],
        dataset: List[EvaluationSample],
        retrieval_fn: Optional[Callable] = None,
        generation_fn: Optional[Callable] = None
    ) -> Dict[str, ExperimentResult]:
        """
        Run multiple experiments for comparison.

        Args:
            configs: List of experiment configurations
            dataset: Evaluation samples
            retrieval_fn: Retrieval function
            generation_fn: Generation function

        Returns:
            Dict of experiment_name -> ExperimentResult
        """
        results = {}

        for config in configs:
            # Create fresh copy of dataset for each experiment
            dataset_copy = [
                EvaluationSample(
                    query=s.query,
                    ground_truth_answer=s.ground_truth_answer,
                    ground_truth_chunks=s.ground_truth_chunks.copy(),
                    metadata=s.metadata.copy()
                )
                for s in dataset
            ]

            result = self.run_experiment(
                config=config,
                dataset=dataset_copy,
                retrieval_fn=retrieval_fn,
                generation_fn=generation_fn
            )
            results[config.name] = result

        return results

    def compare_results(
        self,
        results: Dict[str, ExperimentResult],
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare results across experiments.

        Args:
            results: Dict of experiment results
            metric_names: Metrics to compare (default: all)

        Returns:
            Comparison table: metric -> {experiment: value}
        """
        comparison = {}

        # Get all metric names
        if metric_names is None:
            metric_names = set()
            for result in results.values():
                metric_names.update(result.metrics.keys())
            metric_names = sorted(metric_names)

        for metric_name in metric_names:
            comparison[metric_name] = {}
            for exp_name, result in results.items():
                if metric_name in result.metrics:
                    comparison[metric_name][exp_name] = result.metrics[metric_name].get("mean", 0)

        return comparison

    def print_comparison(self, comparison: Dict[str, Dict[str, float]]):
        """Print comparison table"""
        if not comparison:
            print("No comparison data")
            return

        # Get experiment names
        exp_names = set()
        for values in comparison.values():
            exp_names.update(values.keys())
        exp_names = sorted(exp_names)

        # Header
        header = ["Metric"] + list(exp_names)
        print(" | ".join(f"{h:20}" for h in header))
        print("-" * (22 * len(header)))

        # Rows
        for metric_name, values in comparison.items():
            row = [metric_name[:20]]
            for exp_name in exp_names:
                val = values.get(exp_name, 0)
                row.append(f"{val:.4f}")
            print(" | ".join(f"{r:20}" for r in row))

    def _save_result(self, result: ExperimentResult):
        """Save result to disk"""
        filename = f"{result.config.name}_{result.timestamp.replace(':', '-')}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved result to {filepath}")

    def load_results(self, pattern: str = "*.json") -> List[ExperimentResult]:
        """Load results from disk"""
        if not self.results_dir:
            return []

        results = []
        for filepath in self.results_dir.glob(pattern):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                config = ExperimentConfig(
                    name=data["config"]["name"],
                    description=data["config"].get("description", ""),
                    retrieval_mode=data["config"].get("retrieval_mode", "hybrid"),
                    top_k=data["config"].get("top_k", 10),
                    use_cot=data["config"].get("use_cot", False),
                    chunking_strategy=data["config"].get("chunking_strategy", "semantic"),
                    metadata=data["config"].get("metadata", {})
                )

                result = ExperimentResult(
                    config=config,
                    metrics=data["metrics"],
                    samples_evaluated=data["samples_evaluated"],
                    total_time_seconds=data["total_time_seconds"],
                    timestamp=data["timestamp"],
                    errors=data.get("errors", [])
                )
                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        return results

    def get_best_config(
        self,
        metric_name: str = "NDCG@10"
    ) -> Optional[ExperimentConfig]:
        """Get the config with best performance on a metric"""
        if not self.experiments:
            return None

        best_result = max(
            self.experiments,
            key=lambda r: r.metrics.get(metric_name, {}).get("mean", 0)
        )
        return best_result.config


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_test_dataset(n_samples: int = 10) -> List[EvaluationSample]:
    """Create a simple test dataset for development"""
    samples = []

    test_queries = [
        ("What is MIRAGE?", "MIRAGE is a RAG system for Arabic."),
        ("Who developed MIRAGE?", "MIRAGE was developed by the research team."),
        ("ما هو نظام MIRAGE؟", "MIRAGE هو نظام استرجاع معلومات باللغة العربية."),
        ("What databases does MIRAGE use?", "MIRAGE uses Neo4j and Qdrant."),
        ("How does retrieval work?", "Retrieval uses both vector and graph search."),
    ]

    for i in range(n_samples):
        query, answer = test_queries[i % len(test_queries)]
        samples.append(EvaluationSample(
            query=query,
            ground_truth_answer=answer,
            ground_truth_chunks=[f"chunk_{i}_1", f"chunk_{i}_2"]
        ))

    return samples
