"""
MIRAGE GraphRAG Evaluation Framework

Runs benchmark questions against the RAG system and scores responses.
Supports multiple retrieval modes and LLM-based answer evaluation.
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from loguru import logger

# API Configuration
API_BASE_URL = "http://localhost:8000"


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question_id: str
    question_type: str
    question_text: str
    retrieval_mode: str
    answer: str
    response_time_ms: float
    chunks_retrieved: int
    entities_found: List[str]

    # Scores (0-1)
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0

    # Metadata
    expected_mode: str = ""
    difficulty: str = ""
    error: Optional[str] = None

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        return (
            self.relevance_score * 0.4 +
            self.completeness_score * 0.3 +
            self.accuracy_score * 0.3
        )


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    total_questions: int
    total_time_seconds: float
    retrieval_mode: str

    # Aggregate scores
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_accuracy: float = 0.0
    avg_overall: float = 0.0
    avg_response_time_ms: float = 0.0

    # By type breakdown
    scores_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scores_by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Individual results
    results: List[Dict[str, Any]] = field(default_factory=list)

    # Errors
    error_count: int = 0
    errors: List[str] = field(default_factory=list)


class MIRAGEEvaluator:
    """Evaluator for MIRAGE RAG system."""

    def __init__(
        self,
        api_base_url: str = API_BASE_URL,
        benchmark_file: str = None
    ):
        self.api_base_url = api_base_url
        self.benchmark_file = benchmark_file or str(
            Path(__file__).parent / "benchmark_questions.json"
        )
        self.questions = self._load_questions()

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load benchmark questions from JSON file."""
        with open(self.benchmark_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("questions", [])

    def ask_question(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Send a question to the MIRAGE API."""
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/ask",
                json={
                    "message": question,
                    "retrieval_mode": retrieval_mode,
                    "top_k": top_k
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API error: {e}")
            return {"error": str(e)}

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        expected_entities: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate answer quality using heuristics.

        Returns scores for:
        - relevance: Does the answer address the question?
        - completeness: Does it cover expected entities/aspects?
        - accuracy: Is the information correct (based on entity presence)?
        """
        scores = {
            "relevance": 0.0,
            "completeness": 0.0,
            "accuracy": 0.0
        }

        if not answer or len(answer.strip()) < 10:
            return scores

        answer_lower = answer.lower()
        question_lower = question.lower()

        # Relevance: Check if answer is on-topic
        # Simple heuristic: keyword overlap
        question_words = set(question_lower.split())
        answer_words = set(answer_lower.split())

        # Remove common stopwords
        stopwords = {"ما", "هي", "هو", "من", "في", "على", "إلى", "عن", "مع", "أو", "و",
                    "what", "is", "the", "are", "how", "does", "do", "a", "an", "of", "to", "in"}
        question_words -= stopwords
        answer_words -= stopwords

        if question_words:
            keyword_overlap = len(question_words & answer_words) / len(question_words)
            scores["relevance"] = min(keyword_overlap * 1.5, 1.0)  # Boost and cap at 1.0

        # Answer length bonus (longer answers tend to be more complete)
        length_score = min(len(answer) / 500, 1.0)  # Max bonus at 500 chars
        scores["relevance"] = (scores["relevance"] + length_score) / 2

        # Completeness: Check expected entities
        if expected_entities:
            found_count = 0
            for entity in expected_entities:
                if entity.lower() in answer_lower:
                    found_count += 1
            scores["completeness"] = found_count / len(expected_entities)
        else:
            # No expected entities - use length as proxy
            scores["completeness"] = length_score

        # Accuracy: Higher if mentions specific entities
        # This is a proxy since we can't verify factual accuracy without ground truth
        specific_indicators = [
            "2030", "2024", "2025",  # Years
            "%", "٪",  # Percentages
            "مليون", "مليار", "million", "billion",  # Numbers
        ]
        specificity = sum(1 for ind in specific_indicators if ind in answer) / len(specific_indicators)
        scores["accuracy"] = max(scores["completeness"], specificity)

        return scores

    def run_single_evaluation(
        self,
        question_data: Dict[str, Any],
        retrieval_mode: str = "hybrid",
        use_arabic: bool = True
    ) -> EvaluationResult:
        """Evaluate a single question."""
        question_id = question_data["id"]
        question_type = question_data["type"]
        question = question_data.get("question_ar" if use_arabic else "question_en", "")
        expected_entities = question_data.get("expected_entities", [])
        expected_mode = question_data.get("expected_mode", "")
        difficulty = question_data.get("difficulty", "")

        logger.info(f"Evaluating {question_id}: {question[:50]}...")

        # Time the API call
        start_time = time.time()
        response = self.ask_question(question, retrieval_mode)
        response_time_ms = (time.time() - start_time) * 1000

        # Handle errors
        if "error" in response:
            return EvaluationResult(
                question_id=question_id,
                question_type=question_type,
                question_text=question,
                retrieval_mode=retrieval_mode,
                answer="",
                response_time_ms=response_time_ms,
                chunks_retrieved=0,
                entities_found=[],
                expected_mode=expected_mode,
                difficulty=difficulty,
                error=response["error"]
            )

        # Extract answer and metadata
        answer = response.get("answer", response.get("response", ""))
        chunks = response.get("chunks", response.get("sources", []))
        entities_found = response.get("entities_found", [])

        # Evaluate answer quality
        scores = self.evaluate_answer(question, answer, expected_entities)

        return EvaluationResult(
            question_id=question_id,
            question_type=question_type,
            question_text=question,
            retrieval_mode=retrieval_mode,
            answer=answer,
            response_time_ms=response_time_ms,
            chunks_retrieved=len(chunks) if chunks else 0,
            entities_found=entities_found if isinstance(entities_found, list) else [],
            relevance_score=scores["relevance"],
            completeness_score=scores["completeness"],
            accuracy_score=scores["accuracy"],
            expected_mode=expected_mode,
            difficulty=difficulty
        )

    def run_benchmark(
        self,
        retrieval_mode: str = "hybrid",
        question_types: Optional[List[str]] = None,
        max_questions: Optional[int] = None,
        use_arabic: bool = True
    ) -> BenchmarkReport:
        """
        Run full benchmark evaluation.

        Args:
            retrieval_mode: RAG retrieval mode to test
            question_types: Filter to specific types (factual, analytical, etc.)
            max_questions: Limit number of questions
            use_arabic: Use Arabic or English questions

        Returns:
            BenchmarkReport with all results and scores
        """
        # Filter questions
        questions = self.questions
        if question_types:
            questions = [q for q in questions if q["type"] in question_types]
        if max_questions:
            questions = questions[:max_questions]

        logger.info(f"Running benchmark: {len(questions)} questions, mode={retrieval_mode}")

        start_time = time.time()
        results: List[EvaluationResult] = []

        for q in questions:
            result = self.run_single_evaluation(q, retrieval_mode, use_arabic)
            results.append(result)

            # Brief delay to avoid overwhelming API
            time.sleep(0.5)

        total_time = time.time() - start_time

        # Calculate aggregate scores
        valid_results = [r for r in results if r.error is None]

        if valid_results:
            avg_relevance = sum(r.relevance_score for r in valid_results) / len(valid_results)
            avg_completeness = sum(r.completeness_score for r in valid_results) / len(valid_results)
            avg_accuracy = sum(r.accuracy_score for r in valid_results) / len(valid_results)
            avg_overall = sum(r.overall_score for r in valid_results) / len(valid_results)
            avg_response_time = sum(r.response_time_ms for r in valid_results) / len(valid_results)
        else:
            avg_relevance = avg_completeness = avg_accuracy = avg_overall = avg_response_time = 0.0

        # Group by type
        scores_by_type = {}
        for q_type in ["factual", "analytical", "comparative", "global", "relationship"]:
            type_results = [r for r in valid_results if r.question_type == q_type]
            if type_results:
                scores_by_type[q_type] = {
                    "count": len(type_results),
                    "avg_relevance": sum(r.relevance_score for r in type_results) / len(type_results),
                    "avg_completeness": sum(r.completeness_score for r in type_results) / len(type_results),
                    "avg_accuracy": sum(r.accuracy_score for r in type_results) / len(type_results),
                    "avg_overall": sum(r.overall_score for r in type_results) / len(type_results),
                }

        # Group by difficulty
        scores_by_difficulty = {}
        for difficulty in ["easy", "medium", "hard"]:
            diff_results = [r for r in valid_results if r.difficulty == difficulty]
            if diff_results:
                scores_by_difficulty[difficulty] = {
                    "count": len(diff_results),
                    "avg_overall": sum(r.overall_score for r in diff_results) / len(diff_results),
                }

        # Collect errors
        errors = [f"{r.question_id}: {r.error}" for r in results if r.error]

        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_questions=len(questions),
            total_time_seconds=total_time,
            retrieval_mode=retrieval_mode,
            avg_relevance=avg_relevance,
            avg_completeness=avg_completeness,
            avg_accuracy=avg_accuracy,
            avg_overall=avg_overall,
            avg_response_time_ms=avg_response_time,
            scores_by_type=scores_by_type,
            scores_by_difficulty=scores_by_difficulty,
            results=[asdict(r) for r in results],
            error_count=len(errors),
            errors=errors
        )

        return report

    def compare_modes(
        self,
        modes: List[str] = None,
        question_types: Optional[List[str]] = None,
        max_questions: int = 10
    ) -> Dict[str, BenchmarkReport]:
        """
        Compare multiple retrieval modes.

        Args:
            modes: List of modes to compare
            question_types: Filter questions
            max_questions: Questions per mode

        Returns:
            Dict mapping mode to BenchmarkReport
        """
        if modes is None:
            modes = ["naive", "local", "global", "hybrid"]

        results = {}
        for mode in modes:
            logger.info(f"\n{'='*50}\nTesting mode: {mode}\n{'='*50}")
            report = self.run_benchmark(
                retrieval_mode=mode,
                question_types=question_types,
                max_questions=max_questions
            )
            results[mode] = report

            # Print summary
            print(f"\n{mode.upper()} Mode Results:")
            print(f"  Overall Score: {report.avg_overall:.2%}")
            print(f"  Relevance: {report.avg_relevance:.2%}")
            print(f"  Completeness: {report.avg_completeness:.2%}")
            print(f"  Avg Response Time: {report.avg_response_time_ms:.0f}ms")
            print(f"  Errors: {report.error_count}/{report.total_questions}")

        return results

    def save_report(self, report: BenchmarkReport, filename: str = None):
        """Save benchmark report to JSON file."""
        if filename is None:
            filename = f"benchmark_report_{report.retrieval_mode}_{report.timestamp[:10]}.json"

        output_dir = Path(__file__).parent / "reports"
        output_dir.mkdir(exist_ok=True)

        output_path = output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

        logger.info(f"Report saved to: {output_path}")
        return output_path


def print_comparison_table(results: Dict[str, BenchmarkReport]):
    """Print a comparison table of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON TABLE")
    print("="*80)

    # Header
    header = f"{'Mode':<12} {'Overall':>10} {'Relevance':>10} {'Complete':>10} {'Accuracy':>10} {'Time(ms)':>10} {'Errors':>8}"
    print(header)
    print("-"*80)

    # Rows
    for mode, report in sorted(results.items()):
        row = f"{mode:<12} {report.avg_overall:>9.1%} {report.avg_relevance:>9.1%} {report.avg_completeness:>9.1%} {report.avg_accuracy:>9.1%} {report.avg_response_time_ms:>9.0f} {report.error_count:>7}"
        print(row)

    print("="*80)

    # By type breakdown
    print("\nSCORES BY QUESTION TYPE:")
    print("-"*80)

    types = ["factual", "analytical", "comparative", "global", "relationship"]

    for q_type in types:
        print(f"\n{q_type.upper()}:")
        for mode, report in sorted(results.items()):
            if q_type in report.scores_by_type:
                type_score = report.scores_by_type[q_type]
                print(f"  {mode:<10}: {type_score['avg_overall']:.1%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIRAGE RAG Evaluation")
    parser.add_argument("--mode", default="hybrid", help="Retrieval mode to test")
    parser.add_argument("--compare", action="store_true", help="Compare all modes")
    parser.add_argument("--types", nargs="+", help="Question types to test")
    parser.add_argument("--max", type=int, default=50, help="Max questions")
    parser.add_argument("--api", default=API_BASE_URL, help="API base URL")

    args = parser.parse_args()

    evaluator = MIRAGEEvaluator(api_base_url=args.api)

    if args.compare:
        results = evaluator.compare_modes(
            question_types=args.types,
            max_questions=args.max
        )
        print_comparison_table(results)

        # Save all reports
        for mode, report in results.items():
            evaluator.save_report(report)
    else:
        report = evaluator.run_benchmark(
            retrieval_mode=args.mode,
            question_types=args.types,
            max_questions=args.max
        )

        print(f"\nBenchmark Results ({args.mode} mode):")
        print(f"  Overall Score: {report.avg_overall:.2%}")
        print(f"  Relevance: {report.avg_relevance:.2%}")
        print(f"  Completeness: {report.avg_completeness:.2%}")
        print(f"  Accuracy: {report.avg_accuracy:.2%}")
        print(f"  Avg Response Time: {report.avg_response_time_ms:.0f}ms")
        print(f"  Errors: {report.error_count}/{report.total_questions}")

        evaluator.save_report(report)
