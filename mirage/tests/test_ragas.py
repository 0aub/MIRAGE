import pytest
import os
from src.core.retrieval.answer_synthesizer import get_answer_synthesizer, SynthesisStrategy
from src.core.retrieval.base_retriever import RetrievalResult

# Try to import ragas
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    from src.evaluation.rag_metrics import RAGEvaluator

@pytest.mark.skipif(not RAGAS_AVAILABLE, reason="RAGAS not installed")
def test_ragas_evaluation():
    """Run verification using RAGAS standard metrics."""
    
    # 1. Synthesize an answer
    synthesizer = get_answer_synthesizer()
    query = "What is the capital of France?"
    
    # Mock retrieval results
    results = [
        RetrievalResult(
            chunk_id="1", document_id="d1", score=0.9,
            text="Paris is the capital and most populous city of France."
        ),
        RetrievalResult(
            chunk_id="2", document_id="d2", score=0.8,
            text="France is a country in Western Europe."
        )
    ]
    
    synthesized = synthesizer.synthesize(query, results, strategy=SynthesisStrategy.EXTRACTIVE)
    
    # 2. Prepare data for RAGAS
    data = {
        'question': [query],
        'answer': [synthesized.answer],
        'contexts': [[r.text for r in results]],
        'ground_truth': ["Paris"] # Optional
    }
    dataset = Dataset.from_dict(data)
    
    # 3. Evaluate
    # Note: Requires OpenAI API key or configured LLM in RAGAS
    if os.environ.get("OPENAI_API_KEY"):
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy]
        )
        
        print(f"RAGAS Results: {results}")
        assert results['faithfulness'] > 0.7
        assert results['answer_relevancy'] > 0.7
    else:
        pytest.skip("No OpenAI API key for RAGAS")

def test_fallback_evaluation():
    """Run verification using internal custom metrics (SLM-optimized)."""
    if RAGAS_AVAILABLE:
        return # Skip if RAGAS is used
        
    evaluator = RAGEvaluator()
    
    query = "Who founded SpaceX?"
    answer = "SpaceX was founded by Elon Musk in 2002."
    contexts = ["Elon Musk founded Space Exploration Technologies Corp. (SpaceX) in 2002."]
    
    metrics = evaluator.evaluate_answer(query, answer, contexts)
    
    print(f"Custom Metrics: {metrics}")
    assert metrics.faithfulness > 0.8
    assert metrics.answer_relevancy > 0.8
