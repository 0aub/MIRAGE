#!/usr/bin/env python3
"""
Comprehensive test suite for ALLaM-7B quality assessment.
Tests the 3 critical tasks for GraphRAG implementation.
"""

import sys
import json
import requests
from typing import Dict, List
from datetime import datetime

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class AllamTester:
    """Test ALLaM-7B for GraphRAG tasks."""

    def __init__(self, endpoint: str = "http://localhost:8765"):
        self.endpoint = endpoint
        self.results = []

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """Call Allam via TGI endpoint."""
        try:
            response = requests.post(
                f"{self.endpoint}/generate",
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.95,
                        "do_sample": True,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('generated_text', '').strip()
            else:
                return f"ERROR: {response.status_code}"

        except Exception as e:
            return f"ERROR: {str(e)}"

    def print_section(self, title: str):
        """Print section header."""
        print(f"\n{'='*80}")
        print(f"{BLUE}{title}{RESET}")
        print(f"{'='*80}\n")

    def print_test(self, test_name: str, score: int, details: str = ""):
        """Print test result."""
        color = GREEN if score >= 75 else YELLOW if score >= 60 else RED
        status = "✓ PASS" if score >= 75 else "⚠ MARGINAL" if score >= 60 else "✗ FAIL"

        print(f"{color}{status}{RESET} {test_name}: {score}/100")
        if details:
            print(f"  {details}\n")

    # ========================================================================
    # TEST 1: Entity Extraction (Critical for GraphRAG)
    # ========================================================================

    def test_entity_extraction(self):
        """Test entity extraction quality."""
        self.print_section("TEST 1: Entity Extraction (Arabic)")

        test_cases = [
            {
                "text": "أعلن الرئيس محمد في القاهرة عن مبادرة جديدة للتعامل مع تغير المناخ في مصر والشرق الأوسط.",
                "expected_entities": ["محمد", "القاهرة", "مصر", "الشرق الأوسط", "تغير المناخ"],
                "expected_types": ["Person", "Location", "Location", "Location", "Concept"]
            },
            {
                "text": "وقعت شركة أرامكو السعودية اتفاقية مع جامعة الملك عبد الله للعلوم والتقنية في جدة.",
                "expected_entities": ["أرامكو السعودية", "جامعة الملك عبد الله", "جدة"],
                "expected_types": ["Organization", "Organization", "Location"]
            },
            {
                "text": "Dr. Ahmed Hassan works at IBM in Riyadh developing AI solutions.",
                "expected_entities": ["Ahmed Hassan", "IBM", "Riyadh", "AI"],
                "expected_types": ["Person", "Organization", "Location", "Concept"]
            }
        ]

        total_score = 0

        for i, test_case in enumerate(test_cases, 1):
            prompt = f"""استخرج جميع الكيانات من النص التالي.

النص: {test_case['text']}

لكل كيان، حدد:
- الاسم
- النوع (Person, Organization, Location, Concept, Date, etc.)

أعد القائمة بصيغة JSON:
{{"entities": [{{"name": "اسم الكيان", "type": "النوع"}}]}}
"""

            print(f"Test {i}: {test_case['text'][:60]}...")
            response = self.generate(prompt, max_tokens=300)
            print(f"Response: {response}\n")

            # Score the response
            score = self._score_entity_extraction(response, test_case)
            total_score += score

            self.print_test(f"Entity Extraction Test {i}", score)

        avg_score = total_score / len(test_cases)
        self.results.append({
            "test": "Entity Extraction",
            "score": avg_score,
            "threshold": 75
        })

        return avg_score

    def _score_entity_extraction(self, response: str, test_case: Dict) -> int:
        """Score entity extraction response."""
        score = 0

        # Check if we got valid JSON
        try:
            # Try to extract JSON from response
            if '{' in response:
                json_start = response.index('{')
                json_end = response.rindex('}') + 1
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                score += 20  # Valid JSON format
            else:
                return 10  # No JSON found
        except:
            return 10  # Invalid JSON

        # Check if entities were extracted
        if 'entities' in data and isinstance(data['entities'], list):
            score += 20
            extracted_names = [e.get('name', '') for e in data['entities']]

            # Check how many expected entities were found
            found_count = 0
            for expected in test_case['expected_entities']:
                # Fuzzy matching
                if any(expected.lower() in name.lower() or name.lower() in expected.lower()
                       for name in extracted_names):
                    found_count += 1

            coverage = (found_count / len(test_case['expected_entities'])) * 60
            score += int(coverage)

        return min(100, score)

    # ========================================================================
    # TEST 2: Summarization (Critical for Community Summaries)
    # ========================================================================

    def test_summarization(self):
        """Test summarization quality."""
        self.print_section("TEST 2: Community Summarization")

        test_cases = [
            {
                "entities": ["التغير المناخي", "مصر", "الطاقة المتجددة", "السد العالي", "نهر النيل"],
                "relationships": [
                    "التغير المناخي → يؤثر على → مصر",
                    "مصر → تستثمر في → الطاقة المتجددة",
                    "السد العالي → يقع على → نهر النيل",
                    "التغير المناخي → يهدد → نهر النيل"
                ],
                "expected_theme": "environmental issues in Egypt"
            },
            {
                "entities": ["IBM", "Microsoft", "Google", "AI", "Cloud Computing"],
                "relationships": [
                    "IBM → competes with → Microsoft",
                    "Microsoft → develops → AI",
                    "Google → leads in → Cloud Computing",
                    "AI → powers → Cloud Computing"
                ],
                "expected_theme": "tech companies and AI"
            }
        ]

        total_score = 0

        for i, test_case in enumerate(test_cases, 1):
            entity_list = ", ".join(test_case['entities'])
            rel_list = "\n".join(test_case['relationships'])

            prompt = f"""لخّص هذه المجموعة من الكيانات المترابطة.

الكيانات: {entity_list}

العلاقات الرئيسية:
{rel_list}

التعليمات:
1. حدد الموضوع الرئيسي
2. اذكر الكيانات الأساسية
3. اشرح العلاقات المهمة
4. اكتب 2-3 فقرات

الملخص:
"""

            print(f"Test {i}: Summarizing {len(test_case['entities'])} entities...")
            response = self.generate(prompt, max_tokens=500)
            print(f"Summary: {response[:200]}...\n")

            # Score the response
            score = self._score_summarization(response, test_case)
            total_score += score

            self.print_test(f"Summarization Test {i}", score)

        avg_score = total_score / len(test_cases)
        self.results.append({
            "test": "Summarization",
            "score": avg_score,
            "threshold": 70
        })

        return avg_score

    def _score_summarization(self, response: str, test_case: Dict) -> int:
        """Score summarization response."""
        score = 0

        # Check length (should be reasonable)
        if 100 < len(response) < 1000:
            score += 30
        elif 50 < len(response) < 100:
            score += 15

        # Check if entities are mentioned
        mentioned_count = 0
        for entity in test_case['entities']:
            if entity.lower() in response.lower():
                mentioned_count += 1

        entity_coverage = (mentioned_count / len(test_case['entities'])) * 40
        score += int(entity_coverage)

        # Check if it's coherent (has Arabic or English sentences)
        has_structure = (
            ('.' in response or '،' in response or '؛' in response) and
            len(response.split()) > 20
        )
        if has_structure:
            score += 30

        return min(100, score)

    # ========================================================================
    # TEST 3: Question Answering (Critical for Query Processing)
    # ========================================================================

    def test_question_answering(self):
        """Test question answering quality."""
        self.print_section("TEST 3: Question Answering with Context")

        test_cases = [
            {
                "question": "ما هو تأثير التغير المناخي على مصر؟",
                "context": [
                    "يؤثر التغير المناخي بشكل كبير على مصر من خلال ارتفاع مستوى سطح البحر الذي يهدد دلتا النيل.",
                    "تستثمر مصر في الطاقة المتجددة لمواجهة تحديات التغير المناخي.",
                    "السد العالي يلعب دوراً مهماً في توفير الكهرباء والحماية من الفيضانات."
                ],
                "expected_keywords": ["دلتا النيل", "البحر", "الطاقة المتجددة", "تهديد"]
            },
            {
                "question": "What are the main AI technologies developed by tech companies?",
                "context": [
                    "Microsoft has developed GPT-4 and Azure AI services for enterprise customers.",
                    "Google leads in cloud computing and has created TensorFlow for machine learning.",
                    "IBM focuses on Watson AI for business intelligence and data analysis."
                ],
                "expected_keywords": ["GPT-4", "TensorFlow", "Watson", "machine learning"]
            }
        ]

        total_score = 0

        for i, test_case in enumerate(test_cases, 1):
            context_text = "\n\n".join(test_case['context'])

            prompt = f"""السؤال: {test_case['question']}

السياق:
{context_text}

أجب على السؤال بناءً على السياق المقدم.
كن دقيقاً وموجزاً (2-3 فقرات).

الإجابة:
"""

            print(f"Test {i}: {test_case['question']}")
            response = self.generate(prompt, max_tokens=400)
            print(f"Answer: {response[:200]}...\n")

            # Score the response
            score = self._score_qa(response, test_case)
            total_score += score

            self.print_test(f"QA Test {i}", score)

        avg_score = total_score / len(test_cases)
        self.results.append({
            "test": "Question Answering",
            "score": avg_score,
            "threshold": 75
        })

        return avg_score

    def _score_qa(self, response: str, test_case: Dict) -> int:
        """Score QA response."""
        score = 0

        # Check if response is not empty and reasonable length
        if 50 < len(response) < 800:
            score += 30
        elif 20 < len(response) < 50:
            score += 15

        # Check if response mentions expected keywords
        keyword_count = 0
        for keyword in test_case['expected_keywords']:
            if keyword.lower() in response.lower():
                keyword_count += 1

        keyword_coverage = (keyword_count / len(test_case['expected_keywords'])) * 50
        score += int(keyword_coverage)

        # Check if response is relevant (mentions terms from question)
        question_terms = set(test_case['question'].lower().split())
        response_terms = set(response.lower().split())
        overlap = len(question_terms & response_terms)

        if overlap > 2:
            score += 20

        return min(100, score)

    # ========================================================================
    # Final Report
    # ========================================================================

    def generate_report(self):
        """Generate final test report."""
        self.print_section("FINAL TEST REPORT")

        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Model: ALLaM-7B-Instruct-preview")
        print(f"Endpoint: {self.endpoint}\n")

        print("Results:")
        print("-" * 80)

        overall_pass = True
        total_score = 0

        for result in self.results:
            test_name = result['test']
            score = result['score']
            threshold = result['threshold']
            passed = score >= threshold

            color = GREEN if passed else RED
            status = "PASS ✓" if passed else "FAIL ✗"

            print(f"{color}{status:10}{RESET} {test_name:30} {score:5.1f}/100 (threshold: {threshold})")

            total_score += score
            if not passed:
                overall_pass = False

        avg_score = total_score / len(self.results)

        print("-" * 80)
        print(f"\nOverall Average Score: {avg_score:.1f}/100")

        print("\n" + "="*80)
        if overall_pass and avg_score >= 75:
            print(f"{GREEN}✓ RECOMMENDATION: Allam is GOOD ENOUGH - Proceed with implementation{RESET}")
            return "PROCEED"
        elif avg_score >= 65:
            print(f"{YELLOW}⚠ RECOMMENDATION: Allam is MARGINAL - Consider hybrid approach{RESET}")
            print(f"{YELLOW}  Use Allam for simple tasks, fallback to GPT-4 for complex ones{RESET}")
            return "HYBRID"
        else:
            print(f"{RED}✗ RECOMMENDATION: Allam is INSUFFICIENT - Test alternative models{RESET}")
            return "TEST_OTHERS"
        print("="*80 + "\n")

def main():
    """Run all tests."""
    print(f"\n{BLUE}{'='*80}")
    print("ALLaM-7B Quality Assessment for GraphRAG Implementation")
    print(f"{'='*80}{RESET}\n")

    tester = AllamTester()

    # Run all tests
    entity_score = tester.test_entity_extraction()
    summary_score = tester.test_summarization()
    qa_score = tester.test_question_answering()

    # Generate report
    recommendation = tester.generate_report()

    return recommendation

if __name__ == "__main__":
    recommendation = main()

    # Exit code based on recommendation
    if recommendation == "PROCEED":
        sys.exit(0)  # Success
    elif recommendation == "HYBRID":
        sys.exit(1)  # Marginal
    else:
        sys.exit(2)  # Test others
