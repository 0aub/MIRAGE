"""
GraphRAG Query Decomposition (Phase 7)
Breaks complex multi-hop queries into simpler sub-queries for better retrieval.

Microsoft GraphRAG uses query decomposition for:
1. Multi-hop questions: "What is X and how does it relate to Y?"
2. Comparison queries: "Compare A with B"
3. Aggregation queries: "What are all the programs launched by X?"
4. Conditional queries: "If X happens, what does Y do?"

This improves retrieval by:
- Converting complex queries into atomic sub-queries
- Retrieving context for each sub-query independently
- Combining results for comprehensive answers
"""

import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ...config import settings


class QueryType(Enum):
    """Types of complex queries that need decomposition."""
    SIMPLE = "simple"               # No decomposition needed
    MULTI_HOP = "multi_hop"         # Requires following relationships
    COMPARISON = "comparison"        # Compare two or more entities
    AGGREGATION = "aggregation"     # Collect information across entities
    CONDITIONAL = "conditional"     # If-then type queries
    TEMPORAL = "temporal"           # Time-based queries
    CAUSAL = "causal"               # Why/cause-effect queries


@dataclass
class SubQuery:
    """A decomposed sub-query."""
    query: str                      # The sub-query text
    query_type: str                 # Type of sub-query
    depends_on: List[int]           # Indices of sub-queries this depends on
    target_entities: List[str]      # Entities to focus on
    reasoning: str                  # Why this sub-query was created


@dataclass
class DecompositionResult:
    """Result of query decomposition."""
    original_query: str
    query_type: QueryType
    needs_decomposition: bool
    sub_queries: List[SubQuery]
    execution_order: List[int]      # Order to execute sub-queries
    combination_strategy: str       # How to combine results
    reasoning: str                  # Explanation of decomposition


class QueryDecomposer:
    """
    LLM-powered query decomposition for GraphRAG.

    Analyzes complex queries and breaks them into simpler sub-queries
    that can be answered independently, then combines the results.

    Usage:
        decomposer = QueryDecomposer()
        result = decomposer.decompose("ما علاقة هيئة الحكومة الرقمية بمنتدى 2025؟")

        if result.needs_decomposition:
            for sq in result.sub_queries:
                # Execute each sub-query
                context = retriever.retrieve(sq.query)
    """

    # Patterns that indicate complex queries (Arabic)
    COMPLEXITY_PATTERNS = {
        QueryType.MULTI_HOP: [
            r'علاقة.*ب',              # relationship with
            r'كيف\s+(?:يرتبط|ترتبط)',  # how is related
            r'من خلال',              # through
            r'عبر',                  # via
        ],
        QueryType.COMPARISON: [
            r'مقارنة\s+بين',          # compare between
            r'الفرق\s+بين',           # difference between
            r'أفضل',                  # better
            r'مقابل',                 # versus
        ],
        QueryType.AGGREGATION: [
            r'جميع',                  # all
            r'كل',                    # every
            r'كم\s+عدد',              # how many
            r'اذكر',                  # list/mention
            r'ما هي',                 # what are (plural)
        ],
        QueryType.CONDITIONAL: [
            r'إذا',                   # if
            r'في\s+حالة',             # in case
            r'عندما',                 # when
            r'لو',                    # if (conditional)
        ],
        QueryType.TEMPORAL: [
            r'متى',                   # when
            r'قبل',                   # before
            r'بعد',                   # after
            r'خلال',                  # during
            r'منذ',                   # since
        ],
        QueryType.CAUSAL: [
            r'لماذا',                 # why
            r'بسبب',                  # because of
            r'نتيجة',                 # result
            r'سبب',                   # cause
        ],
    }

    def __init__(
        self,
        llm_endpoint: str = None,
        use_llm: bool = True,
        min_query_length: int = 10
    ):
        """
        Initialize query decomposer.

        Args:
            llm_endpoint: TGI endpoint for LLM decomposition
            use_llm: Whether to use LLM (vs rule-based only)
            min_query_length: Minimum query length to consider for decomposition
        """
        self.llm_endpoint = llm_endpoint or settings.tgi_endpoint or "http://tgi:80"
        self.use_llm = use_llm
        self.min_query_length = min_query_length
        logger.info(f"QueryDecomposer initialized: llm={self.use_llm}")

    def decompose(self, query: str) -> DecompositionResult:
        """
        Decompose a query into sub-queries if needed.

        Args:
            query: User query to decompose

        Returns:
            DecompositionResult with sub-queries and execution plan
        """
        # Detect query type
        query_type = self._detect_query_type(query)

        # Simple queries don't need decomposition
        if query_type == QueryType.SIMPLE or len(query) < self.min_query_length:
            return DecompositionResult(
                original_query=query,
                query_type=QueryType.SIMPLE,
                needs_decomposition=False,
                sub_queries=[],
                execution_order=[],
                combination_strategy="none",
                reasoning="Query is simple enough to answer directly"
            )

        # Use LLM for decomposition
        if self.use_llm:
            return self._llm_decompose(query, query_type)
        else:
            return self._rule_based_decompose(query, query_type)

    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of complex query."""
        import re

        for qtype, patterns in self.COMPLEXITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return qtype

        # Check word count and structure
        words = query.split()
        if len(words) > 12:
            # Long queries often need decomposition
            if ' و ' in query:  # Contains "and"
                return QueryType.MULTI_HOP

        return QueryType.SIMPLE

    def _llm_decompose(
        self,
        query: str,
        query_type: QueryType
    ) -> DecompositionResult:
        """Use LLM for intelligent query decomposition."""

        # Build type-specific prompt
        type_guidance = self._get_type_guidance(query_type)

        system_prompt = f"""أنت محلل استعلامات متخصص في تقسيم الأسئلة المعقدة إلى أسئلة فرعية بسيطة.

## مهمتك:
تحليل الاستعلام وتقسيمه إلى أسئلة فرعية يمكن الإجابة عليها بشكل مستقل.

{type_guidance}

## قواعد التقسيم:
1. كل سؤال فرعي يجب أن يكون بسيطاً ومستقلاً
2. رتّب الأسئلة بحيث تُبنى على إجابات الأسئلة السابقة
3. حدد الكيانات المستهدفة في كل سؤال
4. اشرح لماذا كل سؤال ضروري

## تنسيق الإخراج (JSON):
{{
  "needs_decomposition": true/false,
  "sub_queries": [
    {{
      "query": "السؤال الفرعي",
      "type": "factual/relational/aggregation",
      "depends_on": [0, 1],  // أرقام الأسئلة التي يعتمد عليها
      "target_entities": ["كيان 1", "كيان 2"],
      "reasoning": "سبب هذا السؤال"
    }}
  ],
  "combination_strategy": "sequential/parallel/merge",
  "reasoning": "شرح عام للتقسيم"
}}"""

        user_prompt = f"""حلل وقسّم الاستعلام التالي:

الاستعلام: {query}

نوع الاستعلام المكتشف: {query_type.value}

قم بتقسيمه إلى أسئلة فرعية إذا كان معقداً، أو أشر أنه لا يحتاج تقسيم إذا كان بسيطاً.

JSON:"""

        try:
            response = requests.post(
                f"{self.llm_endpoint}/v1/chat/completions",
                json={
                    "model": "tgi",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                    "top_p": 0.9
                },
                timeout=30
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            # Parse JSON
            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content)

            # Convert to DecompositionResult
            sub_queries = []
            for sq in result.get("sub_queries", []):
                sub_queries.append(SubQuery(
                    query=sq.get("query", ""),
                    query_type=sq.get("type", "factual"),
                    depends_on=sq.get("depends_on", []),
                    target_entities=sq.get("target_entities", []),
                    reasoning=sq.get("reasoning", "")
                ))

            # Calculate execution order
            execution_order = self._calculate_execution_order(sub_queries)

            return DecompositionResult(
                original_query=query,
                query_type=query_type,
                needs_decomposition=result.get("needs_decomposition", len(sub_queries) > 0),
                sub_queries=sub_queries,
                execution_order=execution_order,
                combination_strategy=result.get("combination_strategy", "sequential"),
                reasoning=result.get("reasoning", "")
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse decomposition JSON: {e}")
            return self._rule_based_decompose(query, query_type)
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            return self._rule_based_decompose(query, query_type)

    def _get_type_guidance(self, query_type: QueryType) -> str:
        """Get type-specific decomposition guidance."""
        guidance = {
            QueryType.MULTI_HOP: """## إرشادات للأسئلة متعددة القفزات:
- قسّم إلى سلسلة من الأسئلة المترابطة
- السؤال الأول يحدد الكيان الأساسي
- الأسئلة التالية تتبع العلاقات""",

            QueryType.COMPARISON: """## إرشادات لأسئلة المقارنة:
- سؤال منفصل لكل كيان للمقارنة
- سؤال أخير يجمع المقارنة""",

            QueryType.AGGREGATION: """## إرشادات لأسئلة التجميع:
- حدد المعايير المطلوبة
- سؤال لكل معيار أو فئة""",

            QueryType.CONDITIONAL: """## إرشادات للأسئلة الشرطية:
- سؤال لتحديد الشرط
- سؤال لتحديد النتيجة""",

            QueryType.TEMPORAL: """## إرشادات للأسئلة الزمنية:
- سؤال لتحديد الحدث/الفترة
- سؤال للمعلومات المرتبطة بالوقت""",

            QueryType.CAUSAL: """## إرشادات للأسئلة السببية:
- سؤال لتحديد السبب
- سؤال لتحديد النتيجة
- سؤال للعلاقة بينهما""",
        }
        return guidance.get(query_type, "")

    def _rule_based_decompose(
        self,
        query: str,
        query_type: QueryType
    ) -> DecompositionResult:
        """Rule-based decomposition fallback."""
        import re

        sub_queries = []

        # Split on common conjunctions
        parts = re.split(r'\s+و\s+|\s+وكذلك\s+|\s+وأيضاً\s+', query)

        if len(parts) > 1:
            for i, part in enumerate(parts):
                part = part.strip()
                if len(part) > 5:
                    sub_queries.append(SubQuery(
                        query=part,
                        query_type="factual",
                        depends_on=[],
                        target_entities=[],
                        reasoning=f"Part {i+1} of conjunction split"
                    ))

        # For comparison queries, extract entities
        if query_type == QueryType.COMPARISON and not sub_queries:
            match = re.search(r'(?:مقارنة|الفرق)\s+بين\s+(.+?)\s+و\s+(.+?)(?:\?|$)', query)
            if match:
                entity1, entity2 = match.groups()
                sub_queries = [
                    SubQuery(
                        query=f"ما هو {entity1}؟",
                        query_type="factual",
                        depends_on=[],
                        target_entities=[entity1],
                        reasoning="Get information about first entity"
                    ),
                    SubQuery(
                        query=f"ما هو {entity2}؟",
                        query_type="factual",
                        depends_on=[],
                        target_entities=[entity2],
                        reasoning="Get information about second entity"
                    ),
                ]

        return DecompositionResult(
            original_query=query,
            query_type=query_type,
            needs_decomposition=len(sub_queries) > 0,
            sub_queries=sub_queries,
            execution_order=list(range(len(sub_queries))),
            combination_strategy="sequential" if sub_queries else "none",
            reasoning="Rule-based decomposition"
        )

    def _calculate_execution_order(self, sub_queries: List[SubQuery]) -> List[int]:
        """Calculate optimal execution order based on dependencies."""
        if not sub_queries:
            return []

        # Topological sort based on depends_on
        order = []
        visited = set()

        def visit(idx: int):
            if idx in visited:
                return
            visited.add(idx)
            for dep in sub_queries[idx].depends_on:
                if dep < len(sub_queries):
                    visit(dep)
            order.append(idx)

        for i in range(len(sub_queries)):
            visit(i)

        return order


class DecomposeAndRetrieve:
    """
    Orchestrates decomposition + retrieval for complex queries.

    Usage:
        dar = DecomposeAndRetrieve(retriever)
        result = dar.query("ما علاقة X ب Y وكيف يؤثر على Z؟")
    """

    def __init__(
        self,
        retriever,
        decomposer: Optional[QueryDecomposer] = None
    ):
        """
        Initialize decompose-and-retrieve.

        Args:
            retriever: Retrieval engine to use
            decomposer: Query decomposer (creates one if not provided)
        """
        self.retriever = retriever
        self.decomposer = decomposer or QueryDecomposer()

    def query(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Decompose query if needed and retrieve context.

        Args:
            query: User query
            top_k: Number of results per sub-query

        Returns:
            Combined retrieval results
        """
        # Decompose
        decomposition = self.decomposer.decompose(query)

        if not decomposition.needs_decomposition:
            # Simple query - direct retrieval
            result = self.retriever.retrieve(query, top_k=top_k)
            return {
                "query": query,
                "decomposed": False,
                "sub_results": [],
                "combined_context": result,
                "decomposition": None
            }

        # Execute sub-queries in order
        sub_results = []
        accumulated_context = []

        for idx in decomposition.execution_order:
            sq = decomposition.sub_queries[idx]

            # Retrieve for sub-query
            result = self.retriever.retrieve(sq.query, top_k=top_k)

            sub_results.append({
                "sub_query": sq.query,
                "query_type": sq.query_type,
                "target_entities": sq.target_entities,
                "result": result
            })

            # Accumulate context
            if hasattr(result, 'chunks'):
                accumulated_context.extend(result.chunks[:3])

        # Combine results based on strategy
        combined = self._combine_results(
            sub_results,
            decomposition.combination_strategy
        )

        return {
            "query": query,
            "decomposed": True,
            "decomposition_reasoning": decomposition.reasoning,
            "sub_results": sub_results,
            "combined_context": combined,
            "decomposition": decomposition
        }

    def _combine_results(
        self,
        sub_results: List[Dict],
        strategy: str
    ) -> List[Dict]:
        """Combine sub-query results."""
        if strategy == "parallel":
            # Merge all results, deduplicate
            all_chunks = []
            seen_ids = set()

            for sr in sub_results:
                result = sr.get("result")
                if hasattr(result, 'chunks'):
                    for chunk in result.chunks:
                        chunk_id = chunk.get('id') if isinstance(chunk, dict) else getattr(chunk, 'id', None)
                        if chunk_id and chunk_id not in seen_ids:
                            seen_ids.add(chunk_id)
                            all_chunks.append(chunk)

            return all_chunks

        elif strategy == "sequential":
            # Keep in order
            all_chunks = []
            for sr in sub_results:
                result = sr.get("result")
                if hasattr(result, 'chunks'):
                    all_chunks.extend(result.chunks[:3])
            return all_chunks

        else:  # merge
            # Weighted combination (first sub-queries more important)
            all_chunks = []
            for i, sr in enumerate(sub_results):
                result = sr.get("result")
                weight = max(1, 3 - i)  # First gets more chunks
                if hasattr(result, 'chunks'):
                    all_chunks.extend(result.chunks[:weight])
            return all_chunks


# Factory function
def get_query_decomposer(**kwargs) -> QueryDecomposer:
    """Get query decomposer instance."""
    return QueryDecomposer(**kwargs)


def decompose_query(query: str, **kwargs) -> DecompositionResult:
    """Convenience function to decompose a query."""
    decomposer = QueryDecomposer(**kwargs)
    return decomposer.decompose(query)
