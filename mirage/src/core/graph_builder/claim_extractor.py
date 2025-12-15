"""
Claim Extractor for GraphRAG
Extracts factual claims (subject-predicate-object) from text with evidence.

Microsoft GraphRAG defines claims as:
- Subject: The entity making or affected by the claim
- Predicate: The action or relationship
- Object: Target entity or value
- Status: Active, historical, proposed, disputed, etc.
- Start/End Date: Temporal bounds
- Description: Detailed explanation
- Evidence: Source text supporting the claim
- Confidence: Extraction confidence score

Usage:
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(text, language="ar")
"""

import os
import json
import requests
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger

from ...config import settings


@dataclass
class Claim:
    """A factual claim extracted from text"""
    claim_id: str  # Unique identifier
    subject: str  # Entity making/affected by claim
    subject_type: str  # Entity type
    predicate: str  # Action/relationship
    object: str  # Target entity or value
    object_type: str  # Entity type or "VALUE"
    status: str  # active, historical, proposed, disputed, uncertain
    start_date: Optional[str]  # When claim became true
    end_date: Optional[str]  # When claim ceased being true (None = ongoing)
    description: str  # Full description of the claim
    evidence: List[str]  # Source chunk IDs
    evidence_text: str  # Original text supporting the claim
    confidence: float  # 0.0 to 1.0
    language: str  # ar or en

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class ClaimExtractor:
    """
    Extract factual claims from text using LLM.

    Claims are structured facts that can be verified:
    - "Organization X launched Program Y in 2024"
    - "Person A leads Department B since 2020"
    - "Event X will be held at Location Y on Date Z"

    This is a key GraphRAG component for answering factual questions.
    """

    def __init__(self):
        """Initialize claim extractor"""
        self.provider = None
        self.model = None
        self.tgi_endpoint = None
        self.max_claims_per_chunk = 10  # Limit claims per chunk

        self._detect_provider()

    def _detect_provider(self) -> None:
        """Auto-detect which LLM provider to use"""

        # Check for TGI (local GPU) first
        if settings.use_tgi and settings.tgi_endpoint:
            self.provider = "tgi"
            self.model = "tgi"
            self.tgi_endpoint = settings.tgi_endpoint
            logger.info(f"ClaimExtractor using TGI at {settings.tgi_endpoint}")
            return

        # Check for API keys
        if settings.openai_api_key:
            self.provider = "openai"
            self.model = "gpt-4o-mini"
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            logger.info("ClaimExtractor using OpenAI")

        elif settings.anthropic_api_key:
            self.provider = "anthropic"
            self.model = "claude-3-haiku-20240307"
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
            logger.info("ClaimExtractor using Anthropic")

        elif settings.google_api_key:
            self.provider = "google"
            self.model = "gemini/gemini-2.0-flash-exp"
            os.environ["GOOGLE_API_KEY"] = settings.google_api_key
            logger.info("ClaimExtractor using Google")

        else:
            logger.warning("No LLM provider available for ClaimExtractor")
            self.provider = None

    def extract_claims(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        language: str = "auto"
    ) -> List[Claim]:
        """
        Extract factual claims from text.

        Args:
            text: Source text to extract claims from
            chunk_id: Optional chunk ID for evidence tracking
            language: Language code ("ar", "en", or "auto")

        Returns:
            List of Claim objects
        """
        if not self.provider:
            logger.error("No LLM provider available")
            return []

        # Detect language
        if language == "auto":
            arabic_ratio = len([c for c in text if '\u0600' <= c <= '\u06FF']) / max(len(text), 1)
            language = "ar" if arabic_ratio > 0.3 else "en"

        try:
            raw_claims = self._extract_claims_from_text(text, language)

            # Convert to Claim objects
            claims = []
            for i, raw in enumerate(raw_claims):
                claim_id = f"{chunk_id or 'unknown'}_{i}" if chunk_id else f"claim_{int(time.time())}_{i}"

                claim = Claim(
                    claim_id=claim_id,
                    subject=raw.get("subject", ""),
                    subject_type=raw.get("subject_type", "Entity"),
                    predicate=raw.get("predicate", ""),
                    object=raw.get("object", ""),
                    object_type=raw.get("object_type", "Entity"),
                    status=raw.get("status", "active"),
                    start_date=raw.get("start_date"),
                    end_date=raw.get("end_date"),
                    description=raw.get("description", ""),
                    evidence=[chunk_id] if chunk_id else [],
                    evidence_text=raw.get("evidence_text", ""),
                    confidence=raw.get("confidence", 0.7),
                    language=language
                )

                # Validate claim has required fields
                if claim.subject and claim.predicate and claim.object:
                    claims.append(claim)

            logger.info(f"Extracted {len(claims)} valid claims from text")
            return claims

        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            return []

    def _extract_claims_from_text(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract raw claims from text using LLM"""

        # Create language-specific prompts
        if language == "ar":
            system_prompt = """أنت مستخرج ادعاءات متخصص في GraphRAG. مهمتك استخراج الادعاءات الواقعية (الحقائق) من النص.

## ما هو الادعاء (Claim)؟
الادعاء هو حقيقة قابلة للتحقق تتكون من:
- **الفاعل (subject)**: الكيان الذي يقوم بالفعل أو يتأثر به
- **الفعل (predicate)**: العمل أو العلاقة
- **المفعول (object)**: الكيان أو القيمة المستهدفة
- **الحالة (status)**: active (حالي), historical (تاريخي), proposed (مقترح), disputed (متنازع عليه)
- **التاريخ**: متى بدأ/انتهى هذا الادعاء
- **الدليل**: النص الأصلي الذي يدعم الادعاء

## أنواع الادعاءات المطلوبة:
1. **إجراءات**: "أطلقت المنظمة X البرنامج Y في 2024"
2. **قيادة/إدارة**: "يرأس الشخص A الإدارة B منذ 2020"
3. **أحداث**: "سيُقام الحدث X في الموقع Y بتاريخ Z"
4. **إحصائيات**: "حققت المبادرة X نتيجة Y"
5. **سياسات**: "تتطلب السياسة X الإجراء Y"

## تنسيق الإخراج (JSON):
{"claims": [
  {
    "subject": "هيئة الحكومة الرقمية",
    "subject_type": "Organization",
    "predicate": "أطلقت",
    "object": "منتدى الحكومة الرقمية 2025",
    "object_type": "Event",
    "status": "active",
    "start_date": "2025",
    "end_date": null,
    "description": "قامت هيئة الحكومة الرقمية بإطلاق منتدى الحكومة الرقمية لعام 2025 كفعالية سنوية لمناقشة التحول الرقمي",
    "evidence_text": "أعلنت هيئة الحكومة الرقمية عن إطلاق منتدى الحكومة الرقمية 2025",
    "confidence": 0.95
  }
]}

## قواعد مهمة:
- استخرج فقط الادعاءات المدعومة بدليل واضح في النص
- لا تستنتج أو تخمن معلومات غير موجودة
- الثقة العالية (0.9+) للادعاءات الصريحة، المتوسطة (0.7-0.9) للضمنية
- التاريخ قد يكون null إذا لم يُذكر"""

            user_prompt = f"""استخرج جميع الادعاءات الواقعية من النص التالي.

لكل ادعاء، حدد:
- الفاعل والفعل والمفعول
- حالة الادعاء (active/historical/proposed)
- التواريخ إن وُجدت
- الدليل من النص الأصلي

النص: {text[:2000]}

JSON:"""

        else:  # English
            system_prompt = """You are a specialized claim extractor for GraphRAG. Your task is to extract factual claims from text.

## What is a Claim?
A claim is a verifiable fact consisting of:
- **subject**: The entity performing or affected by the action
- **predicate**: The action or relationship
- **object**: The target entity or value
- **status**: active (current), historical (past), proposed (future/planned), disputed
- **dates**: When this claim started/ended
- **evidence**: Original text supporting the claim

## Types of Claims to Extract:
1. **Actions**: "Organization X launched Program Y in 2024"
2. **Leadership**: "Person A leads Department B since 2020"
3. **Events**: "Event X will be held at Location Y on Date Z"
4. **Statistics**: "Initiative X achieved Result Y"
5. **Policies**: "Policy X requires Action Y"

## Output Format (JSON):
{"claims": [
  {
    "subject": "Digital Government Authority",
    "subject_type": "Organization",
    "predicate": "launched",
    "object": "Digital Government Forum 2025",
    "object_type": "Event",
    "status": "active",
    "start_date": "2025",
    "end_date": null,
    "description": "The Digital Government Authority launched the Digital Government Forum 2025 as an annual event to discuss digital transformation",
    "evidence_text": "The Digital Government Authority announced the launch of the Digital Government Forum 2025",
    "confidence": 0.95
  }
]}

## Important Rules:
- Only extract claims supported by clear evidence in the text
- Do NOT infer or guess information not present
- High confidence (0.9+) for explicit claims, medium (0.7-0.9) for implicit
- Dates may be null if not mentioned"""

            user_prompt = f"""Extract all factual claims from the following text.

For each claim, identify:
- Subject, predicate, and object
- Claim status (active/historical/proposed)
- Dates if mentioned
- Evidence from the original text

Text: {text[:2000]}

JSON:"""

        try:
            if self.provider == "tgi":
                response = requests.post(
                    f"{self.tgi_endpoint}/v1/chat/completions",
                    json={
                        "model": "tgi",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2048,
                        "top_p": 0.9
                    },
                    timeout=45
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            else:
                # Use LiteLLM for cloud providers
                from litellm import completion
                response = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content

            # Parse JSON with robust error handling
            content = content.strip()

            # Try multiple extraction methods
            claims = self._robust_json_parse(content)
            if claims is None:
                logger.warning(f"Could not parse claims JSON, raw content: {content[:200]}...")
                return []

            # Limit claims per chunk
            return claims[:self.max_claims_per_chunk]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse claims JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error calling LLM for claims: {e}")
            return []

    def _robust_json_parse(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Robustly parse JSON from LLM response, handling various formats and errors.

        Handles:
        - JSON wrapped in ```json ... ``` blocks
        - Direct JSON objects/arrays
        - Truncated JSON (tries to recover)
        - Mixed text and JSON
        """
        import re

        # Method 1: Extract from markdown code blocks
        if "```json" in content:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)
        elif "```" in content:
            match = re.search(r'```\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)

        content = content.strip()

        # Method 2: Try direct JSON parse
        try:
            result = json.loads(content)
            if isinstance(result, dict):
                return result.get("claims", [])
            elif isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # Method 3: Extract JSON object/array using regex
        json_patterns = [
            r'\{\s*"claims"\s*:\s*\[([\s\S]*)\]\s*\}',  # {"claims": [...]}
            r'\[([\s\S]*)\]',  # Direct array
        ]

        for pattern in json_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    # Try to parse the extracted portion
                    json_str = match.group(0)
                    result = json.loads(json_str)
                    if isinstance(result, dict):
                        return result.get("claims", [])
                    elif isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    # Try to fix truncated JSON
                    json_str = self._fix_truncated_json(match.group(0))
                    try:
                        result = json.loads(json_str)
                        if isinstance(result, dict):
                            return result.get("claims", [])
                        elif isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        pass

        # Method 4: Try to find and parse individual claim objects
        claim_pattern = r'\{[^{}]*"subject"[^{}]*"predicate"[^{}]*\}'
        claim_matches = re.findall(claim_pattern, content, re.DOTALL)
        if claim_matches:
            claims = []
            for claim_str in claim_matches:
                try:
                    claim = json.loads(claim_str)
                    if isinstance(claim, dict) and "subject" in claim:
                        claims.append(claim)
                except json.JSONDecodeError:
                    continue
            if claims:
                return claims

        return None

    def _fix_truncated_json(self, json_str: str) -> str:
        """
        Attempt to fix truncated JSON by closing unclosed brackets/braces.
        """
        # Count brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')

        # Add missing closers
        json_str = json_str.rstrip(',')  # Remove trailing comma
        json_str += '}' * (open_braces - close_braces)
        json_str += ']' * (open_brackets - close_brackets)

        return json_str

    def extract_claims_batch(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "auto"
    ) -> List[Claim]:
        """
        Extract claims from multiple chunks.

        Args:
            chunks: List of dicts with 'id' and 'text' keys
            language: Language code

        Returns:
            List of all claims from all chunks
        """
        all_claims = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            text = chunk.get("text", "")

            if not text:
                continue

            logger.info(f"Processing chunk {i+1}/{len(chunks)} for claims")
            claims = self.extract_claims(text, chunk_id=chunk_id, language=language)
            all_claims.extend(claims)

        logger.info(f"Total claims extracted: {len(all_claims)}")
        return all_claims


def store_claims_neo4j(neo4j_client, claims: List[Claim], document_id: str):
    """
    Store claims in Neo4j graph database.

    Creates:
    - Claim nodes with all properties
    - Links to Entity nodes (subject, object)
    - Links to Chunk nodes (evidence)
    - Links to Document node
    """
    if not claims:
        return 0

    stored = 0

    for claim in claims:
        try:
            # Create Claim node and link to entities
            query = """
            // Create or get Subject entity
            MERGE (subj:Entity {name: $subject})
            ON CREATE SET subj.type = $subject_type, subj.created_at = timestamp()

            // Create or get Object entity (if it's an entity, not a value)
            WITH subj
            CALL {
                WITH subj
                WITH subj
                WHERE $object_type <> 'VALUE'
                MERGE (obj:Entity {name: $object})
                ON CREATE SET obj.type = $object_type, obj.created_at = timestamp()
                RETURN obj
            }

            // Create Claim node
            WITH subj, obj
            MERGE (c:Claim {claim_id: $claim_id})
            ON CREATE SET
                c.subject = $subject,
                c.predicate = $predicate,
                c.object = $object,
                c.status = $status,
                c.start_date = $start_date,
                c.end_date = $end_date,
                c.description = $description,
                c.evidence_text = $evidence_text,
                c.confidence = $confidence,
                c.language = $language,
                c.created_at = timestamp()

            // Link Claim to Subject
            MERGE (subj)-[:IS_SUBJECT_OF]->(c)

            // Link Claim to Object (if entity)
            WITH c, obj
            WHERE obj IS NOT NULL
            MERGE (obj)-[:IS_OBJECT_OF]->(c)

            // Link Claim to Document
            WITH c
            MATCH (d:Document {document_id: $document_id})
            MERGE (d)-[:HAS_CLAIM]->(c)

            RETURN c.claim_id as claim_id
            """

            result = neo4j_client.execute_query(query, {
                "claim_id": claim.claim_id,
                "subject": claim.subject,
                "subject_type": claim.subject_type,
                "predicate": claim.predicate,
                "object": claim.object,
                "object_type": claim.object_type,
                "status": claim.status,
                "start_date": claim.start_date,
                "end_date": claim.end_date,
                "description": claim.description,
                "evidence_text": claim.evidence_text,
                "confidence": claim.confidence,
                "language": claim.language,
                "document_id": document_id
            })

            if result:
                stored += 1

            # Link to evidence chunks
            for chunk_id in claim.evidence:
                link_query = """
                MATCH (c:Claim {claim_id: $claim_id})
                MATCH (chunk:Chunk {id: $chunk_id})
                MERGE (chunk)-[:SUPPORTS]->(c)
                """
                neo4j_client.execute_query(link_query, {
                    "claim_id": claim.claim_id,
                    "chunk_id": chunk_id
                })

        except Exception as e:
            logger.error(f"Error storing claim {claim.claim_id}: {e}")

    logger.info(f"Stored {stored}/{len(claims)} claims in Neo4j")
    return stored
