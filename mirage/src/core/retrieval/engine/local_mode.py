"""
Local Retrieval Mode

Entity-focused retrieval: query → chunks → entities → enriched context
MIRAGE V4 Enhanced L2 strategy with entity disambiguation.
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from ..base_retriever import RetrievalMode, RetrievalResult, RetrievalResponse
from .arabic_utils import get_arabic_variants


class LocalModeMixin:
    """Mixin providing local retrieval mode and entity extraction helpers"""

    def _local_retrieve(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        top_k: int,
        **kwargs
    ) -> RetrievalResponse:
        """
        Entity-focused retrieval: query → chunks → entities → enriched context

        MIRAGE V4 Enhanced L2 strategy:
        1. Get relevant chunks via vector search
        2. Extract entities from those chunks
        3. Use entity disambiguation (cross-encoder) for better matching
        4. Filter entities by type based on query patterns
        5. Include entity names in retrieval metadata for better answers
        """
        if query_embedding is None or self.index_manager is None:
            return self._naive_retrieve(query, query_embedding, top_k, **kwargs)

        # 1. First get naive results as base
        naive_response = self._naive_retrieve(query, query_embedding, top_k * 2, **kwargs)

        # 1b. Keyword-based search fallback for Arabic entities
        entity_phrases = self._extract_arabic_entity_phrases(query)
        keyword_chunks = []
        if entity_phrases and self.index_manager:
            try:
                seen_chunks = set()
                for phrase in entity_phrases[:5]:
                    if len(phrase) >= 3:
                        variants = get_arabic_variants(phrase)
                        for variant in variants:
                            keyword_results = self.index_manager.keyword_search(
                                variant, limit=3
                            )
                            for kr in keyword_results:
                                chunk_id = kr.get("chunk_id", kr.get("id", ""))
                                if chunk_id and chunk_id not in seen_chunks:
                                    seen_chunks.add(chunk_id)
                                    keyword_chunks.append(RetrievalResult(
                                        chunk_id=chunk_id,
                                        document_id=kr.get("document_id", ""),
                                        text=kr.get("text", ""),
                                        score=0.90,
                                        retrieval_mode="keyword",
                                        metadata={"keyword_match": phrase, "matched_variant": variant}
                                    ))
            except Exception as e:
                logger.debug(f"Keyword search failed: {e}")

        # Merge keyword results
        result_map = {r.chunk_id: r for r in naive_response.results if r.chunk_id}
        added_count = 0
        boosted_count = 0

        for kr in keyword_chunks:
            if not kr.chunk_id:
                continue
            if kr.chunk_id in result_map:
                existing = result_map[kr.chunk_id]
                old_score = existing.score
                existing.score = max(existing.score, 0.92)
                existing.metadata = existing.metadata or {}
                existing.metadata["keyword_boost"] = True
                existing.metadata["keyword_match"] = kr.metadata.get("keyword_match", "")
                boosted_count += 1
                logger.debug(f"Keyword boost: {kr.chunk_id} {old_score:.2f} → {existing.score:.2f}")
            else:
                naive_response.results.insert(0, kr)
                result_map[kr.chunk_id] = kr
                added_count += 1

        if added_count > 0 or boosted_count > 0:
            naive_response.results.sort(key=lambda x: x.score, reverse=True)
            logger.debug(f"Keyword search: added={added_count}, boosted={boosted_count}")

        # 2. Detect entity types from query patterns
        entity_types = self._detect_entity_types(query)

        # 3. Extract entities from retrieved chunks
        extracted_entities = []
        entity_chunks = []
        disambiguated_entities = []

        if self.graph_client and naive_response.results:
            try:
                chunk_ids = [r.chunk_id for r in naive_response.results if r.chunk_id]
                if chunk_ids:
                    entities = self.graph_client.get_entities_from_chunks(
                        chunk_ids=chunk_ids,
                        entity_types=entity_types if entity_types else None,
                        limit=20
                    )
                    extracted_entities = entities

                    for entity in entities[:10]:
                        chunks = self._get_cached_entity_chunks(
                            entity.get("name", ""), limit=2
                        )
                        for chunk in chunks:
                            entity_chunks.append({
                                "chunk_id": chunk.get("chunk_id", ""),
                                "document_id": chunk.get("document_id", ""),
                                "text": chunk.get("text", ""),
                                "entity": entity.get("name", ""),
                                "entity_type": entity.get("type", "")
                            })

                # MIRAGE V4: Entity disambiguation
                if self.entity_disambiguator:
                    query_terms = self._extract_arabic_entity_phrases(query)
                    logger.debug(f"Arabic entity phrases extracted: {query_terms}")
                    for term in query_terms[:8]:
                        try:
                            result = self.entity_disambiguator.disambiguate(
                                query_entity=term,
                                entity_type=entity_types[0] if entity_types else None,
                                context=query
                            )
                            if result.matched_entity:
                                disambiguated_entities.append({
                                    "query_term": term,
                                    "matched": result.matched_entity,
                                    "score": result.similarity_score,
                                    "match_type": result.match_type
                                })
                                chunks = self._get_cached_entity_chunks(
                                    result.matched_entity, limit=3
                                )
                                for chunk in chunks:
                                    entity_chunks.append({
                                        "chunk_id": chunk.get("chunk_id", ""),
                                        "document_id": chunk.get("document_id", ""),
                                        "text": chunk.get("text", ""),
                                        "entity": result.matched_entity,
                                        "entity_type": chunk.get("entity_type", ""),
                                        "disambiguated": True,
                                        "match_score": result.similarity_score
                                    })
                        except Exception as e:
                            logger.debug(f"Disambiguation failed for term '{term}': {e}")
                else:
                    # Fallback: direct term matching
                    query_terms = self._extract_arabic_entity_phrases(query)
                    for term in query_terms[:5]:
                        entities = self.graph_client.search_entities_by_name(term, limit=3)
                        for entity in entities:
                            if entity not in extracted_entities:
                                extracted_entities.append(entity)
                            chunks = self._get_cached_entity_chunks(
                                entity.get("name", ""), limit=2
                            )
                            for chunk in chunks:
                                entity_chunks.append({
                                    "chunk_id": chunk.get("chunk_id", ""),
                                    "document_id": chunk.get("document_id", ""),
                                    "text": chunk.get("text", ""),
                                    "entity": entity.get("name", ""),
                                    "entity_type": entity.get("type", "")
                                })
            except Exception as e:
                logger.warning(f"Neo4j entity search failed: {e}")

        # 4. Text content fallback: Boost naive results containing entity phrases
        entity_phrases = self._extract_arabic_entity_phrases(query)
        for r in naive_response.results:
            text_lower = r.text.lower() if r.text else ""
            for phrase in entity_phrases:
                if phrase in r.text or phrase in text_lower:
                    entity_chunks.append({
                        "chunk_id": r.chunk_id,
                        "document_id": r.document_id,
                        "text": r.text,
                        "entity": phrase,
                        "entity_type": "TextMatch",
                        "text_match": True,
                        "match_score": 0.85
                    })
                    break

        # 5. Combine with naive results
        results = []
        seen_chunks = set()

        entity_chunks_sorted = sorted(
            entity_chunks,
            key=lambda x: (x.get("disambiguated", False), x.get("match_score", 0.5)),
            reverse=True
        )

        for ec in entity_chunks_sorted[:top_k // 2]:
            if ec["chunk_id"] and ec["chunk_id"] not in seen_chunks:
                seen_chunks.add(ec["chunk_id"])
                base_score = 0.90 if ec.get("disambiguated") else 0.85
                match_score = ec.get("match_score", 0.5)
                final_score = min(0.95, base_score * match_score + (0.1 if ec.get("disambiguated") else 0))
                results.append(RetrievalResult(
                    chunk_id=ec["chunk_id"],
                    document_id=ec["document_id"],
                    text=ec["text"],
                    score=final_score,
                    retrieval_mode="local",
                    via_entity=ec["entity"],
                    hop_distance=1
                ))

        for r in naive_response.results:
            if r.chunk_id not in seen_chunks and len(results) < top_k:
                seen_chunks.add(r.chunk_id)
                r.retrieval_mode = "local"
                results.append(r)

        entity_names = list({e.get("name", "") for e in extracted_entities if e.get("name")})

        # SLM ADAPTATION: Inject graph context
        if self.graph_client and entity_names:
            try:
                relationships = self.graph_client.get_relationships_between(entity_names[:10], limit=15)
                for rel in relationships:
                    rel_text = f"{rel['source']} {rel['type'].replace('_', ' ')} {rel['target']}"
                    if rel.get('description'):
                        rel_text += f": {rel['description']}"
                    results.append(RetrievalResult(
                        chunk_id=f"rel_{hash(rel_text)}",
                        document_id="graph_context",
                        text=f"[Graph Relationship] {rel_text}",
                        score=0.85,
                        retrieval_mode="graph_relationship",
                        metadata={"source": rel["source"], "target": rel["target"], "type": rel["type"]}
                    ))

                communities = self.graph_client.get_entity_communities(entity_names[:5], level=0)
                for comm in communities:
                    summary_text = f"Community Context: {comm.get('title', 'Group')} - {comm.get('summary', '')}"
                    if len(summary_text) > 400:
                        summary_text = summary_text[:400] + "..."
                    results.append(RetrievalResult(
                        chunk_id=f"comm_{comm.get('community_id')}",
                        document_id="community_context",
                        text=f"[Community Summary] {summary_text}",
                        score=0.80,
                        retrieval_mode="community_context",
                        metadata={"community_id": comm.get("community_id")}
                    ))
            except Exception as e:
                logger.warning(f"Failed to inject graph context: {e}")

        return RetrievalResponse(
            results=results[:top_k + 5],
            query=query,
            mode=RetrievalMode.LOCAL,
            total_candidates=len(entity_chunks) + len(naive_response.results),
            metadata={
                "entities_found": len(extracted_entities),
                "entity_names": entity_names[:20],
                "entity_types_filtered": entity_types or [],
                "disambiguated_entities": disambiguated_entities[:10] if disambiguated_entities else [],
                "disambiguation_enabled": self.entity_disambiguator is not None,
                "graph_context_added": True
            }
        )

    def _detect_entity_types(self, query: str) -> List[str]:
        """Detect which entity types to filter based on query patterns"""
        query_lower = query.lower()

        if "من هم" in query_lower or "من هي" in query_lower:
            if any(p in query_lower for p in ["مرشح", "شريك", "جهة", "هيئة", "وزارة", "جائزة"]):
                return ["Organization", "Person"]

        org_patterns = [
            "الجهات", "المؤسسات", "الشركات",
            "الوزارات", "الهيئات", "المرشحون",
            "شريك", "الشركاء", "جائزة",
        ]
        if any(p in query_lower for p in org_patterns):
            return ["Organization"]

        person_patterns = [
            "من هو", "من هي",
            "الأشخاص", "الشخص", "المسؤول",
            "المدير", "الرئيس", "الوزير",
        ]
        if any(p in query_lower for p in person_patterns):
            return ["Person"]

        if "من هم" in query_lower:
            return ["Organization", "Person"]

        return []

    def _extract_arabic_entity_phrases(self, query: str) -> List[str]:
        """Extract entity phrases from query (Arabic and English)"""
        phrases = []
        query_clean = query.replace("؟", " ").replace("،", " ").replace("?", " ").strip()

        # Arabic prefixes
        prefixes = ["شركة", "شركه", "هيئة", "هيئه", "وزارة", "وزاره", "جامعة", "جامعه", "مركز", "جائزة", "جائزه"]
        for prefix in prefixes:
            pattern = rf"{prefix}\s+([^\s]+(?:\s+[^\s]+)?)"
            matches = re.findall(pattern, query_clean)
            for match in matches:
                full_phrase = f"{prefix} {match}".strip()
                phrases.append(full_phrase)
                phrases.append(match.strip())

        # English capitalized words
        english_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\b'
        english_matches = re.findall(english_pattern, query)
        for match in english_matches:
            phrases.append(match)
            for word in match.split():
                if len(word) > 1:
                    phrases.append(word)

        # English phrases with numbers
        english_number_pattern = r'\b([A-Z][a-z]*\s*\d{2,4})\b'
        number_matches = re.findall(english_number_pattern, query)
        phrases.extend(number_matches)

        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, query)
        phrases.extend(years)

        # Acronyms
        acronym_pattern = r'\b([A-Z]{2,6})\b'
        acronyms = re.findall(acronym_pattern, query)
        phrases.extend(acronyms)

        # Arabic proper nouns
        common_words = {"من", "هي", "هو", "ما", "هل", "في", "على", "إلى", "عن", "التي", "الذي", "هذا", "هذه", "تلك", "ذلك",
                        "what", "did", "win", "from", "the", "is", "are", "was", "were", "how", "who", "which"}
        words = query_clean.split()
        for word in words:
            word_lower = word.lower()
            if len(word) > 2 and word_lower not in common_words:
                if not word.startswith("ال") or len(word) > 4:
                    phrases.append(word)

        return list(set(phrases))
