"""
Local Retrieval Mode

GraphRAG Local Search: Entity-focused retrieval with semantic entity matching.

Per Microsoft GraphRAG specification:
1. Uses entity description embeddings for semantic matching
2. Builds comprehensive entity context (description, relationships, communities)
3. Integrates community reports for additional context
4. Combines entity-linked chunks with vector search results

MIRAGE V4+ Enhanced with true GraphRAG Local Search algorithm.
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
        GraphRAG Local Search: Entity-focused retrieval with semantic entity matching.

        Per Microsoft GraphRAG specification:
        1. Find semantically similar entities using query embedding
        2. Build entity context (description, relationships, communities)
        3. Get text chunks linked to relevant entities
        4. Combine with vector search for comprehensive results
        5. Include entity descriptions for better LLM context

        Args:
            query: User query
            query_embedding: Query embedding vector
            top_k: Maximum results to return

        Returns:
            RetrievalResponse with entity-enriched results
        """
        if query_embedding is None or self.index_manager is None:
            return self._vector_retrieve(query, query_embedding, top_k, **kwargs)

        # 1. First get vector search results as base
        vector_response = self._vector_retrieve(query, query_embedding, top_k * 2, **kwargs)

        # 2. Detect entity types from query patterns
        entity_types = self._detect_entity_types(query)

        # =====================================================================
        # GraphRAG Enhancement: Semantic Entity Search using Embeddings
        # Per Microsoft GraphRAG spec, find entities semantically similar to query
        # =====================================================================
        semantic_entities = []
        entity_contexts = {}  # Cache entity context for later use

        if self.graph_client and query_embedding is not None:
            try:
                # Search entities by embedding similarity (new GraphRAG method)
                if hasattr(self.graph_client, 'search_entities_by_embedding'):
                    semantic_entities = self.graph_client.search_entities_by_embedding(
                        query_embedding=query_embedding,
                        entity_types=entity_types if entity_types else None,
                        limit=self.config.max_entity_search_results,
                        threshold=0.3
                    )
                    logger.debug(f"GraphRAG semantic entity search: found {len(semantic_entities)} entities")

                    # Build rich context for top entities - use BATCH method to avoid N+1
                    if semantic_entities:
                        entity_names = [
                            e.get("name", "") for e in semantic_entities[:self.config.max_entities_to_process]
                            if e.get("name")
                        ]
                        if entity_names:
                            if hasattr(self.graph_client, 'get_entities_with_context_batch'):
                                # Batch fetch - single query for all entities
                                entity_contexts = self.graph_client.get_entities_with_context_batch(entity_names)
                                logger.debug(f"Batch fetched {len(entity_contexts)} entity contexts")
                            elif hasattr(self.graph_client, 'get_entity_with_context'):
                                # Fallback to individual fetches (less efficient)
                                for entity_name in entity_names:
                                    ctx = self.graph_client.get_entity_with_context(entity_name)
                                    if ctx:
                                        entity_contexts[entity_name] = ctx

            except Exception as e:
                logger.warning(f"Semantic entity search failed: {e}")

        # 1b. Keyword-based search fallback for Arabic entities
        entity_phrases = self._extract_arabic_entity_phrases(query)
        keyword_chunks = []
        if entity_phrases and self.index_manager:
            try:
                # Batch search: collect all variants and search in single pass
                all_variants = []
                phrase_to_original = {}  # Map variant back to original phrase
                for phrase in entity_phrases[:self.config.max_entity_phrases]:
                    if len(phrase) >= 3:
                        variants = get_arabic_variants(phrase)
                        for variant in variants:
                            all_variants.append(variant)
                            phrase_to_original[variant] = phrase

                if all_variants:
                    # Single batch search instead of N individual searches
                    batch_results = self.index_manager.batch_keyword_search(
                        keywords=all_variants,
                        limit_per_keyword=self.config.keyword_limit_per_variant,
                        total_limit=self.config.keyword_batch_total_limit
                    )

                    seen_chunks = set()
                    for variant, matches in batch_results.items():
                        original_phrase = phrase_to_original.get(variant, variant)
                        for kr in matches:
                            chunk_id = kr.get("chunk_id", kr.get("id", ""))
                            if chunk_id and chunk_id not in seen_chunks:
                                seen_chunks.add(chunk_id)
                                keyword_chunks.append(RetrievalResult(
                                    chunk_id=chunk_id,
                                    document_id=kr.get("document_id", ""),
                                    text=kr.get("text", ""),
                                    score=self.config.keyword_match_score_local,
                                    retrieval_mode="keyword",
                                    metadata={"keyword_match": original_phrase, "matched_variant": variant}
                                ))
            except Exception as e:
                logger.debug(f"Keyword search failed: {e}")

        # Merge keyword results
        result_map = {r.chunk_id: r for r in vector_response.results if r.chunk_id}
        added_count = 0
        boosted_count = 0

        for kr in keyword_chunks:
            if not kr.chunk_id:
                continue
            if kr.chunk_id in result_map:
                existing = result_map[kr.chunk_id]
                old_score = existing.score
                existing.score = max(existing.score, self.config.keyword_boost_score)
                existing.metadata = existing.metadata or {}
                existing.metadata["keyword_boost"] = True
                existing.metadata["keyword_match"] = kr.metadata.get("keyword_match", "")
                boosted_count += 1
                logger.debug(f"Keyword boost: {kr.chunk_id} {old_score:.2f} → {existing.score:.2f}")
            else:
                vector_response.results.insert(0, kr)
                result_map[kr.chunk_id] = kr
                added_count += 1

        if added_count > 0 or boosted_count > 0:
            vector_response.results.sort(key=lambda x: x.score, reverse=True)
            logger.debug(f"Keyword search: added={added_count}, boosted={boosted_count}")

        # 3. Extract entities from retrieved chunks
        # Note: entity_types already detected above (before semantic entity search)
        extracted_entities = []
        entity_chunks = []
        disambiguated_entities = []

        # Add semantic entities from GraphRAG embedding search
        for sem_entity in semantic_entities:
            extracted_entities.append({
                "name": sem_entity.get("name", ""),
                "type": sem_entity.get("type", ""),
                "confidence": sem_entity.get("confidence", 0.5),
                "similarity": sem_entity.get("similarity", 0.0),
                "description": sem_entity.get("description", ""),
                "source": "semantic_embedding"  # Mark as from embedding search
            })

        # =====================================================================
        # GraphRAG Enhancement: Get chunks for semantic entities FIRST
        # These are high-priority since they're semantically matched to query
        # =====================================================================
        for sem_entity in semantic_entities[:self.config.max_entities_to_process]:
            entity_name = sem_entity.get("name", "")
            if entity_name:
                chunks = self._get_cached_entity_chunks(entity_name, limit=self.config.chunks_per_entity)
                for chunk in chunks:
                    entity_chunks.append({
                        "chunk_id": chunk.get("chunk_id", ""),
                        "document_id": chunk.get("document_id", ""),
                        "text": chunk.get("text", ""),
                        "entity": entity_name,
                        "entity_type": sem_entity.get("type", ""),
                        "entity_description": sem_entity.get("description", ""),
                        "semantic_match": True,
                        "match_score": sem_entity.get("similarity", 0.5)
                    })

        if self.graph_client and vector_response.results:
            try:
                chunk_ids = [r.chunk_id for r in vector_response.results if r.chunk_id]
                if chunk_ids:
                    entities = self.graph_client.get_entities_from_chunks(
                        chunk_ids=chunk_ids,
                        entity_types=entity_types if entity_types else None,
                        limit=self.config.max_entity_search_results
                    )
                    # Extend extracted_entities (don't overwrite semantic entities)
                    existing_names = {e.get("name", "") for e in extracted_entities}
                    for entity in entities:
                        if entity.get("name", "") not in existing_names:
                            extracted_entities.append(entity)

                    for entity in entities[:self.config.max_entities_to_process]:
                        chunks = self._get_cached_entity_chunks(
                            entity.get("name", ""), limit=self.config.chunks_per_entity
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
                    for term in query_terms[:self.config.max_disambiguate_terms]:
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
                                    result.matched_entity, limit=self.config.keyword_limit_per_variant
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
                    for term in query_terms[:self.config.max_entity_phrases]:
                        entities = self.graph_client.search_entities_by_name(
                            term, limit=self.config.keyword_limit_per_variant
                        )
                        for entity in entities:
                            if entity not in extracted_entities:
                                extracted_entities.append(entity)
                            chunks = self._get_cached_entity_chunks(
                                entity.get("name", ""), limit=self.config.chunks_per_entity
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

        # 4. Text content fallback: Boost vector results containing entity phrases
        entity_phrases = self._extract_arabic_entity_phrases(query)
        for r in vector_response.results:
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
                        "match_score": self.config.text_match_score
                    })
                    break

        # 5. Combine with vector search results
        # GraphRAG Enhancement: Sort by semantic match first, then disambiguated, then score
        results = []
        seen_chunks = set()

        entity_chunks_sorted = sorted(
            entity_chunks,
            key=lambda x: (
                x.get("semantic_match", False),  # Semantic embedding matches first
                x.get("disambiguated", False),   # Then disambiguated entities
                x.get("match_score", 0.5)        # Then by similarity score
            ),
            reverse=True
        )

        for ec in entity_chunks_sorted[:top_k // 2]:
            if ec["chunk_id"] and ec["chunk_id"] not in seen_chunks:
                seen_chunks.add(ec["chunk_id"])
                # Score based on source: semantic > disambiguated > regular
                if ec.get("semantic_match"):
                    base_score = 0.92  # Higher base for semantic matches
                elif ec.get("disambiguated"):
                    base_score = self.config.disambiguated_chunk_score
                else:
                    base_score = self.config.entity_chunk_base_score
                match_score = ec.get("match_score", 0.5)
                final_score = min(self.config.entity_chunk_max_score, base_score * match_score + (0.1 if ec.get("semantic_match") or ec.get("disambiguated") else 0))

                # Build metadata with entity description (GraphRAG enhancement)
                chunk_metadata = {
                    "entity_type": ec.get("entity_type", ""),
                    "match_type": "semantic" if ec.get("semantic_match") else ("disambiguated" if ec.get("disambiguated") else "chunk_linked")
                }
                if ec.get("entity_description"):
                    chunk_metadata["entity_description"] = ec["entity_description"]

                results.append(RetrievalResult(
                    chunk_id=ec["chunk_id"],
                    document_id=ec["document_id"],
                    text=ec["text"],
                    score=final_score,
                    retrieval_mode="local",
                    via_entity=ec["entity"],
                    hop_distance=1,
                    metadata=chunk_metadata
                ))

        for r in vector_response.results:
            if r.chunk_id not in seen_chunks and len(results) < top_k:
                seen_chunks.add(r.chunk_id)
                r.retrieval_mode = "local"
                results.append(r)

        entity_names = list({e.get("name", "") for e in extracted_entities if e.get("name")})

        # SLM ADAPTATION: Inject graph context
        if self.graph_client and entity_names:
            try:
                relationships = self.graph_client.get_relationships_between(
                    entity_names[:self.config.max_entities_to_process],
                    limit=self.config.max_relationships
                )
                for rel in relationships:
                    rel_text = f"{rel['source']} {rel['type'].replace('_', ' ')} {rel['target']}"
                    if rel.get('description'):
                        rel_text += f": {rel['description']}"
                    results.append(RetrievalResult(
                        chunk_id=f"rel_{hash(rel_text)}",
                        document_id="graph_context",
                        text=f"[Graph Relationship] {rel_text}",
                        score=self.config.graph_relationship_score,
                        retrieval_mode="graph_relationship",
                        metadata={"source": rel["source"], "target": rel["target"], "type": rel["type"]}
                    ))

                communities = self.graph_client.get_entity_communities(
                    entity_names[:self.config.max_communities], level=0
                )
                for comm in communities:
                    summary_text = f"Community Context: {comm.get('title', 'Group')} - {comm.get('summary', '')}"
                    if len(summary_text) > 400:
                        summary_text = summary_text[:400] + "..."
                    results.append(RetrievalResult(
                        chunk_id=f"comm_{comm.get('community_id')}",
                        document_id="community_context",
                        text=f"[Community Summary] {summary_text}",
                        score=self.config.community_context_score,
                        retrieval_mode="community_context",
                        metadata={"community_id": comm.get("community_id")}
                    ))
            except Exception as e:
                logger.warning(f"Failed to inject graph context: {e}")

        # Build entity descriptions for metadata (GraphRAG enhancement)
        entity_descriptions = {}
        for entity in extracted_entities:
            name = entity.get("name", "")
            desc = entity.get("description", "")
            if name and desc:
                entity_descriptions[name] = desc

        # Add entity contexts from GraphRAG semantic search
        for name, ctx in entity_contexts.items():
            if ctx.get("description") and name not in entity_descriptions:
                entity_descriptions[name] = ctx.get("description", "")

        return RetrievalResponse(
            results=results[:top_k + 5],
            query=query,
            mode=RetrievalMode.LOCAL,
            total_candidates=len(entity_chunks) + len(vector_response.results),
            metadata={
                "entities_found": len(extracted_entities),
                "entity_names": entity_names[:20],
                "entity_types_filtered": entity_types or [],
                "disambiguated_entities": disambiguated_entities[:10] if disambiguated_entities else [],
                "disambiguation_enabled": self.entity_disambiguator is not None,
                "graph_context_added": True,
                # GraphRAG enhancements
                "semantic_entities_found": len(semantic_entities),
                "entity_descriptions": entity_descriptions,
                "entity_contexts_loaded": len(entity_contexts)
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
