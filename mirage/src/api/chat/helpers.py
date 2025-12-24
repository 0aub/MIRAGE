"""
Chat Service - Helper Functions
LLM response cleaning, Arabic processing, and reranking utilities
"""

import re
from typing import List
from loguru import logger


def clean_llm_response(answer: str) -> str:
    """
    Clean up LLM response artifacts:
    - Remove citation markers like [1], [2], etc.
    - Remove repeated content (question echo, etc.)
    - Clean up "I cannot find" patterns
    """
    if not answer:
        return answer

    # Remove citation markers
    answer = re.sub(r'\[\d+\]', '', answer)

    # Truncate at prompt echo patterns
    for pattern in ["Human:", "Question:", "Context:"]:
        if pattern in answer:
            answer = answer.split(pattern)[0].strip()

    # Handle "I cannot find" patterns
    if "I cannot find" in answer:
        parts = answer.split("I cannot find")
        answer = parts[0].strip()
        if not answer:
            return "لا أجد هذه المعلومات في السياق المقدم."

    # Clean up multiple spaces
    answer = re.sub(r'[ \t]+', ' ', answer).strip()

    # Remove trailing incomplete sentences
    if answer and answer[-1] not in '.؟!':
        for end_marker in ['. ', '؟ ', '! ']:
            if end_marker in answer:
                last_idx = answer.rfind(end_marker)
                if last_idx > len(answer) * 0.5:
                    answer = answer[:last_idx + 1]
                    break

    return answer


def is_no_info_response(answer: str) -> bool:
    """
    Detect if the LLM response indicates "no information found".
    Triggers fallback to chunk summarization.
    """
    if not answer or len(answer) < 30:
        return True

    pure_no_info_patterns = [
        # Arabic patterns
        "لا أجد أي معلومات",
        "لم أجد معلومات",
        "لا توجد معلومات متاحة",
        "لا تتوفر معلومات",
        "لم يتم العثور على أي",
        "لا يمكنني الإجابة",
        "لا أستطيع الإجابة",
        "لا يوجد في السياق المقدم",
        # English patterns
        "i cannot find any",
        "no information available",
        "unable to find any",
        "cannot answer this",
    ]

    answer_lower = answer.lower()

    if len(answer.strip()) < 80:
        if any(pattern in answer_lower for pattern in pure_no_info_patterns):
            return True

    if len(answer.strip()) > 100:
        return False

    if len(answer.strip()) < 50:
        return True

    if answer.count("؟") > 1:
        return True

    return False


def extract_key_terms(question: str) -> list:
    """Extract key terms (entities/nouns) from Arabic/English question."""
    arabic_stopwords = [
        "ما", "من", "هل", "كيف", "لماذا", "أين", "متى", "هي", "هو", "هذا", "هذه",
        "التي", "الذي", "في", "على", "إلى", "عن", "مع", "بين", "أو", "و", "ال",
        "أن", "إن", "كان", "يكون", "له", "لها", "هم", "هن", "نحن", "أنت", "أنا"
    ]

    words = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+', question)

    key_terms = []
    for word in words:
        if len(word) < 3:
            continue
        if word in arabic_stopwords:
            continue
        clean_word = word
        if clean_word.startswith('ال'):
            clean_word = clean_word[2:]
        if clean_word.startswith(('ب', 'و', 'ف', 'ك', 'ل')) and len(clean_word) > 3:
            clean_word = clean_word[1:]
        if len(clean_word) >= 2:
            key_terms.append(clean_word)

    return key_terms


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for matching."""
    if not text:
        return ""
    text = text.replace('ة', 'ه')
    text = re.sub(r'[أإآ]', 'ا', text)
    return text


def create_fallback_answer(question: str, chunks: list) -> str:
    """
    Create a fallback answer by summarizing the top chunks.
    Used when LLM fails to synthesize an answer.
    """
    if not chunks:
        return "لم يتم العثور على معلومات ذات صلة."

    key_terms = extract_key_terms(question)
    normalized_terms = [normalize_arabic(t) for t in key_terms]
    top_chunks = chunks[:3]
    answer_parts = []

    if key_terms:
        main_entity = key_terms[0] if key_terms else ""
        answer_parts.append(f"بخصوص {main_entity}، إليك المعلومات المتوفرة:")
    else:
        answer_parts.append("بناءً على المعلومات المتوفرة:")

    for i, chunk in enumerate(top_chunks, 1):
        text = chunk.get("text", "")
        if not text:
            continue

        normalized_text = normalize_arabic(text)
        best_snippet = None
        best_score = 0

        sentences = text.replace('،', '.').replace('؟', '.').split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            norm_sentence = normalize_arabic(sentence)
            score = sum(1 for t in normalized_terms if t in norm_sentence)

            if score > best_score:
                best_score = score
                best_snippet = sentence

        if best_snippet:
            snippet = best_snippet[:250].strip()
        else:
            snippet = text[:200].strip()

        if len(snippet) > 200:
            last_space = snippet.rfind(' ')
            if last_space > 100:
                snippet = snippet[:last_space]
            snippet += "..."

        answer_parts.append(f"\n• {snippet}")

    return "".join(answer_parts)


def rerank_chunks_by_keywords(question: str, results: list, boost_factor: float = 0.3) -> list:
    """
    Re-rank retrieval results by boosting chunks that contain query keywords.
    """
    if not results:
        return results

    key_terms = extract_key_terms(question)
    if not key_terms:
        return results

    normalized_terms = [normalize_arabic(t) for t in key_terms]

    boosted_results = []
    for r in results:
        text = r.text if hasattr(r, 'text') else ""
        normalized_text = normalize_arabic(text)

        match_count = sum(1 for t in normalized_terms if t in normalized_text)
        keyword_boost = min(match_count * boost_factor, 0.5)

        original_score = r.score if hasattr(r, 'score') else 0.5
        boosted_score = original_score + keyword_boost

        boosted_results.append((r, boosted_score, match_count))

    boosted_results.sort(key=lambda x: x[1], reverse=True)

    if boosted_results:
        reranked_count = sum(1 for i, (r, _, mc) in enumerate(boosted_results) if mc > 0)
        if reranked_count > 0:
            logger.debug(f"Re-ranked {reranked_count} chunks by keyword match ({key_terms[:3]})")

    return [r for r, _, _ in boosted_results]
