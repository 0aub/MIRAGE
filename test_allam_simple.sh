#!/bin/bash

# Simple Allam quality test using curl
# Tests 3 critical tasks for GraphRAG

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TGI_ENDPOINT="http://localhost:8765"

echo -e "${BLUE}================================================================================"
echo "ALLaM-7B Quality Assessment for GraphRAG Implementation"
echo -e "================================================================================${NC}\n"

# Test 1: Entity Extraction
echo -e "${BLUE}TEST 1: Entity Extraction (Arabic)${NC}\n"

PROMPT1='استخرج جميع الكيانات من النص التالي.

النص: أعلن الرئيس محمد في القاهرة عن مبادرة جديدة للتعامل مع تغير المناخ في مصر والشرق الأوسط.

لكل كيان، حدد:
- الاسم
- النوع (Person, Organization, Location, Concept, Date, etc.)

أعد القائمة بصيغة JSON:
{"entities": [{"name": "اسم الكيان", "type": "النوع"}]}'

echo "Test: Arabic entity extraction..."
RESPONSE1=$(curl -s -X POST "$TGI_ENDPOINT/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": \"$PROMPT1\",
    \"parameters\": {
      \"max_new_tokens\": 300,
      \"temperature\": 0.3,
      \"top_p\": 0.95
    }
  }" | python3 -c "import sys, json; print(json.load(sys.stdin).get('generated_text', ''))")

echo -e "Response:\n$RESPONSE1\n"

# Check if entities were extracted
if echo "$RESPONSE1" | grep -q "entities"; then
    echo -e "${GREEN}✓ Entity extraction: JSON format found${NC}\n"
    ENTITY_SCORE=75
else
    echo -e "${RED}✗ Entity extraction: No valid JSON${NC}\n"
    ENTITY_SCORE=40
fi

# Test 2: Summarization
echo -e "${BLUE}TEST 2: Summarization${NC}\n"

PROMPT2='لخّص هذه المجموعة من الكيانات المترابطة.

الكيانات: التغير المناخي, مصر, الطاقة المتجددة, السد العالي, نهر النيل

العلاقات الرئيسية:
التغير المناخي → يؤثر على → مصر
مصر → تستثمر في → الطاقة المتجددة
السد العالي → يقع على → نهر النيل

التعليمات:
1. حدد الموضوع الرئيسي
2. اذكر الكيانات الأساسية
3. اشرح العلاقات المهمة
4. اكتب 2-3 فقرات

الملخص:'

echo "Test: Community summarization..."
RESPONSE2=$(curl -s -X POST "$TGI_ENDPOINT/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": \"$PROMPT2\",
    \"parameters\": {
      \"max_new_tokens\": 500,
      \"temperature\": 0.3,
      \"top_p\": 0.95
    }
  }" | python3 -c "import sys, json; print(json.load(sys.stdin).get('generated_text', ''))")

echo -e "Summary:\n$RESPONSE2\n"

# Check summary quality
WORD_COUNT=$(echo "$RESPONSE2" | wc -w)
if [ "$WORD_COUNT" -gt 30 ] && echo "$RESPONSE2" | grep -q "مصر"; then
    echo -e "${GREEN}✓ Summarization: Good quality ($WORD_COUNT words)${NC}\n"
    SUMMARY_SCORE=80
elif [ "$WORD_COUNT" -gt 15 ]; then
    echo -e "${YELLOW}⚠ Summarization: Marginal quality ($WORD_COUNT words)${NC}\n"
    SUMMARY_SCORE=65
else
    echo -e "${RED}✗ Summarization: Poor quality ($WORD_COUNT words)${NC}\n"
    SUMMARY_SCORE=40
fi

# Test 3: Question Answering
echo -e "${BLUE}TEST 3: Question Answering${NC}\n"

PROMPT3='السؤال: ما هو تأثير التغير المناخي على مصر؟

السياق:
يؤثر التغير المناخي بشكل كبير على مصر من خلال ارتفاع مستوى سطح البحر الذي يهدد دلتا النيل.
تستثمر مصر في الطاقة المتجددة لمواجهة تحديات التغير المناخي.
السد العالي يلعب دوراً مهماً في توفير الكهرباء والحماية من الفيضانات.

أجب على السؤال بناءً على السياق المقدم.
كن دقيقاً وموجزاً (2-3 فقرات).

الإجابة:'

echo "Test: Question answering with context..."
RESPONSE3=$(curl -s -X POST "$TGI_ENDPOINT/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"inputs\": \"$PROMPT3\",
    \"parameters\": {
      \"max_new_tokens\": 400,
      \"temperature\": 0.3,
      \"top_p\": 0.95
    }
  }" | python3 -c "import sys, json; print(json.load(sys.stdin).get('generated_text', ''))")

echo -e "Answer:\n$RESPONSE3\n"

# Check QA quality
if echo "$RESPONSE3" | grep -q "النيل\|البحر\|الطاقة"; then
    echo -e "${GREEN}✓ Question Answering: Relevant answer${NC}\n"
    QA_SCORE=80
elif [ $(echo "$RESPONSE3" | wc -w) -gt 20 ]; then
    echo -e "${YELLOW}⚠ Question Answering: Partially relevant${NC}\n"
    QA_SCORE=60
else
    echo -e "${RED}✗ Question Answering: Irrelevant or too short${NC}\n"
    QA_SCORE=35
fi

# Final Report
echo -e "${BLUE}================================================================================"
echo "FINAL TEST REPORT"
echo -e "================================================================================${NC}\n"

OVERALL_SCORE=$(( (ENTITY_SCORE + SUMMARY_SCORE + QA_SCORE) / 3 ))

echo "Results:"
echo "--------------------------------------------------------------------------------"
printf "Entity Extraction:      %3d/100\n" $ENTITY_SCORE
printf "Summarization:          %3d/100\n" $SUMMARY_SCORE
printf "Question Answering:     %3d/100\n" $QA_SCORE
echo "--------------------------------------------------------------------------------"
printf "Overall Average:        %3d/100\n\n" $OVERALL_SCORE

# Recommendation
echo "================================================================================"
if [ $OVERALL_SCORE -ge 75 ]; then
    echo -e "${GREEN}✓ RECOMMENDATION: Allam is GOOD ENOUGH - Proceed with implementation${NC}"
    echo "  Quality is acceptable for GraphRAG. Start Phase 1!"
    EXIT_CODE=0
elif [ $OVERALL_SCORE -ge 65 ]; then
    echo -e "${YELLOW}⚠ RECOMMENDATION: Allam is MARGINAL - Consider hybrid approach${NC}"
    echo "  Use Allam for simple tasks, GPT-4 for complex ones"
    echo "  Or test alternative models (Gemini, Claude)"
    EXIT_CODE=1
else
    echo -e "${RED}✗ RECOMMENDATION: Allam is INSUFFICIENT - Test alternative models${NC}"
    echo "  Try: GPT-4o-mini, Gemini 2.0 Flash, or Claude Haiku"
    EXIT_CODE=2
fi
echo "================================================================================"

exit $EXIT_CODE
