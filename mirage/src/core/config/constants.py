"""
MIRAGE Centralized Configuration Constants

All magic numbers and thresholds with documented rationale.
Import from this file instead of hardcoding values.

Usage:
    from core.config.constants import (
        ENTITY_SIMILARITY_THRESHOLD,
        MAX_CONTEXT_TOKENS,
        RETRIEVAL_TOP_K_DEFAULT,
    )
"""

# =============================================================================
# ENTITY EXTRACTION & RESOLUTION
# =============================================================================

# Threshold for considering two entities as the same (cosine similarity)
# Tuned on Arabic NER benchmarks - higher = stricter matching
ENTITY_SIMILARITY_THRESHOLD = 0.85

# Confidence boost when multiple extraction methods agree
# 0.15 for 3+ sources, 0.10 for 2 sources
ENTITY_CONFIDENCE_BOOST_MULTI = 0.15
ENTITY_CONFIDENCE_BOOST_DUAL = 0.10

# Confidence boost for entities mentioned multiple times
ENTITY_FREQUENCY_BOOST = 0.10
ENTITY_FREQUENCY_THRESHOLD = 3  # Minimum mentions for boost

# Maximum confidence (never claim 100% certainty)
ENTITY_MAX_CONFIDENCE = 0.98

# Entity extraction source weights
ENTITY_WEIGHT_LLM = 0.85      # Highest - contextual understanding
ENTITY_WEIGHT_CAMEL = 0.75    # Arabic NER
ENTITY_WEIGHT_SPACY = 0.70    # English NER
ENTITY_WEIGHT_PATTERN = 0.65  # Rule-based patterns

# =============================================================================
# RELATIONSHIP EXTRACTION
# =============================================================================

# Co-occurrence window for relationship detection (characters)
COOCCURRENCE_WINDOW = 100

# Minimum confidence to create a relationship
MIN_RELATIONSHIP_CONFIDENCE = 0.3

# Maximum relationships per entity pair (prevents explosion)
MAX_RELATIONSHIPS_PER_PAIR = 5

# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

# Louvain resolution parameter (1.0 = standard, higher = more communities)
LOUVAIN_RESOLUTION = 1.0

# Minimum entities for a valid community
MIN_COMMUNITY_SIZE = 3

# Maximum community levels for hierarchical detection
MAX_COMMUNITY_LEVELS = 5

# Default community level for searches
DEFAULT_COMMUNITY_LEVEL = 0

# =============================================================================
# COMMUNITY SUMMARIZATION
# =============================================================================

# Maximum entities to include in community summary prompt
MAX_ENTITIES_IN_SUMMARY = 15

# Maximum relationships to include in community summary prompt
MAX_RELATIONSHIPS_IN_SUMMARY = 10

# Maximum tokens for generated summary
MAX_SUMMARY_TOKENS = 500

# LLM temperature for summarization (lower = more deterministic)
SUMMARIZATION_TEMPERATURE = 0.3

# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

# Default number of results to retrieve
RETRIEVAL_TOP_K_DEFAULT = 10

# Maximum allowed top_k
RETRIEVAL_TOP_K_MAX = 50

# Minimum score for retrieval results (0-1)
RETRIEVAL_MIN_SCORE = 0.0

# CRAG-style validation threshold (chunk relevance to query)
VALIDATION_RELEVANCE_THRESHOLD = 0.5

# Minimum ratio of relevant chunks to pass validation
VALIDATION_MIN_RELEVANT_RATIO = 0.4

# Maximum correction attempts in CRAG validation
VALIDATION_MAX_CORRECTIONS = 2

# =============================================================================
# FUSION & RANKING
# =============================================================================

# RRF parameter (standard value from literature)
RRF_K = 60

# Mode weights for hybrid fusion
MODE_WEIGHT_NAIVE = 0.6
MODE_WEIGHT_LOCAL = 0.8
MODE_WEIGHT_GLOBAL = 0.9
MODE_WEIGHT_HYBRID = 1.0
MODE_WEIGHT_SEMANTIC = 0.85

# =============================================================================
# REFRAG COMPRESSION
# =============================================================================

# Tokens per chunk for REFRAG embedding (Meta REFRAG default)
REFRAG_CHUNK_SIZE_TOKENS = 16

# Base compression ratio (50% = keep half)
REFRAG_COMPRESSION_RATIO_BASE = 0.5

# Importance threshold for minimal compression
REFRAG_IMPORTANCE_THRESHOLD = 0.7

# Aggressive compression ratio for low-importance content
REFRAG_AGGRESSIVE_RATIO = 0.3

# =============================================================================
# CONTEXT LIMITS (Allam 2K optimization)
# =============================================================================

# Maximum tokens for context (leave room for response)
MAX_CONTEXT_TOKENS = 1800

# Maximum tokens for response
MAX_RESPONSE_TOKENS = 200

# Maximum entities in retrieval context
MAX_ENTITIES_IN_CONTEXT = 15

# Maximum relationships in retrieval context
MAX_RELATIONSHIPS_IN_CONTEXT = 10

# Maximum communities to search in global search
MAX_COMMUNITIES_TO_SEARCH = 30

# Minimum relevance for global search communities
GLOBAL_SEARCH_MIN_RELEVANCE = 0.3

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

# Default chunk size (tokens)
CHUNK_SIZE_DEFAULT = 600

# Chunk overlap (tokens)
CHUNK_OVERLAP_DEFAULT = 50

# Semantic chunking similarity threshold
SEMANTIC_CHUNK_THRESHOLD = 0.7

# Minimum/maximum chunk size for semantic chunking
SEMANTIC_CHUNK_MIN = 100
SEMANTIC_CHUNK_MAX = 800

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

# Embedding dimension (Jina v3)
EMBEDDING_DIM = 1024

# Jina model for embeddings
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"

# Batch size for embedding
EMBEDDING_BATCH_SIZE = 32

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Default TGI endpoint
TGI_ENDPOINT_DEFAULT = "http://tgi:80"

# Default max tokens for generation
LLM_MAX_TOKENS_DEFAULT = 500

# Default temperature
LLM_TEMPERATURE_DEFAULT = 0.7

# Temperature for factual responses (lower)
LLM_TEMPERATURE_FACTUAL = 0.3

# Timeout for LLM requests (seconds)
LLM_REQUEST_TIMEOUT = 60

# =============================================================================
# EVALUATION METRICS TARGETS
# =============================================================================

# Speed targets
TARGET_TTFT_MS = 100
TARGET_E2E_LATENCY_MS = 3000
TARGET_THROUGHPUT_QPS = 10

# Retrieval accuracy targets
TARGET_MRR = 0.75
TARGET_NDCG_AT_10 = 0.65
TARGET_PRECISION_AT_5 = 0.55
TARGET_RECALL_AT_10 = 0.70

# RAG quality targets
TARGET_ANSWER_RELEVANCY = 0.88
TARGET_FAITHFULNESS = 0.90
TARGET_CONTEXT_RELEVANCY = 0.75

# Entity extraction targets
TARGET_ENTITY_F1 = 0.85
TARGET_LINKING_ACCURACY = 0.85

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

# Redis cache TTL (seconds)
CACHE_TTL_DEFAULT = 86400  # 24 hours

# Maximum cache size (entries)
CACHE_MAX_SIZE = 10000

# =============================================================================
# LOGGING & MONITORING
# =============================================================================

# Metrics tracking window size
METRICS_WINDOW_SIZE = 100

# Log level for debug mode
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_PRODUCTION = "INFO"
