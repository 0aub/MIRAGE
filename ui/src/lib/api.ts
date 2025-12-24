/**
 * API Client for MIRAGE Backend
 * Base URL should be configured via environment variable
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiError extends Error {
  constructor(public status: number, message: string, public data?: any) {
    super(message);
    this.name = 'ApiError';
  }
}

async function fetchApi<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  console.log(`[API] Fetching: ${url}`);

  try {
    const startTime = Date.now();

    // Add 60-second timeout for long operations
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.error(`[API] Request timeout after 60s: ${url}`);
      controller.abort();
    }, 60000);

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Accept': 'application/json; charset=utf-8',
        ...options.headers,
      },
      signal: controller.signal,
      cache: 'no-store',
    });

    clearTimeout(timeoutId);
    const elapsed = Date.now() - startTime;
    console.log(`[API] Response received in ${elapsed}ms, status: ${response.status}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error(`[API] Error response:`, errorData);
      throw new ApiError(
        response.status,
        errorData.message || errorData.detail || `HTTP ${response.status}`,
        errorData
      );
    }

    const data = await response.json();
    return data;
  } catch (error: any) {
    console.error(`[API] Fetch error for ${url}:`, error);
    if (error instanceof ApiError) {
      throw error;
    }
    if (error.name === 'AbortError') {
      throw new ApiError(0, `Request timeout: The server took too long to respond`);
    }
    throw new ApiError(0, `Network error: ${error.message}`);
  }
}

// ========== FILE MANAGEMENT APIs ==========
export const filesApi = {
  list: (params?: { skip?: number; limit?: number; file_type?: string }) => {
    const query = new URLSearchParams();
    if (params?.skip !== undefined) query.set('skip', params.skip.toString());
    if (params?.limit !== undefined) query.set('limit', params.limit.toString());
    if (params?.file_type) query.set('file_type', params.file_type);

    return fetchApi<{
      total: number;
      files: Array<{
        file_id: string;
        filename: string;
        file_type: string;
        size: number;
        uploaded_at: string;
        path: string;
        mime_type?: string;
      }>;
    }>(`/files/files?${query}`);
  },

  upload: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/files/files/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(response.status, error.detail || 'Upload failed');
    }

    return response.json();
  },

  process: (fileId: string, extractEntities: boolean = true) => {
    return fetchApi<{
      job_id: string;
      file_id: string;
      status: string;
      message: string;
    }>(`/files/files/${fileId}/process`, {
      method: 'POST',
      body: JSON.stringify({ extract_entities: extractEntities }),
    });
  },

  delete: (fileId: string) => {
    return fetchApi<{ file_id: string; filename: string; message: string }>(
      `/files/files/${fileId}`,
      { method: 'DELETE' }
    );
  },

  stats: () => {
    return fetchApi<{
      total_files: number;
      total_size_bytes: number;
      total_size_mb: number;
      file_types: Record<string, number>;
      upload_directory: string;
    }>('/files/files/stats');
  },
};

// ========== DOCUMENTS REGISTRY APIs ==========
export const documentsApi = {
  list: (params?: { skip?: number; limit?: number; content_type?: string }) => {
    const query = new URLSearchParams();
    if (params?.skip !== undefined) query.set('skip', params.skip.toString());
    if (params?.limit !== undefined) query.set('limit', params.limit.toString());
    if (params?.content_type) query.set('content_type', params.content_type);

    return fetchApi<{
      total: number;
      documents: Array<{
        document_id: string;
        title: string;
        content_type: 'file' | 'webpage' | 'youtube';
        url?: string;
        author?: string;
        language?: string;
        entity_count: number;
        relationship_count: number;
        created_at?: number;
        total_chars?: number;
        total_words?: number;
        video_id?: string;
        transcript_length?: number;
        processing_time_seconds?: number;
      }>;
    }>(`/documents/documents?${query}`);
  },

  delete: (documentId: string) => {
    return fetchApi<{
      document_id: string;
      message: string;
      stats: any;
    }>(`/documents/documents/${documentId}`, { method: 'DELETE' });
  },

  getContent: (documentId: string, type: 'raw' | 'processed' = 'raw') => {
    return fetchApi<{ content: string }>(`/documents/documents/${documentId}/content?type=${type}`);
  },
};

// ========== GRAPH APIs ==========
export const graphApi = {
  visualizeFull: (limit: number = 50) => {
    return fetchApi<{
      nodes: Array<{
        id: string;
        label: string;
        type: string;
        confidence: number;
        degree?: number;
      }>;
      edges: Array<{
        source: string;
        target: string;
        relationship: string;
        confidence: number;
      }>;
      node_count: number;
      edge_count: number;
    }>(`/graph/visualize/full?limit=${limit}`);
  },

  search: (query: string, entityType?: string, limit: number = 20) => {
    const params = new URLSearchParams({ query, limit: limit.toString() });
    if (entityType) params.set('entity_type', entityType);
    return fetchApi<Array<any>>(`/graph/search?${params}`);
  },

  stats: () => {
    return fetchApi<{
      total_nodes: number;
      total_edges: number;
      node_types: Record<string, number>;
      edge_types: Record<string, number>;
      largest_component_size: number;
      average_degree: number;
    }>('/graph/stats');
  },
};

// ========== DATABASE MANAGEMENT APIs ==========
export const dbApi = {
  health: () => {
    return fetchApi<{
      overall_status: string;
      databases: Array<{
        database: string;
        status: string;
        connected: boolean;
        nodes?: number;
        edges?: number;
        vectors?: number;
        error?: string;
      }>;
    }>('/db/health');
  },

  stats: () => {
    return fetchApi<{
      graph: {
        status: string;
        total_nodes?: number;
        total_edges?: number;
        node_types?: Record<string, number>;
        edge_types?: Record<string, number>;
        document_count?: number;
        error?: string;
      };
      vector: {
        status: string;
        collection?: string;
        vectors_count?: number;
        points_count?: number;
        error?: string;
      };
    }>('/db/stats');
  },

  graphInfo: () => {
    return fetchApi<{
      total_nodes: number;
      total_edges: number;
      node_types: Record<string, number>;
      edge_types: Record<string, number>;
      document_count: number;
    }>('/db/graph/info');
  },

  vectorInfo: () => {
    return fetchApi<{
      collection_name: string;
      vectors_count: number;
      points_count: number;
      status: string;
    }>('/db/vector/info');
  },

  getSettings: () => {
    return fetchApi<{
      settings: {
        models?: any;
        processing?: any;
        databases?: any;
        chunking_strategies?: any;
        prompts?: Record<string, string>;
      };
      config_dir: string;
    }>('/db/settings');
  },

  vector: {
    getChunks: (limit: number = 50, offset: number = 0, documentId?: string) => {
      const params = new URLSearchParams();
      params.set('limit', limit.toString());
      params.set('offset', offset.toString());
      if (documentId) params.set('document_id', documentId);

      return fetchApi<{
        chunks: Array<{
          id: string;
          document_id: string;
          chunk_index: number;
          text: string;
          char_count: number;
          word_count: number;
          sentence_count: number;
          metadata: Record<string, any>;
        }>;
        total: number;
        limit: number;
        offset: number;
        next_offset: any;
        has_more: boolean;
      }>(`/db/vector/chunks?${params}`);
    },

    getStats: () => {
      return fetchApi<{
        total_chunks: number;
        total_documents: number;
        avg_chunk_size: number;
        total_vectors: number;
      }>('/db/vector/stats');
    },
  },
};

// ========== URL PROCESSING APIs ==========
export const urlApi = {
  preview: (url: string) => {
    return fetchApi<{
      url: string;
      title: string | null;
      description: string | null;
      content_preview: string;
      estimated_words: number;
      content_type: 'webpage' | 'youtube';
    }>('/url/preview', {
      method: 'POST',
      body: JSON.stringify({
        url,
        use_semantic_chunking: true,
      }),
    });
  },

  processAsync: (url: string, useSemanticChunking: boolean = true) => {
    return fetchApi<{
      job_id: string;
      status: string;
      message: string;
    }>('/url/process-async', {
      method: 'POST',
      body: JSON.stringify({
        url,
        use_semantic_chunking: useSemanticChunking,
      }),
    });
  },

  getJobStatus: (jobId: string) => {
    return fetchApi<{
      job_id: string;
      status: string;
      progress: number;
      current_phase?: string;
      document_id?: string;
      title?: string;
      content_type?: string;
      url?: string;
      created_at: number;
      started_at?: number;
      completed_at?: number;
      elapsed_seconds: number;
      error?: string;
    }>(`/url/jobs/${jobId}/status`);
  },

  getActiveJobs: () => {
    return fetchApi<{
      jobs: any[];
      count: number;
    }>('/url/jobs/active');
  },
};

// ========== RETRIEVAL TYPES ==========
export type RetrievalMode = 'naive' | 'local' | 'global' | 'hybrid' | 'semantic' | 'mix' | 'global_search' | 'drift';

export interface AskRequest {
  message: string;
  retrieval_mode?: RetrievalMode;
  top_k?: number;
  conversation_id?: string;
  // V5 Advanced options
  use_hyde?: boolean;
  use_ppr?: boolean;
  use_community_selection?: boolean;
  use_refrag?: boolean;  // Enable RefRAG compression
  ppr_damping?: number;
  community_level?: number;
}

export interface ChunkInfo {
  chunk_id: string;
  document_id: string;
  text: string;
  score: number;
  retrieval_mode: string;
  metadata?: any;
  via_entity?: string;
  via_relationship?: string;
  hop_distance?: number;
  source_type?: 'vector' | 'graph_1hop' | 'graph_2hop';  // Source breakdown
}

export interface RetrievalStats {
  total_chunks: number;
  vector_chunks: number;
  graph_1hop_chunks: number;
  graph_2hop_chunks: number;
  graph_total: number;
  entities_used: string[];
}

export interface CompressionStats {
  enabled: boolean;
  original_length: number;
  compressed_length: number;
  compression_ratio: number;
  chunks_compressed: number;
  strategy: string;
  compression_time_ms: number;
}

export interface AskResponse {
  query: string;
  answer: string;
  chunks: ChunkInfo[];
  sources?: Array<{
    chunk_id: string;
    document_id: string;
    text_preview: string;
    score: number;
  }>;
  retrieval_mode: string;
  chunks_retrieved: number;
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  entities_found?: string[];
  entity_count?: number;
  enhanced_query?: string;
  trace_id?: string;
  warning?: string;
  // New top-level stats from enhanced API
  retrieval_stats?: RetrievalStats;
  compression_stats?: CompressionStats;
  metadata: {
    query_entities?: string[];
    graph_traversal?: any;
    fusion_method?: string;
    modes_used?: string[];
    hyde_used?: boolean;
    ppr_used?: boolean;
    community_selection_used?: boolean;
    compression_stats?: {
      compression_ratio: number;
      speedup_factor: number;
      chunks_compressed: number;
      original_tokens?: number;
      compressed_tokens?: number;
    };
    model_used?: string;
  };
}

// ========== CHAT APIs ==========
export const chatApi = {
  // Main chat endpoint with full mode support
  ask: (request: AskRequest) => {
    return fetchApi<AskResponse>('/chat/ask', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  // Retrieval-only (no LLM generation)
  retrieve: (request: AskRequest) => {
    return fetchApi<{
      query: string;
      chunks: ChunkInfo[];
      retrieval_mode: string;
      retrieval_time_ms: number;
      entities_found?: string[];
    }>('/chat/retrieve', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  // V5 endpoint with advanced features
  askV5: (request: AskRequest) => {
    return fetchApi<AskResponse>('/chat/v5/ask', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  // Get available retrieval modes
  getModes: () => {
    return fetchApi<{
      modes: Array<{
        name: string;
        description: string;
        recommended_for: string;
      }>;
    }>('/chat/retrieval/modes');
  },

  // Workflow statistics
  workflowStats: () => {
    return fetchApi<{
      workflow_status: string;
      v2_components: {
        retrieval_engine: {
          default_mode: string;
          auto_route: boolean;
          available_modes: string[];
        };
        prompt_manager: {
          template_count: number;
        };
        response_generator: {
          mode: string;
          temperature: number;
        };
      };
    }>('/chat/workflow/stats');
  },

  // V5 Engine metrics
  v5Metrics: () => {
    return fetchApi<{
      metrics: {
        total_queries: number;
        avg_latency_ms: number;
        p50_latency_ms: number;
        p95_latency_ms: number;
        p99_latency_ms: number;
        qps: number;
        error_rate: number;
        cache_hit_rate: number;
      };
      mode_distribution: Record<string, number>;
      recent_traces: any[];
      slow_traces: any[];
    }>('/chat/v5/metrics');
  },
};

// ========== GraphRAG APIs ==========
export const graphragApi = {
  // Hybrid search (auto-routed)
  search: (query: string, mode?: string) => {
    return fetchApi<{
      query: string;
      answer: string;
      search_mode: string;
      confidence: number;
      metadata: any;
    }>('/graphrag/search', {
      method: 'POST',
      body: JSON.stringify({ query, mode }),
    });
  },

  // Global search (community summaries)
  globalSearch: (query: string) => {
    return fetchApi<{
      query: string;
      answer: string;
      search_mode: string;
      confidence: number;
      metadata: any;
    }>('/graphrag/global', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  },

  // Get statistics
  getStats: () => {
    return fetchApi<{
      global_stats: any;
      local_stats: any;
    }>('/graphrag/stats');
  },

  // Get communities hierarchy
  getCommunities: (level?: number) => {
    const query = level !== undefined ? `?level=${level}` : '';
    return fetchApi<{
      communities: Array<{
        id: string;
        level: number;
        summary: string;
        themes: string[];
        member_count: number;
        sample_members: string[];
        parent_id?: string;
      }>;
      total: number;
      levels: number;
    }>(`/graphrag/communities${query}`);
  },

  // Health check
  health: () => {
    return fetchApi<{
      status: string;
      neo4j_connected: boolean;
      entity_count: number;
      community_count: number;
      tgi_endpoint: string;
    }>('/graphrag/health');
  },
};

// ========== RefRAG APIs ==========
export const refragApi = {
  // Compress text
  compress: (text: string) => {
    return fetchApi<{
      original_text: string;
      compressed_text: string;
      compression_ratio: number;
      speedup_factor: number;
      original_tokens: number;
      compressed_tokens: number;
    }>('/refrag/compress', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  },

  // Get compression metrics
  getMetrics: () => {
    return fetchApi<{
      total_compressions: number;
      average_compression_ratio: number;
      average_speedup: number;
      cache_size: number;
    }>('/refrag/metrics');
  },

  // Get policy stats
  getPolicyStats: () => {
    return fetchApi<{
      total_policies: number;
      policies_by_type: Record<string, number>;
      compression_stats: {
        average_ratio: number;
        total_compressed: number;
      };
    }>('/refrag/policy/stats');
  },

  // Get config
  getConfig: () => {
    return fetchApi<{
      compression_rate: number;
      cache_size: number;
      enabled: boolean;
    }>('/refrag/config');
  },
};

// ========== BENCHMARK APIs ==========
export const benchmarkApi = {
  // Test a query across all modes
  testAllModes: (query: string, topK: number = 5) => {
    return fetchApi<{
      query: string;
      results: Array<{
        mode: string;
        answer: string;
        chunks_count: number;
        retrieval_time_ms: number;
        generation_time_ms: number;
        total_time_ms: number;
        error?: string;
      }>;
      total_time_ms: number;
    }>('/benchmark/modes', {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK }),
    });
  },

  // Compare specific modes
  compareModes: (query: string, modes: RetrievalMode[], topK: number = 5) => {
    return fetchApi<{
      query: string;
      comparison: Array<{
        mode: string;
        answer: string;
        chunks: any[];
        metrics: {
          retrieval_time_ms: number;
          generation_time_ms: number;
          total_time_ms: number;
          chunk_count: number;
        };
      }>;
    }>('/benchmark/compare-modes', {
      method: 'POST',
      body: JSON.stringify({ query, modes, top_k: topK }),
    });
  },
};

// ========== RAGAS EVALUATION APIs ==========
export interface RagasScores {
  chunk_count: number;
  entity_match: number;
  chunk_content: number;
  answer_keywords: number;
  ground_truth_similarity: number;
  weighted_score: number;
  passed: boolean;
}

export interface RagasModeResult {
  mode: string;
  answer: string;
  scores: RagasScores;
  timing_ms: number;
  chunks_used: number;
  entities_found: string[];
  refrag_scores?: RagasScores;
  refrag_answer?: string;
  refrag_timing_ms?: number;
  ground_truth_delta?: number;
  speed_improvement?: number;
  token_savings?: number;
}

export interface RagasTestResult {
  test_id: string;
  query: string;
  expected_answer: string;
  difficulty: string;
  mode_results: RagasModeResult[];
  best_mode: string;
  best_score: number;
}

export interface RefragImpactAnalysis {
  modes_improved: number;
  modes_hurt: number;
  modes_unchanged: number;
  avg_ground_truth_delta: number;
  avg_speed_improvement: number;
  avg_token_savings: number;
}

export interface RagasEvaluationSummary {
  total_tests: number;
  modes_tested: string[];
  mode_scores: Record<string, { avg_score: number; passed: number; failed: number }>;
  best_overall_mode: string;
  best_overall_score: number;
}

export interface RagasEvaluationResponse {
  test_results: RagasTestResult[];
  summary: RagasEvaluationSummary;
  refrag_impact?: RefragImpactAnalysis;
  timestamp: string;
  duration_ms: number;
}

export interface RagasTestCase {
  id: string;
  query: string;
  query_type: string;
  description: string;
  expected_entities: string[];
  expected_answer_preview: string;
  best_modes: string[];
  difficulty: string;
}

export const ragasApi = {
  // Get available test cases
  getTestCases: () => {
    return fetchApi<{
      test_cases: RagasTestCase[];
      total: number;
    }>('/benchmark/ragas/test-cases');
  },

  // Run RAGAS evaluation
  runEvaluation: (request: {
    test_case_ids?: string[];
    modes?: string[];
    compare_refrag?: boolean;
    top_k?: number;
  }) => {
    return fetchApi<RagasEvaluationResponse>('/benchmark/ragas', {
      method: 'POST',
      body: JSON.stringify({
        test_case_ids: request.test_case_ids || [],
        modes: request.modes || ['naive', 'local', 'global', 'hybrid', 'mix'],
        compare_refrag: request.compare_refrag ?? true,
        top_k: request.top_k || 5,
      }),
    });
  },
};

// ========== SYSTEM APIs ==========
export const systemApi = {
  health: () => {
    return fetchApi<{
      status: string;
      services: Record<string, string>;
    }>('/health');
  },

  info: () => {
    return fetchApi<{
      name: string;
      version: string;
      description: string;
      available_services: string[];
    }>('/');
  },
};

export { ApiError };
