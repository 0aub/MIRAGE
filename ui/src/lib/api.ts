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

    // Add 30-second timeout to prevent browser connection stalling
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.error(`[API] Request timeout after 30s: ${url}`);
      controller.abort();
    }, 30000);

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      signal: controller.signal,
      // Force new connection, don't reuse keep-alive connections that might be stale
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
    console.log(`[API] Parsed JSON response:`, data);
    return data;
  } catch (error) {
    console.error(`[API] Fetch error for ${url}:`, error);
    if (error instanceof ApiError) {
      throw error;
    }
    // Handle abort/timeout
    if (error.name === 'AbortError') {
      throw new ApiError(0, `Request timeout: The server took too long to respond`);
    }
    throw new ApiError(0, `Network error: ${error.message}`);
  }
}

// File Management APIs
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

  get: (fileId: string) => {
    return fetchApi<any>(`/files/files/${fileId}`);
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

  uploadMultiple: async (files: File[]) => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const response = await fetch(`${API_BASE_URL}/files/files/upload-multiple`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(response.status, error.detail || 'Upload failed');
    }

    return response.json();
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

// Documents Registry APIs (files, URLs, YouTube videos)
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

  getContent: (documentId: string) => {
    return fetchApi<{
      document_id: string;
      title: string;
      full_text: string;
    }>(`/documents/documents/${documentId}/content`);
  },
};

// Document Processing APIs
export const documentApi = {
  process: (fileId: string) => {
    return fetchApi<any>(`/document/process/${fileId}`, { method: 'POST' });
  },

  processWithRefrag: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/document/process-with-refrag`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new ApiError(response.status, error.detail || 'Processing failed');
    }

    return response.json();
  },

  status: (documentId: string) => {
    return fetchApi<any>(`/document/status/${documentId}`);
  },
};

// Graph APIs
export const graphApi = {
  visualizeDocument: (documentId: string) => {
    return fetchApi<{
      document_id: string;
      nodes: Array<{
        id: string;
        label: string;
        type: string;
        confidence: number;
      }>;
      edges: Array<{
        source: string;
        target: string;
        relationship: string;
        confidence: number;
      }>;
      node_count: number;
      edge_count: number;
    }>(`/graph/visualize/document/${documentId}`);
  },

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

  documents: () => {
    return fetchApi<{
      total: number;
      document_ids: string[];
    }>('/graph/documents');
  },

  clear: () => {
    return fetchApi<any>('/graph/clear', { method: 'POST' });
  },

  rebuild: () => {
    return fetchApi<any>('/graph/rebuild', { method: 'POST' });
  },

  deleteDocument: (documentId: string) => {
    return fetchApi<any>(`/graph/document/${documentId}`, { method: 'DELETE' });
  },
};

// Database Management APIs
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

  clearAll: () => {
    return fetchApi<any>('/db/clear-all', { method: 'POST' });
  },

  graph: {
    health: () => {
      return fetchApi<any>('/db/graph/health');
    },
    stats: () => {
      return fetchApi<any>('/db/graph/stats');
    },
    clear: () => {
      return fetchApi<any>('/db/graph/clear', { method: 'POST' });
    },
    rebuild: () => {
      return fetchApi<any>('/db/graph/rebuild', { method: 'POST' });
    },
    deleteDocument: (documentId: string) => {
      return fetchApi<any>(`/db/graph/document/${documentId}`, { method: 'DELETE' });
    },
  },

  vector: {
    health: () => {
      return fetchApi<any>('/db/vector/health');
    },
    stats: () => {
      return fetchApi<any>('/db/vector/stats');
    },
    clear: () => {
      return fetchApi<any>('/db/vector/clear', { method: 'POST' });
    },
    deleteDocument: (documentId: string) => {
      return fetchApi<any>(`/db/vector/document/${documentId}`, { method: 'DELETE' });
    },
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
  },

  getSettings: () => {
    return fetchApi<{
      settings: {
        models?: any;
        processing?: any;
        databases?: any;
        chunking_strategies?: any;
      };
      config_dir: string;
    }>('/db/settings');
  },
};

// URL Processing APIs
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

  process: (url: string, useSemanticChunking: boolean = true) => {
    return fetchApi<{
      document_id: string;
      url: string;
      title: string | null;
      content_type: 'webpage' | 'youtube';
      processing_time_seconds: number;
      phase1: {
        total_chars: number;
        total_words: number;
        chunk_count: number;
        chunking_method: string;
      };
      phase2: {
        entities_extracted: number;
        relationships_extracted: number;
        graph_storage: any;
      };
      phase3: {
        original_length: number;
        compressed_length: number;
        compression_ratio: number;
        chunks_compressed: number;
        speedup_factor: number;
        processing_time: number;
        strategy: string;
      };
      phase5: {
        vector_storage: any;
      };
    }>('/url/process', {
      method: 'POST',
      body: JSON.stringify({
        url,
        use_semantic_chunking: useSemanticChunking,
      }),
    });
  },

  getProcessingStatus: () => {
    return fetchApi<{
      processing: Array<{
        document_id: string;
        title: string;
        url: string;
        content_type: 'webpage' | 'youtube';
        status: string;
        start_time: number;
        elapsed_seconds: number;
        current_chunk?: number;
        total_chunks?: number;
      }>;
      count: number;
    }>('/url/processing-status');
  },

  // NEW: Async job-based processing
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

  getJobResult: (jobId: string) => {
    return fetchApi<{
      document_id: string;
      url: string;
      title: string | null;
      content_type: 'webpage' | 'youtube';
      processing_time_seconds: number;
      phase1: any;
      phase2: any;
      phase3: any;
      phase5: any;
    }>(`/url/jobs/${jobId}/result`);
  },

  getActiveJobs: () => {
    return fetchApi<{
      jobs: any[];
      count: number;
    }>('/url/jobs/active');
  },
};

// Chat APIs
export const chatApi = {
  sendMessage: (message: string, conversationId?: string) => {
    return fetchApi<{
      message: string;
      conversation_id: string;
      sources: string[];
      retrieved_nodes: number;
      compression_rate?: number;
      response_time_ms: number;
    }>('/chat/message', {
      method: 'POST',
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        use_graph: true,
        use_refrag: true,
      }),
    });
  },

  queryDetailed: (message: string, conversationId?: string) => {
    return fetchApi<{
      query: string;
      response: string;
      citations: any[];
      workflow_metadata: any;
      graph_visualization: {
        nodes: any[];
        edges: any[];
      };
      compression_comparison: {
        original_chunks: any[];
        compressed_chunks: any[];
        stats: any;
      };
      success: boolean;
      error?: string;
    }>('/chat/query-detailed', {
      method: 'POST',
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
      }),
    });
  },

  getHistory: (conversationId: string) => {
    return fetchApi<any>(`/chat/history/${conversationId}`);
  },

  workflowStats: () => {
    return fetchApi<any>('/chat/workflow/stats');
  },
};

// GraphRAG APIs
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

  // Local search (entity traversal)
  localSearch: (query: string) => {
    return fetchApi<{
      query: string;
      answer: string;
      search_mode: string;
      confidence: number;
      metadata: any;
    }>('/graphrag/local', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  },

  // Get search statistics
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

export { ApiError };
