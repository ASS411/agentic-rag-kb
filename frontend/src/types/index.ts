/* Shared TypeScript type definitions for Agentic RAG frontend. */

// ---------------------------------------------------------------------------
// API envelope
// ---------------------------------------------------------------------------

export type ApiEnvelope<T> = {
  success: boolean;
  code: number;
  message: string;
  data: T;
};

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

export type DocType = 'pdf' | 'md' | 'txt';

export type DocumentItem = {
  doc_id: string;
  file_name: string;
  doc_type: DocType;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  uploaded_at: string;
};

export type DocumentList = {
  items: DocumentItem[];
  total: number;
  page: number;
  size: number;
};

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

export type SearchChunk = {
  chunk_id: string;
  content: string;
  score: number;
  doc_id: string;
  doc_name: string;
  doc_type: string;
  page: number;
  chunk_index: number;
  metadata: Record<string, unknown>;
};

export type SearchResponse = {
  query: string;
  total_results: number;
  results: SearchChunk[];
};

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

export type StreamEventType = 'token' | 'sources' | 'done' | 'error';

export type StreamEvent = {
  type: StreamEventType;
  content: string;
};

export type ChatRequest = {
  question: string;
  stream?: boolean;
  top_k?: number;
  conversation_id?: string | null;
};

export type ChatResponse = {
  answer: string;
  sources: string;
  question: string;
  conversation_id: string | null;
};

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------

export type UploadPhase = 'idle' | 'uploading' | 'success' | 'error';

export type UploadState = {
  phase: UploadPhase;
  progress: number;
  message: string;
};
