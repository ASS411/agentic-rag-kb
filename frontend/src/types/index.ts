export type ApiEnvelope<T> = {
  success: boolean;
  code: number;
  message: string;
  data: T;
};

export type DocType = 'pdf' | 'md' | 'txt';

export type DocumentStatus = 'processing' | 'ready' | 'error' | (string & {});

export type DocumentItem = {
  doc_id: string;
  file_name: string;
  doc_type: DocType;
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  status?: DocumentStatus;
  error_message?: string | null;
  uploaded_at: string;
};

export type DocumentList = {
  items: DocumentItem[];
  total: number;
  page: number;
  size: number;
};

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

export type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

export type AgentStepName =
  | 'rewrite'
  | 'search'
  | 'rerank'
  | 'check'
  | 'replan'
  | 'generate'
  | 'done'
  | (string & {});

export type ThinkingStep = {
  step: AgentStepName;
  message?: string;
  queries?: string[];
  count?: number;
  round?: number;
  query_count?: number;
  total_recalled?: number;
  deduplicated?: number;
  verdict?: string;
  reasoning?: string;
  gap?: string;
  total_rounds?: number;
  chunks_used?: number;
  timestamp?: string;
};

export type SSEAgentStepEvent = {
  type: 'agent-step';
  step: AgentStepName;
  message?: string;
  queries?: string[];
  count?: number;
  round?: number;
  query_count?: number;
  total_recalled?: number;
  deduplicated?: number;
  verdict?: string;
  reasoning?: string;
  gap?: string;
  timestamp?: string;
};

export type SSEAnswerChunkEvent = {
  type: 'answer-chunk';
  content: string;
  timestamp?: string;
};

export type SSEAnswerDoneEvent = {
  type: 'answer-done';
  content?: string;
  timestamp?: string;
};

export type SSETokenEvent = {
  type: 'token';
  content: string;
  timestamp?: string;
};

export type SSESourcesEvent = {
  type: 'sources';
  content?: string;
  sources?: unknown;
  source_chunks?: SearchChunk[];
  timestamp?: string;
};

export type SSEDoneEvent = {
  type: 'done';
  content?: string;
  conversation_id?: string;
  total_rounds?: number;
  chunks_used?: number;
  timestamp?: string;
};

export type SSEErrorEvent = {
  type: 'error';
  content: string;
  timestamp?: string;
};

export type StreamEvent =
  | SSEAgentStepEvent
  | SSEAnswerChunkEvent
  | SSEAnswerDoneEvent
  | SSETokenEvent
  | SSESourcesEvent
  | SSEDoneEvent
  | SSEErrorEvent;

export type ChatRequest = {
  question: string;
  stream?: boolean;
  top_k?: number;
  max_rounds?: number;
  use_agent?: boolean;
  conversation_id?: string | null;
};

export type ChatResponse = {
  answer: string;
  sources: string;
  source_chunks: SearchChunk[];
  question: string;
  conversation_id: string | null;
};

export type UploadPhase = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

export type UploadState = {
  phase: UploadPhase;
  progress: number;
  message: string;
};
