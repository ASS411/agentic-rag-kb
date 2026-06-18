import type { ApiEnvelope } from '../types';
import { readJson } from '../lib/utils';

const API_BASE = '/api/v1';

export type ConversationSummary = {
  conversation_id: string;
  title: string | null;
  record_count: number;
  last_question: string;
  created_at: string | null;
};

export type ConversationList = {
  items: ConversationSummary[];
  total: number;
  page: number;
  size: number;
};

export type QARecord = {
  record_id: string;
  conversation_id: string;
  question: string;
  answer: string | null;
  status: string | null;
  sources: Array<Record<string, unknown>>;
  agent_steps: Array<Record<string, unknown>>;
  total_rounds: number | null;
  model: string | null;
  tokens_used: number | null;
  created_at: string | null;
};

export type QARecordList = {
  items: QARecord[];
  total: number;
  page: number;
  size: number;
};

export async function fetchConversations(
  page = 1,
  size = 20,
): Promise<ConversationList> {
  const payload = await readJson<ApiEnvelope<ConversationList>>(
    await fetch(`${API_BASE}/qa/conversations?page=${page}&size=${size}`),
  );
  return payload.data;
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/qa/conversations/${conversationId}`, { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as any).message || `删除失败 (${res.status})`);
  }
}

export async function fetchHistory(
  conversationId: string,
  page = 1,
  size = 20,
): Promise<QARecordList> {
  const payload = await readJson<ApiEnvelope<QARecordList>>(
    await fetch(
      `${API_BASE}/qa/history?conversation_id=${conversationId}&page=${page}&size=${size}`,
    ),
  );
  return payload.data;
}

export async function fetchSuggestions(): Promise<string[]> {
  const payload = await readJson<ApiEnvelope<string[]>>(
    await fetch(`${API_BASE}/qa/suggestions`),
  );
  return payload.data;
}
