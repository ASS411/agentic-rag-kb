import type { ChatRequest, StreamEvent } from '../types';
import { parseSseBlock } from '../lib/utils';

const API_BASE = '/api/v1';

// ---------------------------------------------------------------------------
// SSE streaming chat
// ---------------------------------------------------------------------------

/**
 * Initiate a streaming RAG chat via POST + ReadableStream SSE.
 *
 * Returns an abort function and an async iterable over events.
 * This avoids a tight coupling between fetch/SSE plumbing and the React
 * component that displays the result.
 */
export function createChatStream(
  body: ChatRequest,
): {
  abort: () => void;
  stream: AsyncIterable<StreamEvent>;
} {
  const controller = new AbortController();

  async function* streamEvents(): AsyncIterable<StreamEvent> {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, stream: true }),
      signal: controller.signal,
    });

    if (!response.ok || !response.body) {
      throw new Error(response.statusText || '流式回答连接失败');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() ?? '';

        for (const block of blocks) {
          const event = parseSseBlock(block);
          if (event) yield event;

          if (event?.type === 'error') {
            throw new Error(event.content || '生成回答失败');
          }
        }
      }

      // Drain remaining buffer
      if (buffer.trim()) {
        const event = parseSseBlock(buffer);
        if (event && event.type === 'error') {
          throw new Error(event.content || '生成回答失败');
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  return {
    abort: () => controller.abort(),
    stream: streamEvents(),
  };
}

// ---------------------------------------------------------------------------
// Non-streaming chat (fallback)
// ---------------------------------------------------------------------------

// Re-use the fetchSearchChunks from documents API; the non-streaming code
// path in App.tsx can import from there.  This module focuses on SSE.
