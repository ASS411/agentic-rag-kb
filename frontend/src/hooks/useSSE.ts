import { useCallback, useRef, useState } from 'react';
import type { ChatMessage } from '../types';
import { createChatStream } from '../api/chat';
import { fetchSearchChunks } from '../api/documents';

export type SSEState = {
  /** Whether a stream is currently in progress. */
  streaming: boolean;
  /** Last error message from the chat pipeline. */
  error: string;
};

export type UseChatStreamOptions = {
  /** Called once when the stream starts.  Returns the assistant message id. */
  onStart: (userMessage: ChatMessage, assistantMessage: ChatMessage) => void;
  /** Called for every token event. */
  onToken: (assistantId: string, token: string) => void;
  /** Called when sources event is received. */
  onSources: (sourcesText: string) => void;
  /** Called when an error occurs (before the error is recorded in state). */
  onError?: (assistantId: string, message: string) => void;
  /** Called when the stream finishes (success or failure). */
  onDone?: () => void;
};

/**
 * Hook that manages the full chat flow: search → SSE stream → state updates.
 *
 * Returns the current state plus a `submit` function that starts a new turn.
 *
 * The hook does **not** own the messages array — it uses callbacks so the
 * calling component stays in control of its own message storage.
 *
 * Also fires a side-channel search request in parallel to populate the
 * source panel eagerly (even before the LLM finishes).
 */
export function useChatStream(options: UseChatStreamOptions) {
  const { onStart, onToken, onSources, onError, onDone } = options;

  const [state, setState] = useState<SSEState>({
    streaming: false,
    error: '',
  });

  const abortRef = useRef<{ abort: () => void } | null>(null);

  const submit = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed) return;

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
      };
      const assistantId = crypto.randomUUID();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
      };

      setState({ streaming: true, error: '' });
      onStart(userMessage, assistantMessage);

      // Fire search in parallel for eager source display
      fetchSearchChunks(trimmed).catch(() => {
        /* best-effort */
      });

      const { abort, stream } = createChatStream({ question: trimmed });

      abortRef.current = { abort };

      try {
        for await (const event of stream) {
          switch (event.type) {
            case 'token':
              onToken(assistantId, event.content);
              break;
            case 'sources':
              onSources(event.content);
              break;
            case 'error':
              throw new Error(event.content || '生成回答失败');
            case 'done':
              // terminal — handled by loop exit
              break;
          }
        }
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          const message = error instanceof Error ? error.message : '生成回答失败';
          setState((s) => ({ ...s, error: message }));
          onError?.(assistantId, message);
        }
      } finally {
        setState((s) => ({ ...s, streaming: false }));
        abortRef.current = null;
        onDone?.();
      }
    },
    [onStart, onToken, onSources, onError, onDone],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setState((s) => ({ ...s, streaming: false }));
    abortRef.current = null;
  }, []);

  return {
    ...state,
    submit,
    stop,
  };
}
