import { useCallback, useRef, useState } from 'react';
import type { ChatMessage, SearchChunk, ThinkingStep } from '../types';
import { createChatStream } from '../api/chat';

export type SSEState = {
  /** Whether a stream is currently in progress. */
  streaming: boolean;
  /** Last error message from the chat pipeline. */
  error: string;
  /** Agent thinking steps streamed from the backend. */
  thinkingSteps: ThinkingStep[];
};

export type UseChatStreamOptions = {
  /** Called once when the stream starts.  Returns the assistant message id. */
  onStart: (userMessage: ChatMessage, assistantMessage: ChatMessage) => void;
  /** Called for every token event. */
  onToken: (assistantId: string, token: string) => void;
  /** Called when sources event is received. */
  onSources: (sourcesText: string, sourceChunks: SearchChunk[]) => void;
  /** Called when an error occurs (before the error is recorded in state). */
  onError?: (assistantId: string, message: string) => void;
  /** Called when the stream finishes (success or failure). */
  onDone?: () => void;
  /** Called when an agent step arrives. */
  onAgentStep?: (step: ThinkingStep) => void;
};

/**
 * Hook that manages the full chat flow: search → SSE stream → state updates.
 *
 * Returns the current state plus a `submit` function that starts a new turn.
 *
 * The hook does **not** own the messages array — it uses callbacks so the
 * calling component stays in control of its own message storage.
 *
 * Agent steps are emitted separately from answer tokens so the caller can
 * render the retrieval loop while the answer is still streaming.
 */
export function useChatStream(options: UseChatStreamOptions) {
  const { onStart, onToken, onSources, onError, onDone, onAgentStep } = options;

  const [state, setState] = useState<SSEState>({
    streaming: false,
    error: '',
    thinkingSteps: [],
  });

  const abortRef = useRef<{ abort: () => void } | null>(null);
  const hasAgentStepsRef = useRef(false);

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

      setState((prev) => ({ ...prev, streaming: true, error: '', thinkingSteps: [] }));
      hasAgentStepsRef.current = false;
      onStart(userMessage, assistantMessage);

      const { abort, stream } = createChatStream({ question: trimmed });

      abortRef.current = { abort };

      try {
        for await (const event of stream) {
          switch (event.type) {
            case 'agent-step': {
              const step = {
                step: event.step,
                message: event.message,
                queries: event.queries,
                count: event.count,
                round: event.round,
                query_count: event.query_count,
                total_recalled: event.total_recalled,
                deduplicated: event.deduplicated,
                verdict: event.verdict,
                reasoning: event.reasoning,
                gap: event.gap,
                timestamp: event.timestamp,
              } satisfies ThinkingStep;
              hasAgentStepsRef.current = true;
              setState((s) => ({
                ...s,
                thinkingSteps: [...s.thinkingSteps, step],
              }));
              onAgentStep?.(step);
              break;
            }
            case 'answer-chunk':
              onToken(assistantId, event.content);
              break;
            case 'answer-done':
              break;
            case 'token':
              onToken(assistantId, event.content);
              break;
            case 'sources':
              onSources(
                typeof event.sources === 'string'
                  ? event.sources
                  : event.sources
                    ? JSON.stringify(event.sources, null, 2)
                    : event.content ?? '',
                Array.isArray(event.source_chunks) ? event.source_chunks : [],
              );
              break;
            case 'error':
              throw new Error(event.content || '生成回答失败');
            case 'done': {
              if (hasAgentStepsRef.current) {
                const step = {
                  step: 'done',
                  message: '回答完成',
                  total_rounds: event.total_rounds,
                  chunks_used: event.chunks_used,
                  timestamp: event.timestamp,
                } satisfies ThinkingStep;
                setState((s) => ({
                  ...s,
                  thinkingSteps: [...s.thinkingSteps, step],
                }));
                onAgentStep?.(step);
              }
              break;
            }
          }
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          onError?.(assistantId, '已停止生成');
        } else {
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
    [onStart, onToken, onSources, onError, onDone, onAgentStep],
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
