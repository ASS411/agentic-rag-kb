import { AlertTriangle, Loader2, ChevronDown, CircuitBoard } from 'lucide-react';
import clsx from 'clsx';
import type { ChatMessage, SearchChunk, ThinkingStep } from '../../types';
import { SourceHighlight } from './SourceHighlight';

export type AnswerPanelProps = {
  messages: ChatMessage[];
  error: string;
  sources: SearchChunk[];
  selectedSourceId: string | null;
  onSelectSource: (chunkId: string | null) => void;
  thinkingSteps: ThinkingStep[];
  thinkingOpen: boolean;
  onToggleThinking: () => void;
};

function formatStepMsg(step: ThinkingStep): string {
  if (step.message) return step.message;
  const map: Record<string, string> = {
    rewrite: '改写查询',
    search: '检索中',
    rerank: '精排中',
    check: '评估质量',
    replan: '补充检索',
    generate: '生成回答',
    done: '完成',
  };
  return map[step.step] ?? step.step;
}

export function AnswerPanel({
  messages,
  error,
  sources,
  selectedSourceId,
  onSelectSource,
  thinkingSteps,
  thinkingOpen,
  onToggleThinking,
}: AnswerPanelProps) {
  const hasSources = sources.length > 0;
  const hasThinking = thinkingSteps.length > 0;
  const lastAssistantIdx = [...messages].reverse().findIndex((m) => m.role === 'assistant');

  return (
    <>
      {messages.map((message, idx) => {
        const isLastAssistant = message.role === 'assistant' &&
          messages.length - 1 - idx === lastAssistantIdx;

        return (
          <article className={clsx('message', message.role)} key={message.id}>
            <div className="message-label">
              {message.role === 'user' ? '你' : '知识库'}
            </div>
            <div className="message-body">
              {!message.content ? (
                <span className="typing">
                  <Loader2 className="spin" size={16} aria-hidden="true" />
                  正在生成回答
                </span>
              ) : message.role === 'assistant' ? (
                <SourceHighlight
                  content={message.content}
                  sources={hasSources ? sources : []}
                  selectedSourceId={hasSources ? selectedSourceId : null}
                  onSelectSource={hasSources ? onSelectSource : () => {}}
                />
              ) : (
                message.content
              )}

              {/* Inline thinking bubble below last assistant message */}
              {isLastAssistant && hasThinking ? (
                <details className="thinking-inline" open={thinkingOpen} onToggle={onToggleThinking}>
                  <summary>
                    <CircuitBoard size={13} className="inline mr-1" />
                    思考过程 ({thinkingSteps.length} 步)
                    <ChevronDown size={13} className="inline ml-1 opacity-50" />
                  </summary>
                  <div className="thinking-inline-steps">
                    {thinkingSteps.map((step, i) => (
                      <div key={i} className="text-xs py-0.5">
                        <span className="font-medium text-[hsl(var(--primary))]">
                          {step.step}
                        </span>
                        {': '}
                        {formatStepMsg(step)}
                        {step.verdict ? (
                          <span className={clsx(
                            'ml-1 px-1 rounded text-[10px]',
                            step.verdict === 'sufficient' ? 'bg-primary/20 text-primary' : 'text-muted-foreground'
                          )}>
                            {step.verdict === 'sufficient' ? '充足' : '不足'}
                          </span>
                        ) : null}
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}
            </div>
          </article>
        );
      })}
      {error ? (
        <div className="chat-error">
          <AlertTriangle size={16} aria-hidden="true" />
          {error}
        </div>
      ) : null}
    </>
  );
}
