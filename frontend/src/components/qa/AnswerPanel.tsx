import { AlertTriangle, ChevronDown, ChevronRight, CircuitBoard, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ChatMessage, SearchChunk, ThinkingStep } from '../../types';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import { SourceHighlight } from './SourceHighlight';

export type AnswerPanelProps = {
  messages: ChatMessage[];
  error: string;
  sources: SearchChunk[];
  selectedSourceId: string | null;
  onSelectSource: (chunkId: string | null) => void;
  thinkingSteps: ThinkingStep[];
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
}: AnswerPanelProps) {
  const hasSources = sources.length > 0;
  const hasThinking = thinkingSteps.length > 0;
  const [thinkingOpen, setThinkingOpen] = useState(true);
  const lastAssistantIdx = [...messages].reverse().findIndex((m) => m.role === 'assistant');

  // Group messages into Q&A turns (user + assistant pairs)
  const turns = useMemo(() => {
    const result: { user: ChatMessage; assistant: ChatMessage | null }[] = [];
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].role === 'user') {
        const assistant = i + 1 < messages.length && messages[i + 1].role === 'assistant'
          ? messages[i + 1]
          : null;
        result.push({ user: messages[i], assistant });
        if (assistant) i++;
      }
    }
    return result;
  }, [messages]);

  const [collapsedTurns, setCollapsedTurns] = useState<Set<number>>(() => {
    // Auto-collapse older turns when there are 3+ turns
    if (turns.length <= 2) return new Set<number>();
    const s = new Set<number>();
    for (let i = 0; i < turns.length - 2; i++) s.add(i);
    return s;
  });

  // Update collapsed state when turns change (new turn arrives)
  const prevTurnsLen = useRef(turns.length);
  useEffect(() => {
    if (turns.length > prevTurnsLen.current && turns.length > 2) {
      // New turn added — ensure older turns stay collapsed
      setCollapsedTurns((prev) => {
        const next = new Set(prev);
        for (let i = 0; i < turns.length - 2; i++) next.add(i);
        return next;
      });
    }
    prevTurnsLen.current = turns.length;
  }, [turns.length]);

  const toggleTurn = (idx: number) => {
    setCollapsedTurns((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const globalMsgIdx = (turnIdx: number, pos: 'user' | 'assistant') => {
    let idx = 0;
    for (let i = 0; i < turnIdx; i++) {
      idx += turns[i].assistant ? 2 : 1;
    }
    return pos === 'user' ? idx : idx + 1;
  };

  return (
    <>
      {turns.map((turn, tIdx) => {
        const collapsed = collapsedTurns.has(tIdx);
        const assistantMsgIdx = turn.assistant ? globalMsgIdx(tIdx, 'assistant') : -1;
        const isLastAssistant = turn.assistant &&
          messages.length - 1 - assistantMsgIdx === lastAssistantIdx;

        // Show collapsible wrapper for older turns (not the last 2)
        const showCollapse = turns.length > 2 && tIdx < turns.length - 2;

        const turnContent = (
          <>
            {/* User message */}
            <article className="message user" key={turn.user.id}>
              <div className="message-label">你</div>
              <div className="message-body">{turn.user.content}</div>
            </article>

            {/* Assistant message */}
            {turn.assistant ? (
              <article className="message assistant" key={turn.assistant.id}>
                <div className="message-label">知识库</div>
                <div className="message-body">
                  {isLastAssistant && hasThinking ? (
                    <details
                      className="thinking-inline"
                      open={thinkingOpen}
                      onToggle={(e) => setThinkingOpen((e.target as HTMLDetailsElement).open)}
                    >
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
                  {!turn.assistant.content ? (
                    <span className="typing">
                      <Loader2 className="spin" size={16} aria-hidden="true" />
                      正在生成回答
                    </span>
                  ) : (
                    <SourceHighlight
                      content={turn.assistant.content}
                      sources={hasSources ? sources : []}
                      selectedSourceId={hasSources ? selectedSourceId : null}
                      onSelectSource={hasSources ? onSelectSource : () => {}}
                    />
                  )}
                </div>
              </article>
            ) : null}
          </>
        );

        if (!showCollapse) return <div key={turn.user.id}>{turnContent}</div>;

        return (
          <Collapsible
            key={turn.user.id}
            open={!collapsed}
            onOpenChange={() => toggleTurn(tIdx)}
            className="turn-collapsible"
          >
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="turn-collapse-trigger"
                aria-label={collapsed ? '展开对话轮次' : '折叠对话轮次'}
              >
                {collapsed ? (
                  <ChevronRight size={14} className="inline" />
                ) : (
                  <ChevronDown size={14} className="inline" />
                )}
                <span className="turn-collapse-summary">
                  {turn.user.content.slice(0, 60)}
                  {turn.user.content.length > 60 ? '…' : ''}
                </span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="turn-collapsible-content">
              {turnContent}
            </CollapsibleContent>
          </Collapsible>
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
