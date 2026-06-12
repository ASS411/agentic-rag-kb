import { AlertTriangle, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import type { ChatMessage, SearchChunk } from '../../types';
import { SourceHighlight } from './SourceHighlight';

export type AnswerPanelProps = {
  messages: ChatMessage[];
  error: string;
  sources: SearchChunk[];
  selectedSourceId: string | null;
  onSelectSource: (chunkId: string | null) => void;
};

export function AnswerPanel({
  messages,
  error,
  sources,
  selectedSourceId,
  onSelectSource,
}: AnswerPanelProps) {
  const hasSources = sources.length > 0;

  return (
    <>
      {messages.map((message) => (
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
              hasSources ? (
                <SourceHighlight
                  content={message.content}
                  sources={sources}
                  selectedSourceId={selectedSourceId}
                  onSelectSource={onSelectSource}
                />
              ) : (
                <SourceHighlight
                  content={message.content}
                  sources={[]}
                  selectedSourceId={null}
                  onSelectSource={() => {}}
                />
              )
            ) : (
              message.content
            )}
          </div>
        </article>
      ))}
      {error ? (
        <div className="chat-error">
          <AlertTriangle size={16} aria-hidden="true" />
          {error}
        </div>
      ) : null}
    </>
  );
}
