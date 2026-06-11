import { AlertTriangle, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '../../types';

export type AnswerPanelProps = {
  messages: ChatMessage[];
  error: string;
};

export function AnswerPanel({ messages, error }: AnswerPanelProps) {
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
              <ReactMarkdown
                className="message-markdown"
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
              >
                {message.content}
              </ReactMarkdown>
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
