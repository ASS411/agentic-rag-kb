import { Send } from 'lucide-react';
import { type FormEvent, type KeyboardEvent, useCallback, useState } from 'react';

export type QuestionInputProps = {
  /** Whether the chat is currently streaming. */
  streaming: boolean;
  /** Whether there are documents available (controls placeholder). */
  hasDocuments: boolean;
  /** Called when the user submits a question. */
  onSubmit: (question: string) => void;
  /** Called to stop streaming mid-response. */
  onStop: () => void;
};

export function QuestionInput({
  streaming,
  hasDocuments,
  onSubmit,
  onStop,
}: QuestionInputProps) {
  const [value, setValue] = useState('');

  const handleSubmit = useCallback(
    (event?: FormEvent) => {
      event?.preventDefault();
      const trimmed = value.trim();
      if (!trimmed || streaming) return;
      setValue('');
      onSubmit(trimmed);
    },
    [value, streaming, onSubmit],
  );

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  return (
    <form
      className="question-dock"
      onSubmit={handleSubmit}
    >
      <textarea
        value={value}
        placeholder={
          hasDocuments
            ? '向当前知识库提问...'
            : '先上传文档，也可以直接试问接口状态...'
        }
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={streaming}
        rows={1}
      />
      {streaming ? (
        <button className="send-button stop" type="button" onClick={onStop}>
          停止
        </button>
      ) : (
        <button className="send-button" type="submit" disabled={!value.trim()}>
          <Send size={17} aria-hidden="true" />
          发送
        </button>
      )}
    </form>
  );
}
