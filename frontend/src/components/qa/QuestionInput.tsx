import { Send } from 'lucide-react';
import { type FormEvent, type KeyboardEvent, useCallback, useState } from 'react';

export type QuestionInputProps = {
  streaming: boolean;
  hasDocuments: boolean;
  onSubmit: (question: string) => void;
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
    <form className="question-dock" onSubmit={handleSubmit}>
      <textarea
        value={value}
        placeholder={
          hasDocuments
            ? '向当前知识库提问...'
            : '请先上传文档，或输入问题测试聊天管道...'
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
