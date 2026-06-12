import { PanelRight, Send } from 'lucide-react';
import { type FormEvent, type KeyboardEvent, useCallback, useState } from 'react';
import { Button } from '@/components/ui/button';

export type QuestionInputProps = {
  streaming: boolean;
  hasDocuments: boolean;
  hasSources: boolean;
  onSubmit: (question: string) => void;
  onStop: () => void;
  onToggleSourcePanel: () => void;
};

export function QuestionInput({
  streaming,
  hasDocuments,
  hasSources,
  onSubmit,
  onStop,
  onToggleSourcePanel,
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
      <div className="flex items-center gap-2">
        {hasSources ? (
          <Button variant="ghost" size="icon" type="button" onClick={onToggleSourcePanel} aria-label="切换来源面板">
            <PanelRight size={17} aria-hidden="true" />
          </Button>
        ) : null}
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
          className="flex-1"
        />
        {streaming ? (
          <Button variant="destructive" type="button" onClick={onStop}>
            停止
          </Button>
        ) : (
          <Button variant="primary" type="submit" disabled={!value.trim()}>
            <Send size={17} aria-hidden="true" />
            发送
          </Button>
        )}
      </div>
    </form>
  );
}
