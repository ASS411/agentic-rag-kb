import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';

import type { ChatMessage, DocumentItem, SearchChunk, ThinkingStep, UploadState } from './types';
import { fetchDocuments, uploadFile } from './api/documents';

import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { DropZone } from './components/upload/DropZone';
import { DocListSection } from './components/upload/DocCard';
import { WelcomePanel } from './components/qa/WelcomePanel';
import { AnswerPanel } from './components/qa/AnswerPanel';
import { QuestionInput } from './components/qa/QuestionInput';
import { ThinkingPanel } from './components/qa/ThinkingPanel';
import { SourcePanel } from './components/qa/SourcePanel';

import { useChatStream } from './hooks/useSSE';

const EXAMPLES = [
  '总结这批文档里的关键结论，并列出依据。',
  '这份资料中有哪些风险点需要优先处理？',
  '把相关段落整理成一个三点行动清单。',
];

export default function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [docLoading, setDocLoading] = useState(true);
  const [docError, setDocError] = useState('');

  const [upload, setUpload] = useState<UploadState>({
    phase: 'idle',
    progress: 0,
    message: '支持 PDF、Markdown、TXT',
  });

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<SearchChunk[]>([]);
  const [sourcesNote, setSourcesNote] = useState('');
  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);

  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [rightOpen, setRightOpen] = useState(true);
  const [thinkingOpen, setThinkingOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const readyDocs = useMemo(
    () => documents.filter((doc) => doc.chunk_count > 0).length,
    [documents],
  );

  const refreshDocuments = useCallback(async () => {
    setDocLoading(true);
    setDocError('');
    try {
      const result = await fetchDocuments();
      setDocuments(result.items);
    } catch (error) {
      setDocError(error instanceof Error ? error.message : '文档列表加载失败');
    } finally {
      setDocLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshDocuments();
  }, [refreshDocuments]);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const file = Array.from(files)[0];
      if (!file) return;

      setUpload({ phase: 'uploading', progress: 4, message: `正在上传 ${file.name}` });
      try {
        const uploaded = await uploadFile(file, (progress) => {
          setUpload({ phase: 'uploading', progress, message: `正在上传 ${file.name}` });
        });
        setUpload({
          phase: 'success',
          progress: 100,
          message: `${uploaded.file_name} 已进入解析队列`,
        });
        await refreshDocuments();
      } catch (error) {
        setUpload({
          phase: 'error',
          progress: 0,
          message: error instanceof Error ? error.message : '上传失败',
        });
      }
    },
    [refreshDocuments],
  );

  const chat = useChatStream({
    onStart: (userMsg, assistantMsg) => {
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setSources([]);
      setSourcesNote('');
      setSelectedSourceId(null);
      setThinkingSteps([]);
      setThinkingOpen(true);
    },
    onToken: (assistantId, token) => {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantId ? { ...message, content: message.content + token } : message,
        ),
      );
    },
    onSources: (text, sourceChunks) => {
      setSourcesNote(text);
      setSources(sourceChunks);
      setSelectedSourceId(sourceChunks[0]?.chunk_id ?? null);
    },
    onAgentStep: (step) => {
      setThinkingSteps((prev) => {
        const next = [...prev, step];
        return next.length > 20 ? next.slice(-20) : next;
      });
    },
    onError: (assistantId, message) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && !m.content
            ? { ...m, content: `回答生成失败：${message}` }
            : m,
        ),
      );
    },
  });

  const handleQuestion = useCallback(
    (question: string) => {
      void chat.submit(question);
    },
    [chat.submit],
  );

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: chat.streaming ? 'smooth' : 'auto',
    });
  }, [messages, chat.streaming]);

  return (
    <main className="min-h-screen text-slate-100">
      <div className={clsx('workspace-shell', !rightOpen && 'sources-collapsed')}>
        <Sidebar>
          <DropZone upload={upload} onFiles={handleFiles} />
          <DocListSection
            documents={documents}
            loading={docLoading}
            error={docError}
            onRefresh={refreshDocuments}
          />
        </Sidebar>

        <section className="conversation-pane">
          <Header
            docCount={documents.length}
            readyCount={readyDocs}
            onToggleSourcePanel={() => setRightOpen((o) => !o)}
          />

          <div className="message-scroll">
            {messages.length === 0 ? (
              <WelcomePanel
                examples={EXAMPLES}
                streaming={chat.streaming}
                onSubmitExample={handleQuestion}
              />
            ) : (
              <AnswerPanel messages={messages} error={chat.error} />
            )}

            <ThinkingPanel
              steps={thinkingSteps}
              expanded={thinkingOpen}
              onToggle={() => setThinkingOpen((open) => !open)}
            />
            <div ref={bottomRef} />
          </div>

          <QuestionInput
            streaming={chat.streaming}
            hasDocuments={documents.length > 0}
            onSubmit={handleQuestion}
            onStop={chat.stop}
          />
        </section>

        <SourcePanel
          sources={sources}
          selectedSourceId={selectedSourceId}
          sourcesNote={sourcesNote}
          open={rightOpen}
          onSelectSource={setSelectedSourceId}
        />
      </div>
    </main>
  );
}
