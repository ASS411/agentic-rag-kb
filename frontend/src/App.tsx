import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';

import type { ChatMessage, DocumentItem, SearchChunk, UploadState } from './types';
import { fetchDocuments, fetchSearchChunks, uploadFile } from './api/documents';

import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { DropZone } from './components/upload/DropZone';
import { DocListSection } from './components/upload/DocCard';
import { WelcomePanel } from './components/qa/WelcomePanel';
import { AnswerPanel } from './components/qa/AnswerPanel';
import { QuestionInput } from './components/qa/QuestionInput';
import { SourcePanel } from './components/qa/SourcePanel';

import { useChatStream } from './hooks/useSSE';

const EXAMPLES = [
  '总结这批文档里的关键结论，并列出依据。',
  '这份资料中有哪些风险点需要优先处理？',
  '把相关段落整理成一个三点行动清单。',
];

export default function App() {
  // ── Document state ──────────────────────────────────────────────────
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [docLoading, setDocLoading] = useState(true);
  const [docError, setDocError] = useState('');

  // ── Upload state ────────────────────────────────────────────────────
  const [upload, setUpload] = useState<UploadState>({
    phase: 'idle',
    progress: 0,
    message: '支持 PDF、Markdown、TXT',
  });

  // ── Chat state ──────────────────────────────────────────────────────
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<SearchChunk[]>([]);
  const [sourcesNote, setSourcesNote] = useState('');

  // ── UI state ────────────────────────────────────────────────────────
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [rightOpen, setRightOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // ── Derived ─────────────────────────────────────────────────────────
  const readyDocs = useMemo(
    () => documents.filter((doc) => doc.chunk_count > 0).length,
    [documents],
  );

  // ── Document list ───────────────────────────────────────────────────
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

  // ── Upload handler ──────────────────────────────────────────────────
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

  // ── Chat stream hook ────────────────────────────────────────────────
  const chat = useChatStream({
    onStart: (userMsg, assistantMsg) => {
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setSources([]);
      setSourcesNote('');
      setSelectedSourceId(null);
    },
    onToken: (assistantId, token) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId ? { ...m, content: m.content + token } : m,
        ),
      );
    },
    onSources: (text) => {
      setSourcesNote(text);
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

  // Fire a side-channel search to populate the source panel eagerly
  const handleQuestion = useCallback(
    (question: string) => {
      // Kick off search in parallel
      fetchSearchChunks(question)
        .then(setSources)
        .catch(() => setSources([]));
      // Start the SSE stream
      void chat.submit(question);
    },
    [chat.submit],
  );

  // ── Scroll to bottom on new messages ────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: chat.streaming ? 'smooth' : 'auto',
    });
  }, [messages, chat.streaming]);

  // ── Render ──────────────────────────────────────────────────────────
  return (
    <main className="min-h-screen text-slate-100">
      <div className={clsx('workspace-shell', !rightOpen && 'sources-collapsed')}>
        {/* ── Left sidebar ─────────────────────────────────────────── */}
        <Sidebar>
          <DropZone upload={upload} onFiles={handleFiles} />
          <DocListSection
            documents={documents}
            loading={docLoading}
            error={docError}
            onRefresh={refreshDocuments}
          />
        </Sidebar>

        {/* ── Center conversation ──────────────────────────────────── */}
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
              <AnswerPanel
                messages={messages}
                error={chat.error}
              />
            )}
            <div ref={bottomRef} />
          </div>

          <QuestionInput
            streaming={chat.streaming}
            hasDocuments={documents.length > 0}
            onSubmit={handleQuestion}
            onStop={chat.stop}
          />
        </section>

        {/* ── Right source panel ───────────────────────────────────── */}
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
