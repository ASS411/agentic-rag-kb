import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import { useQuery } from '@tanstack/react-query';

import type { DocumentItem, UploadState } from './types';
import { fetchDocuments, uploadFile } from './api/documents';
import { useQAStore } from './stores/qaStore';
import { useSourceScroll } from './hooks/useSourceScroll';

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

const PROCESSING_POLL_INTERVAL_MS = 1500;
const PROCESSING_POLL_MAX_ATTEMPTS = 24;

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isDocumentReady(doc: DocumentItem) {
  return doc.status === 'ready' || doc.chunk_count > 0;
}

export default function App() {
  // ── TanStack Query — documents ────────────────────────────────
  const {
    data: documentData,
    isLoading: docLoading,
    error: docError,
    refetch: refreshDocuments,
  } = useQuery({
    queryKey: ['documents'],
    queryFn: fetchDocuments,
    staleTime: 10_000,
  });

  const documents: DocumentItem[] = documentData?.items ?? [];
  const docErrorMsg =
    docError instanceof Error ? docError.message : '';

  // ── Zustand — QA state ───────────────────────────────────────
  const messages = useQAStore((s) => s.messages);
  const sources = useQAStore((s) => s.sources);
  const sourcesNote = useQAStore((s) => s.sourcesNote);
  const thinkingSteps = useQAStore((s) => s.thinkingSteps);
  const selectedSourceId = useQAStore((s) => s.selectedSourceId);
  const thinkingOpen = useQAStore((s) => s.thinkingOpen);

  // ── useSourceScroll ──────────────────────────────────────────
  const { scrollToCitation } = useSourceScroll();

  const handleSelectSource = useCallback(
    (chunkId: string | null) => {
      useQAStore.getState().selectSource(chunkId);
      scrollToCitation(chunkId);
    },
    [scrollToCitation],
  );

  // ── Upload state (local — transient UI) ─────────────────────
  const [upload, setUpload] = useState<UploadState>({
    phase: 'idle',
    progress: 0,
    message: '支持 PDF、Markdown、TXT',
  });

  const [rightOpen, setRightOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const readyDocs = useMemo(
    () => documents.filter(isDocumentReady).length,
    [documents],
  );

  // ── Chat stream with qaStore callbacks ───────────────────────
  const chat = useChatStream({
    onStart: (userMsg, assistantMsg) => {
      useQAStore.getState().addMessage(userMsg);
      useQAStore.getState().addMessage(assistantMsg);
      useQAStore.getState().setSources([], '');
      useQAStore.getState().clearThinkingSteps();
      useQAStore.getState().selectSource(null);
      useQAStore.getState().toggleThinking();
      if (!useQAStore.getState().thinkingOpen) {
        useQAStore.getState().toggleThinking();
      }
    },
    onToken: (assistantId, token) => {
      useQAStore.getState().updateMessage(assistantId, token);
    },
    onSources: (text, sourceChunks) => {
      useQAStore.getState().setSources(sourceChunks, text);
    },
    onAgentStep: (step) => {
      useQAStore.getState().addThinkingStep(step);
    },
    onError: (assistantId, message) => {
      const msgs = useQAStore.getState().messages;
      const updated = msgs.map((m) =>
        m.id === assistantId && !m.content
          ? { ...m, content: `回答生成失败：${message}` }
          : m,
      );
      useQAStore.getState().setMessages(updated);
    },
  });

  // ── Upload handler ──────────────────────────────────────────
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
          phase: 'processing',
          progress: 92,
          message: `${uploaded.file_name} 正在解析并写入索引`,
        });
        await refreshDocuments();

        // Poll until the document is ready
        let processed: DocumentItem | null = null;
        for (let attempt = 0; attempt < PROCESSING_POLL_MAX_ATTEMPTS; attempt += 1) {
          await wait(PROCESSING_POLL_INTERVAL_MS);
          const result = await fetchDocuments();
          const found = result.items.find((doc) => doc.doc_id === uploaded.doc_id);
          if (!found) continue;
          if (found.status === 'error') { processed = found; break; }
          if (isDocumentReady(found)) { processed = found; break; }
        }

        if (processed?.status === 'error') {
          setUpload({
            phase: 'error', progress: 0,
            message: processed.error_message || `${processed.file_name} 解析失败`,
          });
        } else if (processed) {
          setUpload({
            phase: 'success', progress: 100,
            message: `${processed.file_name} 已就绪，生成 ${processed.chunk_count} 个片段`,
          });
        } else {
          setUpload({
            phase: 'processing', progress: 96,
            message: `${uploaded.file_name} 仍在后台处理中，可稍后刷新文档列表`,
          });
        }
      } catch (error) {
        setUpload({
          phase: 'error', progress: 0,
          message: error instanceof Error ? error.message : '上传失败',
        });
      }
    },
    [refreshDocuments],
  );

  const handleQuestion = useCallback(
    (question: string) => { void chat.submit(question); },
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
            error={docErrorMsg}
            onRefresh={() => { void refreshDocuments(); }}
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
              <AnswerPanel
                messages={messages}
                error={chat.error}
                sources={sources}
                selectedSourceId={selectedSourceId}
                onSelectSource={handleSelectSource}
              />
            )}

            <ThinkingPanel
              steps={thinkingSteps}
              expanded={thinkingOpen}
              onToggle={() => useQAStore.getState().toggleThinking()}
            />
            <div ref={bottomRef} />
          </div>

          <QuestionInput
            streaming={chat.streaming}
            hasDocuments={readyDocs > 0}
            onSubmit={handleQuestion}
            onStop={chat.stop}
          />
        </section>

        <SourcePanel
          sources={sources}
          selectedSourceId={selectedSourceId}
          sourcesNote={sourcesNote}
          open={rightOpen}
          onSelectSource={handleSelectSource}
        />
      </div>
    </main>
  );
}



