import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Menu } from 'lucide-react';

import type { DocumentItem, UploadState } from './types';
import { fetchDocuments, uploadFile } from './api/documents';
import { fetchConversations } from './api/history';
import { useQAStore } from './stores/qaStore';
import { useSourceScroll } from './hooks/useSourceScroll';

import { Sidebar } from './components/layout/Sidebar';
import { DropZone } from './components/upload/DropZone';
import { UploadModal } from './components/upload/UploadModal';
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

const PROCESSING_POLL_INTERVAL_MS = 1500;
const PROCESSING_POLL_MAX_ATTEMPTS = 24;

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isDocumentReady(doc: DocumentItem) {
  return doc.status === 'ready' || doc.chunk_count > 0;
}

export default function App() {
  const qc = useQueryClient();

  // ── Documents (TanStack Query) ────────────────────────────────
  const { data: documentData, isLoading: docLoading, error: docError, refetch: refreshDocuments } = useQuery({
    queryKey: ['documents'],
    queryFn: fetchDocuments,
    staleTime: 10_000,
  });
  const documents: DocumentItem[] = documentData?.items ?? [];
  const docErrorMsg = docError instanceof Error ? docError.message : '';

  // ── Conversations history ──────────────────────────────────────
  const { data: convData } = useQuery({
    queryKey: ['conversations'],
    queryFn: () => fetchConversations(1, 20),
    staleTime: 15_000,
  });
  const conversations = convData?.items ?? [];

  // ── Zustand QA state ──────────────────────────────────────────
  const messages = useQAStore((s) => s.messages);
  const sources = useQAStore((s) => s.sources);
  const sourcesNote = useQAStore((s) => s.sourcesNote);
  const thinkingSteps = useQAStore((s) => s.thinkingSteps);
  const selectedSourceId = useQAStore((s) => s.selectedSourceId);

  const { scrollToCitation } = useSourceScroll();
  const handleSelectSource = useCallback((chunkId: string | null) => {
    useQAStore.getState().selectSource(chunkId);
    scrollToCitation(chunkId);
  }, [scrollToCitation]);

  // ── Layout state ──────────────────────────────────────────────
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const userScrolledUpRef = useRef(false);

  // SourcePanel visible only when sources exist AND user hasn't manually hidden it
  const [sourcePanelPinned, setSourcePanelPinned] = useState(true);
  const sourcesVisible = sources.length > 0 && sourcePanelPinned;

  const readyDocs = useMemo(() => documents.filter(isDocumentReady).length, [documents]);

  // ── Upload state ──────────────────────────────────────────────
  const [upload, setUpload] = useState<UploadState>({ phase: 'idle', progress: 0, message: '支持 PDF、Markdown、TXT' });

  // ── New conversation ──────────────────────────────────────────
  const handleNewConversation = useCallback(() => {
    useQAStore.getState().reset();
    setActiveConversationId(null);
  }, []);

  // ── Load conversation from history ────────────────────────────
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const loadConversation = useCallback(async (conversationId: string) => {
    const { fetchHistory } = await import('./api/history');
    const result = await fetchHistory(conversationId);
    useQAStore.getState().reset();
    // API returns newest first — reverse for chronological order
    const records = [...result.items].reverse();
    let lastSources: any[] = [];
    for (const record of records) {
      useQAStore.getState().addMessage({ id: crypto.randomUUID(), role: 'user', content: record.question });
      // Skip generating records with no answer (interrupted / stale)
      if (!record.answer && record.status === 'generating') {
        continue;
      }
      useQAStore.getState().addMessage({ id: crypto.randomUUID(), role: 'assistant', content: record.answer || '（生成已中断）' });
      if (record.sources?.length) {
        lastSources = record.sources;
      }
    }
    // Set sources from the last record that had them
    if (lastSources.length > 0) {
      // Map stored sources (content_snippet) to SearchChunk shape (content)
      const mapped = lastSources.map((s: any) => ({ ...s, content: s.content_snippet || '', doc_type: s.doc_type || 'pdf' }));
      useQAStore.getState().setSources(mapped as any, '');
    }
    setActiveConversationId(conversationId);
    void qc.invalidateQueries({ queryKey: ['conversations'] });
  }, [qc]);

  // ── Chat stream ───────────────────────────────────────────────
  const chat = useChatStream({
    onStart: (userMsg, assistantMsg) => {
      useQAStore.getState().addMessage(userMsg);
      useQAStore.getState().addMessage(assistantMsg);
      useQAStore.getState().setSources([], '');
      useQAStore.getState().clearThinkingSteps();
      useQAStore.getState().selectSource(null);
    },
    onToken: (assistantId, token) => {
      useQAStore.getState().updateMessage(assistantId, token);
    },
    onSources: (text, sourceChunks) => {
      useQAStore.getState().setSources(sourceChunks, text);
      setSourcePanelPinned(true);
    },
    onDone: () => {
      // Refresh history — backend now persists BEFORE stream starts,
      // so the record is already in the database.
      setTimeout(() => {
        void qc.invalidateQueries({ queryKey: ['conversations'] });
      }, 500);
    },
    onMeta: (conversationId, _recordId) => {
      setActiveConversationId(conversationId);
      // Immediately refresh sidebar so the new conversation appears
      void qc.invalidateQueries({ queryKey: ['conversations'] });
    },
    onAgentStep: (step) => {
      useQAStore.getState().addThinkingStep(step);
    },
    onError: (assistantId, message) => {
      const msgs = useQAStore.getState().messages;
      useQAStore.getState().setMessages(msgs.map((m) =>
        m.id === assistantId && !m.content ? { ...m, content: `回答生成失败：${message}` } : m));
    },
  });

  const handleQuestion = useCallback((q: string) => { void chat.submit(q); }, [chat.submit]);

  // ── Upload handler ────────────────────────────────────────────
  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const file = Array.from(files)[0]; if (!file) return;
    setUpload({ phase: 'uploading', progress: 4, message: `正在上传 ${file.name}` });
    try {
      const uploaded = await uploadFile(file, (p) => setUpload({ phase: 'uploading', progress: p, message: `正在上传 ${file.name}` }));
      setUpload({ phase: 'processing', progress: 92, message: `${uploaded.file_name} 正在解析并写入索引` });
      await refreshDocuments();
      let processed: DocumentItem | null = null;
      for (let i = 0; i < PROCESSING_POLL_MAX_ATTEMPTS; i++) {
        await wait(PROCESSING_POLL_INTERVAL_MS);
        const r = await fetchDocuments();
        const f = r.items.find((d) => d.doc_id === uploaded.doc_id);
        if (!f) continue;
        if (f.status === 'error') { processed = f; break; }
        if (isDocumentReady(f)) { processed = f; break; }
      }
      if (processed?.status === 'error') setUpload({ phase: 'error', progress: 0, message: processed.error_message || `${processed.file_name} 解析失败` });
      else if (processed) {
        setUpload({ phase: 'success', progress: 100, message: `${processed.file_name} 已就绪，生成 ${processed.chunk_count} 个片段` });
        setUploadModalOpen(false);
        void qc.invalidateQueries({ queryKey: ['documents'] });
      } else setUpload({ phase: 'processing', progress: 96, message: `${uploaded.file_name} 仍在后台处理中` });
    } catch (e) { setUpload({ phase: 'error', progress: 0, message: e instanceof Error ? e.message : '上传失败' }); }
  }, [refreshDocuments, qc]);

  // ── Smart auto-scroll ─────────────────────────────────────────
  useEffect(() => {
    const c = scrollContainerRef.current; if (!c) return;
    const h = () => { userScrolledUpRef.current = c.scrollHeight - c.scrollTop - c.clientHeight > 80; };
    c.addEventListener('scroll', h, { passive: true });
    return () => c.removeEventListener('scroll', h);
  }, []);

  // Auto-scroll to bottom when new content arrives, only if user
  // hasn't deliberately scrolled up.  Uses `auto` (instant) to avoid
  // fighting the browser's native scroll — `smooth` creates an
  // animation that blocks user input when called in rapid succession
  // (e.g. every token during streaming).
  const scrollFrameRef = useRef<number | null>(null);
  useEffect(() => {
    if (!userScrolledUpRef.current) {
      // Throttle to at most one scroll per animation frame
      if (scrollFrameRef.current !== null) cancelAnimationFrame(scrollFrameRef.current);
      scrollFrameRef.current = requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'auto' });
        scrollFrameRef.current = null;
      });
    }
  }, [messages, chat.streaming]);
  useEffect(() => { if (chat.streaming) userScrolledUpRef.current = false; }, [chat.streaming]);

  // ── Shell classes ─────────────────────────────────────────────
  const shellClass = clsx('workspace-shell',
    sidebarExpanded && 'sidebar-expanded',
    sourcesVisible && 'sources-visible',
  );

  return (
    <main className="min-h-screen text-foreground">
      <div className={shellClass}>
        {/* ── Sidebar ──────────────────────────────────── */}
        <Sidebar
          collapsed={!sidebarExpanded}
          onToggle={() => setSidebarExpanded((v) => !v)}
          mobileOpen={mobileSidebarOpen}
          onMobileClose={() => setMobileSidebarOpen(false)}
          onUploadClick={() => setUploadModalOpen(true)}
          onNewConversation={handleNewConversation}
          history={
            conversations.length > 0 ? (
              conversations.map((c) => (
                <button
                  key={c.conversation_id}
                  className={`history-item${activeConversationId === c.conversation_id ? ' active' : ''}`}
                  type="button"
                  title={c.last_question}
                  onClick={() => { void loadConversation(c.conversation_id); }}
                >
                  {c.title || c.last_question?.slice(0, 40) || '新对话'}
                </button>
              ))
            ) : (
              <p className="text-xs text-muted-foreground px-2">暂无历史对话</p>
            )
          }
        >
          <DocListSection documents={documents} loading={docLoading} error={docErrorMsg} onRefresh={() => { void refreshDocuments(); }} />
        </Sidebar>

        {/* ── Conversation pane ───────────────────────── */}
        <section className="conversation-pane">
          {/* Mobile hamburger */}
          <div className="flex items-center gap-2 px-4 py-2 md:hidden border-b border-[hsl(var(--border))]">
            <button className="mobile-hamburger" type="button" onClick={() => setMobileSidebarOpen(true)} aria-label="菜单">
              <Menu size={20} />
            </button>
            <span className="text-sm font-semibold text-foreground">证据库</span>
            <span className="text-xs text-muted-foreground ml-auto">{readyDocs}/{documents.length} 文档</span>
          </div>

          <div className="message-scroll" ref={scrollContainerRef}>
            {messages.length === 0 ? (
              <WelcomePanel examples={EXAMPLES} streaming={chat.streaming} onSubmitExample={handleQuestion} />
            ) : (
              <AnswerPanel
                messages={messages}
                error={chat.error}
                sources={sources}
                selectedSourceId={selectedSourceId}
                onSelectSource={handleSelectSource}
                thinkingSteps={thinkingSteps}
              />
            )}
            <div ref={bottomRef} />
          </div>

          <QuestionInput
            streaming={chat.streaming}
            hasDocuments={readyDocs > 0}
            hasSources={sources.length > 0}
            onSubmit={handleQuestion}
            onStop={chat.stop}
            onToggleSourcePanel={() => setSourcePanelPinned((v) => !v)}
          />
        </section>

        {/* ── Source panel ─────────────────────────────── */}
        <SourcePanel
          sources={sources}
          selectedSourceId={selectedSourceId}
          sourcesNote={sourcesNote}
          open={sourcesVisible}
          onSelectSource={handleSelectSource}
          onToggleCollapse={() => setSourcePanelPinned((v) => !v)}
        />
      </div>

      {/* ── Upload modal ────────────────────────────────── */}
      <UploadModal open={uploadModalOpen} onClose={() => setUploadModalOpen(false)}>
        <DropZone upload={upload} onFiles={handleFiles} />
      </UploadModal>
    </main>
  );
}



