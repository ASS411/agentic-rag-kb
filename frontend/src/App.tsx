import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  FileText,
  Loader2,
  PanelRight,
  RefreshCw,
  Search,
  Send,
  UploadCloud,
} from 'lucide-react';
import clsx from 'clsx';

type ApiEnvelope<T> = {
  success: boolean;
  code: number;
  message: string;
  data: T;
};

type DocumentItem = {
  doc_id: string;
  file_name: string;
  doc_type: 'pdf' | 'md' | 'txt';
  size_bytes: number;
  page_count: number;
  chunk_count: number;
  uploaded_at: string;
};

type DocumentList = {
  items: DocumentItem[];
  total: number;
  page: number;
  size: number;
};

type SearchChunk = {
  chunk_id: string;
  content: string;
  score: number;
  doc_id: string;
  doc_name: string;
  doc_type: string;
  page: number;
  chunk_index: number;
  metadata: Record<string, unknown>;
};

type SearchResponse = {
  query: string;
  total_results: number;
  results: SearchChunk[];
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
};

type UploadState = {
  phase: 'idle' | 'uploading' | 'success' | 'error';
  progress: number;
  message: string;
};

type StreamEvent = {
  type: 'token' | 'sources' | 'done' | 'error';
  content: string;
};

const API_BASE = '/api/v1';

const examples = [
  '总结这批文档里的关键结论，并列出依据。',
  '这份资料中有哪些风险点需要优先处理？',
  '把相关段落整理成一个三点行动清单。',
];

function formatBytes(bytes: number) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function compactDate(value: string) {
  if (!value) return '刚刚';
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value));
}

function trimSource(text: string) {
  return text.replace(/\s+/g, ' ').trim();
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T;
  if (!response.ok) {
    const message =
      typeof payload === 'object' && payload !== null && 'message' in payload
        ? String((payload as { message?: unknown }).message)
        : response.statusText;
    throw new Error(message || '请求失败');
  }
  return payload;
}

async function fetchDocuments(): Promise<DocumentList> {
  const payload = await readJson<ApiEnvelope<DocumentList>>(
    await fetch(`${API_BASE}/documents?page=1&size=30`),
  );
  return payload.data;
}

async function searchChunks(question: string): Promise<SearchChunk[]> {
  const payload = await readJson<ApiEnvelope<SearchResponse>>(
    await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: question, top_k: 5 }),
    }),
  );
  return payload.data.results;
}

function uploadFile(
  file: File,
  onProgress: (progress: number) => void,
): Promise<DocumentItem> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const form = new FormData();
    form.append('file', file);

    request.open('POST', `${API_BASE}/documents/upload`);

    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    request.onload = () => {
      try {
        const payload = JSON.parse(request.responseText) as ApiEnvelope<DocumentItem>;
        if (request.status >= 200 && request.status < 300 && payload.success) {
          resolve(payload.data);
          return;
        }
        reject(new Error(payload.message || '上传失败'));
      } catch {
        reject(new Error('无法解析上传响应'));
      }
    };

    request.onerror = () => reject(new Error('上传连接失败'));
    request.send(form);
  });
}

function parseSseBlock(block: string): StreamEvent | null {
  const data = block
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');

  if (!data) return null;

  try {
    const parsed = JSON.parse(data) as StreamEvent;
    if (parsed.type === 'token' || parsed.type === 'sources' || parsed.type === 'done' || parsed.type === 'error') {
      return parsed;
    }
  } catch {
    return null;
  }

  return null;
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [docLoading, setDocLoading] = useState(true);
  const [docError, setDocError] = useState('');
  const [upload, setUpload] = useState<UploadState>({
    phase: 'idle',
    progress: 0,
    message: '支持 PDF、Markdown、TXT',
  });
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<SearchChunk[]>([]);
  const [sourcesNote, setSourcesNote] = useState('');
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [chatError, setChatError] = useState('');
  const [rightOpen, setRightOpen] = useState(true);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: streaming ? 'smooth' : 'auto' });
  }, [messages, streaming]);

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

  const submitQuestion = useCallback(
    async (nextQuestion = question.trim()) => {
      if (!nextQuestion || streaming) return;

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: nextQuestion,
      };
      const assistantId = crypto.randomUUID();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
      };

      setQuestion('');
      setChatError('');
      setSources([]);
      setSourcesNote('');
      setSelectedSourceId(null);
      setMessages((current) => [...current, userMessage, assistantMessage]);
      setStreaming(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        searchChunks(nextQuestion)
          .then(setSources)
          .catch(() => setSources([]));

        const response = await fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: nextQuestion, stream: true, top_k: 5 }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          throw new Error(response.statusText || '流式回答连接失败');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split('\n\n');
          buffer = blocks.pop() ?? '';

          for (const block of blocks) {
            const event = parseSseBlock(block);
            if (!event) continue;

            if (event.type === 'token') {
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantId
                    ? { ...message, content: message.content + event.content }
                    : message,
                ),
              );
            }

            if (event.type === 'sources') {
              setSourcesNote(event.content);
            }

            if (event.type === 'error') {
              throw new Error(event.content || '生成回答失败');
            }
          }
        }

        if (buffer.trim()) {
          const event = parseSseBlock(buffer);
          if (event?.type === 'sources') setSourcesNote(event.content);
        }
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          const message = error instanceof Error ? error.message : '生成回答失败';
          setChatError(message);
          setMessages((current) =>
            current.map((item) =>
              item.id === assistantId && !item.content
                ? { ...item, content: `回答生成失败：${message}` }
                : item,
            ),
          );
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [question, streaming],
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
  }, []);

  return (
    <main className="min-h-screen text-slate-100">
      <div className={clsx('workspace-shell', !rightOpen && 'sources-collapsed')}>
        <aside className="left-rail">
          <div className="brand-lockup">
            <div className="brand-mark">
              <Database size={20} aria-hidden="true" />
            </div>
            <div>
              <p className="eyebrow">Agentic RAG</p>
              <h1>知识证据台</h1>
            </div>
          </div>

          <section
            className={clsx('upload-pad', upload.phase)}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              void handleFiles(event.dataTransfer.files);
            }}
          >
            <input
              ref={fileInputRef}
              className="sr-only"
              type="file"
              accept=".pdf,.md,.markdown,.txt,application/pdf,text/markdown,text/plain"
              onChange={(event) => {
                if (event.target.files) void handleFiles(event.target.files);
                event.currentTarget.value = '';
              }}
            />
            <button
              className="upload-target"
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={upload.phase === 'uploading'}
            >
              {upload.phase === 'uploading' ? (
                <Loader2 className="spin" size={22} aria-hidden="true" />
              ) : (
                <UploadCloud size={24} aria-hidden="true" />
              )}
              <span>拖入文档或点击上传</span>
              <small>{upload.message}</small>
            </button>
            <div className="progress-track" aria-label="上传进度">
              <span style={{ width: `${upload.progress}%` }} />
            </div>
          </section>

          <section className="rail-section">
            <div className="section-heading">
              <span>文档</span>
              <button
                className="icon-button"
                type="button"
                onClick={() => void refreshDocuments()}
                aria-label="刷新文档列表"
              >
                <RefreshCw size={16} aria-hidden="true" />
              </button>
            </div>

            {docLoading ? (
              <div className="quiet-state">
                <Loader2 className="spin" size={18} aria-hidden="true" />
                正在读取文档
              </div>
            ) : docError ? (
              <div className="error-strip">
                <AlertTriangle size={16} aria-hidden="true" />
                {docError}
              </div>
            ) : documents.length === 0 ? (
              <div className="empty-copy">上传第一份资料后，就可以开始提问。</div>
            ) : (
              <div className="doc-list">
                {documents.map((doc) => (
                  <article className="doc-row" key={doc.doc_id}>
                    <FileText size={17} aria-hidden="true" />
                    <div>
                      <strong>{doc.file_name}</strong>
                      <span>
                        {doc.doc_type.toUpperCase()} / {doc.chunk_count} chunks /{' '}
                        {formatBytes(doc.size_bytes)} / {compactDate(doc.uploaded_at)}
                      </span>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        </aside>

        <section className="conversation-pane">
          <header className="topbar">
            <div>
              <p className="eyebrow">Evidence Desk</p>
              <h2>知识问答工作台</h2>
            </div>
            <div className="status-pills">
              <span>
                <CheckCircle2 size={15} aria-hidden="true" />
                {documents.length} 份文档
              </span>
              <span>
                <Search size={15} aria-hidden="true" />
                {readyDocs} 份可检索
              </span>
              <button
                className="icon-button"
                type="button"
                onClick={() => setRightOpen((open) => !open)}
                aria-label="切换来源面板"
              >
                <PanelRight size={17} aria-hidden="true" />
              </button>
            </div>
          </header>

          <div className="message-scroll">
            {messages.length === 0 ? (
              <div className="start-panel">
                <div className="start-copy">
                  <span className="signal-line" />
                  <p className="eyebrow">Evidence-first Q&A</p>
                  <h2>把问题丢进知识库，让答案带着来源回来。</h2>
                  <p>
                    适合把零散资料放在同一张桌面上审阅：答案在中间生长，证据在旁边排队等待核验。
                  </p>
                </div>
                <div className="example-grid">
                  {examples.map((item) => (
                    <button
                      type="button"
                      key={item}
                      onClick={() => void submitQuestion(item)}
                      disabled={streaming}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <article className={clsx('message', message.role)} key={message.id}>
                  <div className="message-label">{message.role === 'user' ? '你' : '知识库'}</div>
                  <div className="message-body">
                    {message.content || (
                      <span className="typing">
                        <Loader2 className="spin" size={16} aria-hidden="true" />
                        正在组织答案
                      </span>
                    )}
                  </div>
                </article>
              ))
            )}
            {chatError ? (
              <div className="chat-error">
                <AlertTriangle size={16} aria-hidden="true" />
                {chatError}
              </div>
            ) : null}
            <div ref={bottomRef} />
          </div>

          <form
            className="question-dock"
            onSubmit={(event) => {
              event.preventDefault();
              void submitQuestion();
            }}
          >
            <textarea
              value={question}
              placeholder={documents.length ? '向当前知识库提问...' : '先上传文档，也可以直接试问接口状态...'}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void submitQuestion();
                }
              }}
              disabled={streaming}
              rows={1}
            />
            {streaming ? (
              <button className="send-button stop" type="button" onClick={stopStreaming}>
                停止
              </button>
            ) : (
              <button className="send-button" type="submit" disabled={!question.trim()}>
                <Send size={17} aria-hidden="true" />
                发送
              </button>
            )}
          </form>
        </section>

        <aside className={clsx('source-pane', !rightOpen && 'closed')}>
          <div className="section-heading">
            <span>来源面板</span>
            <small>{sources.length ? `${sources.length} 条候选` : '等待检索'}</small>
          </div>

          {sources.length === 0 ? (
            <div className="source-empty">
              <Search size={24} aria-hidden="true" />
              <strong>提问后显示检索 chunk</strong>
              <span>每张卡片保留摘要、文件名、页码和相似度。</span>
            </div>
          ) : (
            <div className="source-list">
              {sources.map((source, index) => (
                <button
                  className={clsx('source-card', selectedSourceId === source.chunk_id && 'selected')}
                  type="button"
                  key={source.chunk_id}
                  onClick={() => setSelectedSourceId(source.chunk_id)}
                >
                  <span
                    className="score-rail"
                    style={{ height: `${Math.max(10, Math.round(source.score * 100))}%` }}
                  />
                  <span className="source-meta">
                    #{index + 1} / {source.doc_name || '未知文档'} / p.{source.page}
                  </span>
                  <strong>{trimSource(source.content).slice(0, 168)}</strong>
                  <span className="source-score">{Math.round(source.score * 100)}% match</span>
                </button>
              ))}
            </div>
          )}

          {sourcesNote ? (
            <div className="sources-note">
              <span>模型返回来源</span>
              <pre>{sourcesNote}</pre>
            </div>
          ) : null}
        </aside>
      </div>
    </main>
  );
}
