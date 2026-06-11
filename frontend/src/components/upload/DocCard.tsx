import { AlertTriangle, CheckCircle2, FileText, Loader2, RefreshCw } from 'lucide-react';
import type { DocumentItem } from '../../types';
import { formatBytes, compactDate } from '../../lib/utils';

type VisibleDocumentStatus = 'processing' | 'ready' | 'error';

export type DocCardProps = {
  doc: DocumentItem;
};

function visibleDocumentStatus(doc: DocumentItem): VisibleDocumentStatus {
  if (doc.status === 'error') return 'error';
  if (doc.status === 'ready' || doc.chunk_count > 0) return 'ready';
  return 'processing';
}

function documentStatusLabel(status: VisibleDocumentStatus) {
  if (status === 'error') return '解析失败';
  if (status === 'processing') return '处理中';
  return '已就绪';
}

function DocumentStatusIcon({ status }: { status: VisibleDocumentStatus }) {
  if (status === 'error') {
    return <AlertTriangle size={12} aria-hidden="true" />;
  }
  if (status === 'processing') {
    return <Loader2 className="spin" size={12} aria-hidden="true" />;
  }
  return <CheckCircle2 size={12} aria-hidden="true" />;
}

export function DocCard({ doc }: DocCardProps) {
  const status = visibleDocumentStatus(doc);

  return (
    <article className="doc-row">
      <FileText size={17} aria-hidden="true" />
      <div>
        <strong>{doc.file_name}</strong>
        <div className="doc-meta">
          <span className={`doc-status ${status}`}>
            <DocumentStatusIcon status={status} />
            {documentStatusLabel(status)}
          </span>
          <span>{doc.doc_type.toUpperCase()}</span>
          <span>{doc.chunk_count} chunks</span>
          <span>{formatBytes(doc.size_bytes)}</span>
          <span>{compactDate(doc.uploaded_at)}</span>
        </div>
        {status === 'error' && doc.error_message ? (
          <span className="doc-error">{doc.error_message}</span>
        ) : null}
      </div>
    </article>
  );
}

export type DocListSectionProps = {
  documents: DocumentItem[];
  loading: boolean;
  error: string;
  onRefresh: () => void;
};

export function DocListSection({ documents, loading, error, onRefresh }: DocListSectionProps) {
  return (
    <section className="rail-section">
      <div className="section-heading">
        <span>文档列表</span>
        <button
          className="icon-button"
          type="button"
          onClick={() => void onRefresh()}
          aria-label="刷新文档列表"
        >
          <RefreshCw size={16} aria-hidden="true" />
        </button>
      </div>

      {loading ? (
        <div className="quiet-state">
          <Loader2 className="spin" size={18} aria-hidden="true" />
          正在加载文档
        </div>
      ) : error ? (
        <div className="error-strip">
          <AlertTriangle size={16} aria-hidden="true" />
          {error}
        </div>
      ) : documents.length === 0 ? (
        <div className="empty-copy">上传文档以开始提问。</div>
      ) : (
        <div className="doc-list">
          {documents.map((doc) => (
            <DocCard key={doc.doc_id} doc={doc} />
          ))}
        </div>
      )}
    </section>
  );
}
