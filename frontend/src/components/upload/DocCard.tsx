import { AlertTriangle, FileText, Loader2, RefreshCw } from 'lucide-react';
import type { DocumentItem } from '../../types';
import { formatBytes, compactDate } from '../../lib/utils';

export type DocCardProps = {
  doc: DocumentItem;
};

export function DocCard({ doc }: DocCardProps) {
  return (
    <article className="doc-row">
      <FileText size={17} aria-hidden="true" />
      <div>
        <strong>{doc.file_name}</strong>
        <span>
          {doc.doc_type.toUpperCase()} / {doc.chunk_count} chunks /{' '}
          {formatBytes(doc.size_bytes)} / {compactDate(doc.uploaded_at)}
        </span>
      </div>
    </article>
  );
}

// ---------------------------------------------------------------------------
// Document list section (used inside Sidebar)
// ---------------------------------------------------------------------------

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
        <span>文档</span>
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
          正在读取文档
        </div>
      ) : error ? (
        <div className="error-strip">
          <AlertTriangle size={16} aria-hidden="true" />
          {error}
        </div>
      ) : documents.length === 0 ? (
        <div className="empty-copy">上传第一份资料后，就可以开始提问。</div>
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
