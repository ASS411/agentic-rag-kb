import { AlertTriangle, FileText, Loader2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
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
        <div className="doc-meta">
          <span>{formatBytes(doc.size_bytes)}</span>
          <span>{compactDate(doc.uploaded_at)}</span>
        </div>
        {doc.status === 'error' && doc.error_message ? (
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
        <Button
          variant="ghost"
          size="icon"
          type="button"
          onClick={() => void onRefresh()}
          aria-label="刷新文档列表"
        >
          <RefreshCw size={16} aria-hidden="true" />
        </Button>
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
