import { useState } from 'react';
import { AlertTriangle, FileText, Loader2, RefreshCw, Trash2 } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import type { DocumentItem } from '../../types';
import { formatBytes, compactDate } from '../../lib/utils';
import { deleteDocument } from '../../api/documents';

export type DocCardProps = {
  doc: DocumentItem;
};

function DeleteDocButton({ doc }: DocCardProps) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await deleteDocument(doc.doc_id);
      void qc.invalidateQueries({ queryKey: ['documents'] });
    } catch {
      // keep dialog open on error
    } finally {
      setDeleting(false);
      setOpen(false);
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 ml-auto shrink-0"
        type="button"
        aria-label={`删除 ${doc.file_name}`}
        onClick={(e) => { e.stopPropagation(); setOpen(true); }}
      >
        <Trash2 size={13} className="text-muted-foreground hover:text-destructive" />
      </Button>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>确认删除文档</AlertDialogTitle>
          <AlertDialogDescription>
            将删除 <strong>{doc.file_name}</strong> 及其所有索引数据，此操作不可撤销。
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={deleting}>取消</AlertDialogCancel>
          <AlertDialogAction
            disabled={deleting}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            onClick={(e) => { e.preventDefault(); void handleDelete(); }}
          >
            {deleting ? '删除中...' : '删除'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

export function DocCard({ doc }: DocCardProps) {
  return (
    <article className="doc-row">
      <FileText size={17} aria-hidden="true" />
      <div className="min-w-0">
        <div className="flex items-center gap-1">
          <strong className="truncate">{doc.file_name}</strong>
          <DeleteDocButton doc={doc} />
        </div>
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
