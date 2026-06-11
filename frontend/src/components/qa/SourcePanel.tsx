import { Search } from 'lucide-react';
import clsx from 'clsx';
import type { SearchChunk } from '../../types';
import { SourceCard } from './SourceCard';

export type SourcePanelProps = {
  sources: SearchChunk[];
  selectedSourceId: string | null;
  sourcesNote: string;
  open: boolean;
  onSelectSource: (chunkId: string | null) => void;
};

export function SourcePanel({
  sources,
  selectedSourceId,
  sourcesNote,
  open,
  onSelectSource,
}: SourcePanelProps) {
  return (
    <aside className={clsx('source-pane', !open && 'closed')}>
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
            <SourceCard
              key={source.chunk_id}
              source={source}
              index={index}
              selected={selectedSourceId === source.chunk_id}
              onSelect={onSelectSource}
            />
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
  );
}
