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
        <span>来源</span>
        <small>{sources.length ? `${sources.length} 候选项` : '等待中'}</small>
      </div>

      {sources.length === 0 ? (
        <div className="source-empty">
          <Search size={24} aria-hidden="true" />
          <strong>检索到的片段将显示在这里</strong>
          <span>每张卡片展示摘录、文档、页码和匹配分数。</span>
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
          <span>流式来源数据</span>
          <pre>{sourcesNote}</pre>
        </div>
      ) : null}
    </aside>
  );
}
