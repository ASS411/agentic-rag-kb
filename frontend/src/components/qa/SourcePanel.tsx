import { ChevronRight, FileSearch } from 'lucide-react';
import clsx from 'clsx';
import type { SearchChunk } from '../../types';
import { SourceCard } from './SourceCard';

export type SourcePanelProps = {
  sources: SearchChunk[];
  selectedSourceId: string | null;
  sourcesNote: string;
  open: boolean;
  onSelectSource: (chunkId: string | null) => void;
  onToggleCollapse: () => void;
};

export function SourcePanel({
  sources,
  selectedSourceId,
  sourcesNote,
  open,
  onSelectSource,
  onToggleCollapse,
}: SourcePanelProps) {
  return (
    <aside className={clsx('source-pane', !open && 'closed')}>
      <div className="section-heading">
        <button
          type="button"
          className="mr-1 p-0.5 rounded hover:bg-popover text-muted-foreground"
          onClick={onToggleCollapse}
          aria-label="折叠来源面板"
        >
          <ChevronRight size={16} />
        </button>
        <span>来源</span>
        <small>{sources.length ? `${sources.length} 引用片段` : '等待回答来源'}</small>
      </div>

      {sources.length === 0 ? (
        <div className="source-empty">
          <FileSearch size={24} aria-hidden="true" />
          <strong>回答引用的片段将显示在这里</strong>
          <span>来源会跟随当前回答流返回，避免展示预检索候选项。</span>
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
          <span>来源摘要</span>
          <pre>{sourcesNote}</pre>
        </div>
      ) : null}
    </aside>
  );
}
