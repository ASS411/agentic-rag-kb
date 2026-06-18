import { ChevronLeft, ChevronRight, FileSearch } from 'lucide-react';
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
    <>
      {/* Collapsed re-expand bar */}
      {!open && sources.length > 0 ? (
        <button
          type="button"
          className="fixed right-0 top-1/2 -translate-y-1/2 z-20 w-7 h-20 rounded-l-md bg-card border border-r-0 border-border flex items-center justify-center hover:bg-popover transition-colors"
          onClick={onToggleCollapse}
          aria-label="展开来源面板"
        >
          <ChevronLeft size={14} className="text-muted-foreground" />
        </button>
      ) : null}

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
    </>
  );
}
