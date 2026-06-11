import clsx from 'clsx';
import type { SearchChunk } from '../../types';
import { trimSource } from '../../lib/utils';

export type SourceCardProps = {
  source: SearchChunk;
  index: number;
  selected: boolean;
  onSelect: (chunkId: string) => void;
};

export function SourceCard({ source, index, selected, onSelect }: SourceCardProps) {
  return (
    <button
      className={clsx('source-card', selected && 'selected')}
      type="button"
      onClick={() => onSelect(source.chunk_id)}
    >
      <span
        className="score-rail"
        style={{ height: `${Math.max(10, Math.round(source.score * 100))}%` }}
      />
      <span className="source-meta">
        #{index + 1} / {source.doc_name || '未知文档'} / p.{source.page}
      </span>
      <strong>{trimSource(source.content).slice(0, 168)}</strong>
      <span className="source-score">
        {Math.round(source.score * 100)}% match
      </span>
    </button>
  );
}
