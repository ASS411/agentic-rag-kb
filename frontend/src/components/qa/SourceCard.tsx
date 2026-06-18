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
  const score = Number.isFinite(source.score) ? Math.max(0, Math.min(1, source.score)) : 0;

  return (
    <button
      className={clsx('source-card', selected && 'selected')}
      type="button"
      data-chunk-id={source.chunk_id}
      onClick={() => onSelect(source.chunk_id)}
    >
      <span
        className="score-rail"
        style={{ height: `${Math.max(10, Math.round(score * 100))}%` }}
      />
      <span className="source-meta">
        #{index + 1} / {source.doc_name || '未知文档'} / 第{source.page}页
      </span>
      <strong>{trimSource(source.content).slice(0, 168)}</strong>
      <span className="source-score">{Math.round(score * 100)}% 匹配度</span>
    </button>
  );
}
