import { useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';

export type SourceHighlightProps = {
  content: string;
  sources: Array<{ chunk_id: string; doc_name?: string; page?: number }>;
  selectedSourceId: string | null;
  onSelectSource: (chunkId: string | null) => void;
  onHoverSource?: (chunkId: string | null) => void;
};

/** Regex matching [来源 N] / [chunk_N] / [chunk N] with optional separators. */
const CITATION_RE = /\[(?:来源|chunk)[ _-]?(\d+)\]/gi;

type Segment =
  | { kind: 'text'; value: string }
  | { kind: 'citation'; sourceIdx: number; label: string };

function splitText(text: string, sourcesCount: number): Segment[] {
  const segments: Segment[] = [];
  let last = 0;
  let match: RegExpExecArray | null;

  // Reset regex state
  CITATION_RE.lastIndex = 0;
  while ((match = CITATION_RE.exec(text)) !== null) {
    const idx = parseInt(match[1], 10);
    const before = text.slice(last, match.index);
    if (before) segments.push({ kind: 'text', value: before });

    if (idx >= 1 && idx <= sourcesCount) {
      segments.push({
        kind: 'citation',
        sourceIdx: idx - 1,
        label: match[0],
      });
    } else {
      // Out of range — keep as plain text
      segments.push({ kind: 'text', value: match[0] });
    }
    last = match.index + match[0].length;
  }

  const tail = text.slice(last);
  if (tail) segments.push({ kind: 'text', value: tail });

  return segments;
}

export function SourceHighlight({
  content,
  sources,
  selectedSourceId,
  onSelectSource,
  onHoverSource,
}: SourceHighlightProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Also apply selected marker styling via a side effect
  const handleSelect = useCallback(
    (sourceIdx: number) => {
      const source = sources[sourceIdx];
      if (!source) return;
      onSelectSource(
        source.chunk_id === selectedSourceId ? null : source.chunk_id,
      );
    },
    [sources, selectedSourceId, onSelectSource],
  );

  if (!content || sources.length === 0) {
    return (
      <ReactMarkdown
        className="message-markdown"
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
      >
        {content}
      </ReactMarkdown>
    );
  }

  const segments = splitText(content, sources.length);

  return (
    <div ref={containerRef} className="source-highlight message-markdown">
      {segments.map((seg, i) => {
        if (seg.kind === 'citation') {
          const source = sources[seg.sourceIdx];
          const isSelected = source?.chunk_id === selectedSourceId;
          return (
            <button
              key={i}
              type="button"
              className={`citation-chip${isSelected ? ' selected' : ''}`}
              data-source-index={seg.sourceIdx}
              title={source ? `${source.doc_name ?? ''} 第${source.page ?? '?'}页` : ''}
              onClick={() => handleSelect(seg.sourceIdx)}
              onMouseEnter={() => {
                if (source && onHoverSource) onHoverSource(source.chunk_id);
              }}
              onMouseLeave={() => {
                if (onHoverSource) onHoverSource(null);
              }}
            >
              [{seg.sourceIdx + 1}]
            </button>
          );
        }
        // Text segment — render as markdown
        return (
          <ReactMarkdown
            key={i}
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              // Keep inline rendering
              p: ({ children, ...props }) => <span {...props}>{children}</span>,
            }}
          >
            {seg.value}
          </ReactMarkdown>
        );
      })}
    </div>
  );
}
