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

/**
 * Detect citation markers in text and wrap them in interactive <mark> elements.
 *
 * Matches [来源 N], [来源N], [chunk_N], [chunk N] variants.
 * Uses 1-based indexing (N starts at 1, maps to sources[N-1]).
 */
function highlightCitations(
  text: string,
  sourcesCount: number,
): string {
  if (!text || sourcesCount === 0) return text;

  // Match all citation formats
  const pattern = /\[(?:来源|chunk)[ _-]?(\d+)\]/gi;

  return text.replace(pattern, (_match, digits) => {
    const index = parseInt(digits, 10);
    if (index < 1 || index > sourcesCount) {
      return _match; // out of range — leave unchanged
    }
    return (
      '<mark class="citation-marker" ' +
      `data-source-index="${index - 1}" ` +
      `data-citation="${_match}">` +
      `&#91;来源 ${index}&#93;` +
      '</mark>'
    );
  });
}

/**
 * Renders answer markdown with interactive citation markers.
 *
 * Citation patterns like `[来源 1]` are rendered as clickable
 * <mark> elements.  Clicking one selects the corresponding source
 * card; hovering highlights it.
 */
export function SourceHighlight({
  content,
  sources,
  selectedSourceId,
  onSelectSource,
  onHoverSource,
}: SourceHighlightProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  const processed = highlightCitations(content, sources.length);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      const target = e.target as HTMLElement;
      const marker = target.closest?.('.citation-marker') as HTMLElement | null;
      if (!marker) return;

      const index = marker.dataset.sourceIndex;
      if (index == null) return;

      const source = sources[parseInt(index, 10)];
      if (!source) return;

      // Toggle: clicking an already-selected marker deselects it
      onSelectSource(
        source.chunk_id === selectedSourceId ? null : source.chunk_id,
      );
    },
    [sources, selectedSourceId, onSelectSource],
  );

  const handleMouseOver = useCallback(
    (e: React.MouseEvent) => {
      const target = e.target as HTMLElement;
      const marker = target.closest?.('.citation-marker') as HTMLElement | null;
      if (!marker) return;

      const index = marker.dataset.sourceIndex;
      if (index == null) return;

      const source = sources[parseInt(index, 10)];
      if (source && onHoverSource) {
        onHoverSource(source.chunk_id);
      }
    },
    [sources, onHoverSource],
  );

  const handleMouseOut = useCallback(
    (e: React.MouseEvent) => {
      const target = e.target as HTMLElement;
      const marker = target.closest?.('.citation-marker') as HTMLElement | null;
      if (!marker) return;
      if (onHoverSource) onHoverSource(null);
    },
    [onHoverSource],
  );

  // Style the selected marker
  if (selectedSourceId && containerRef.current) {
    const idx = sources.findIndex((s) => s.chunk_id === selectedSourceId);
    const markers = containerRef.current.querySelectorAll('.citation-marker');
    markers.forEach((m) => {
      const el = m as HTMLElement;
      const i = parseInt(el.dataset.sourceIndex ?? '', 10);
      el.classList.toggle('selected', i === idx);
    });
  }

  return (
    <div
      ref={containerRef}
      className="source-highlight"
      onClick={handleClick}
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      <ReactMarkdown
        className="message-markdown"
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
      >
        {processed}
      </ReactMarkdown>
    </div>
  );
}
