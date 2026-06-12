import { useCallback } from 'react';
import { useQAStore } from '../stores/qaStore';

/**
 * Hook for synchronising scroll position between source cards and
 * citation markers in the answer panel.
 *
 * When a source card is selected, scroll the corresponding citation
 * marker into view.  When a citation marker is clicked, that marker
 * is scrolled into view automatically by the browser's default
 * ``scrollIntoView`` behaviour on focus.
 */
export function useSourceScroll() {
  const selectedSourceId = useQAStore((s) => s.selectedSourceId);

  /**
   * Scroll the citation marker for *chunkId* into view within the
   * answer panel.
   */
  const scrollToCitation = useCallback((chunkId: string | null) => {
    if (!chunkId) return;

    // Find the citation marker that references this source
    const markers = document.querySelectorAll('.citation-chip');
    for (const marker of markers) {
      const el = marker as HTMLElement;
      const idxStr = el.dataset.sourceIndex;
      if (idxStr == null) continue;

      const sources = useQAStore.getState().sources;
      const idx = parseInt(idxStr, 10);
      if (sources[idx]?.chunk_id === chunkId) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.classList.add('flash');
        setTimeout(() => el.classList.remove('flash'), 600);
        return;
      }
    }
  }, []);

  /**
   * Scroll the source card for *chunkId* into view within the
   * source panel.
   */
  const scrollToSource = useCallback((chunkId: string | null) => {
    if (!chunkId) return;

    // Find the source card button
    const cards = document.querySelectorAll('.source-card');
    for (const card of cards) {
      const btn = card as HTMLElement;
      // SourceCards are identified by the click handler — we match by the
      // source meta text containing doc_name or chunk info.
      // Instead, use a data attribute approach via the source list.
      if (btn.dataset.chunkId === chunkId) {
        btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
    }

    // Fallback: try finding by chunk_id in source list
    const allSourceCards = document.querySelectorAll('.source-card');
    const sources = useQAStore.getState().sources;
    const idx = sources.findIndex((s) => s.chunk_id === chunkId);
    if (idx >= 0 && allSourceCards[idx]) {
      (allSourceCards[idx] as HTMLElement).scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, []);

  return { selectedSourceId, scrollToCitation, scrollToSource };
}

