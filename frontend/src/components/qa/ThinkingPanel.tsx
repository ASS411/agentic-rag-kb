import { Brain } from 'lucide-react';

export type ThinkingStep = {
  step: string;
  message?: string;
  queries?: string[];
  count?: number;
  verdict?: string;
  reasoning?: string;
  gap?: string;
};

export type ThinkingPanelProps = {
  steps: ThinkingStep[];
  expanded: boolean;
  onToggle: () => void;
};

/**
 * Displays the Agent's thinking process in a collapsible panel.
 *
 * In Phase 1 MVP the Agent loop is not yet implemented, so this panel renders
 * a placeholder.  When Phase 2 (Agent loop) is integrated the SSE events will
 * populate ``steps`` in real time.
 */
export function ThinkingPanel({ steps, expanded, onToggle }: ThinkingPanelProps) {
  if (steps.length === 0) return null;

  return (
    <div className="thinking-panel">
      <button
        className="thinking-toggle"
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <Brain size={16} aria-hidden="true" />
        <span>思考过程</span>
        <span className="thinking-count">{steps.length}</span>
      </button>

      {expanded && (
        <ol className="thinking-steps">
          {steps.map((s, i) => (
            <li key={i} className="thinking-step">
              <span className="thinking-step-label">{s.step}</span>
              {s.message ? (
                <span className="thinking-step-msg">{s.message}</span>
              ) : null}
              {s.queries?.length ? (
                <ul className="thinking-queries">
                  {s.queries.map((q, qi) => (
                    <li key={qi}>{q}</li>
                  ))}
                </ul>
              ) : null}
              {s.count != null ? (
                <span className="thinking-step-meta">{s.count} 条候选</span>
              ) : null}
              {s.verdict ? (
                <span className={`thinking-verdict ${s.verdict}`}>
                  {s.verdict === 'sufficient' ? '✅' : '⚠️'} {s.verdict}
                </span>
              ) : null}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
