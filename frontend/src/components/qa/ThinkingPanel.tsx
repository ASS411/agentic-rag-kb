import clsx from 'clsx';
import {
  ChevronDown,
  CircuitBoard,
  DatabaseZap,
  Search,
  Sparkles,
  TriangleAlert,
  WandSparkles,
} from 'lucide-react';
import { useMemo } from 'react';
import type { ThinkingStep } from '../../types';

export type ThinkingPanelProps = {
  steps: ThinkingStep[];
  expanded: boolean;
  onToggle: () => void;
};

const STEP_META: Record<
  string,
  {
    label: string;
    icon: typeof Search;
  }
> = {
  rewrite: { label: '查询改写', icon: WandSparkles },
  search: { label: '多路检索', icon: Search },
  rerank: { label: '精排', icon: CircuitBoard },
  check: { label: '质量检查', icon: DatabaseZap },
  replan: { label: '补充检索', icon: TriangleAlert },
  generate: { label: '生成回答', icon: Sparkles },
  done: { label: '完成', icon: Sparkles },
};

function formatVerdict(verdict?: string) {
  if (!verdict) return '';
  if (verdict === 'sufficient') return '证据充足';
  if (verdict === 'insufficient') return '需要更多上下文';
  return verdict;
}

function StepDot({ active }: { active: boolean }) {
  return <span className={clsx('thinking-dot', active && 'active')} aria-hidden="true" />;
}

export function ThinkingPanel({ steps, expanded, onToggle }: ThinkingPanelProps) {
  const visibleSteps = useMemo(() => steps.slice(-8), [steps]);
  if (visibleSteps.length === 0) return null;

  const latest = visibleSteps[visibleSteps.length - 1];

  return (
    <section className={clsx('thinking-panel', expanded && 'expanded')}>
      <button
        className="thinking-toggle"
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <div className="thinking-toggle-copy">
          <span className="thinking-kicker">代理追踪</span>
          <span className="thinking-title">思考面板</span>
        </div>
        <span className="thinking-toggle-meta">
          <span className="thinking-count">{visibleSteps.length}</span>
          <ChevronDown
            size={16}
            aria-hidden="true"
            className={clsx('thinking-chevron', expanded && 'open')}
          />
        </span>
      </button>

      {expanded ? (
        <ol className="thinking-steps">
          {visibleSteps.map((step, index) => {
            const meta = STEP_META[step.step] ?? STEP_META.generate;
            const Icon = meta.icon;
            const isLatest = index === visibleSteps.length - 1;

            return (
              <li className={clsx('thinking-step', isLatest && 'latest')} key={`${step.step}-${index}`}>
                <StepDot active={isLatest} />
                <div className="thinking-step-body">
                  <div className="thinking-step-head">
                    <span className="thinking-step-label">
                      <Icon size={14} aria-hidden="true" />
                      {meta.label}
                    </span>
                    {step.timestamp ? <span className="thinking-step-time">{step.timestamp}</span> : null}
                  </div>

                  {step.message ? <p className="thinking-step-msg">{step.message}</p> : null}

                  {step.queries?.length ? (
                    <div className="thinking-chip-row">
                      {step.queries.map((query) => (
                        <span className="thinking-chip" key={query}>
                          {query}
                        </span>
                      ))}
                    </div>
                  ) : null}

                  <div className="thinking-metrics">
                    {step.count != null ? (
                      <span className="thinking-metric">{step.count} 条结果</span>
                    ) : null}
                    {step.verdict ? (
                      <span className={clsx('thinking-verdict', step.verdict)}>
                        {formatVerdict(step.verdict)}
                      </span>
                    ) : null}
                    {step.gap ? <span className="thinking-gap">{step.gap}</span> : null}
                    {step.reasoning ? <span className="thinking-reasoning">{step.reasoning}</span> : null}
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <div className="thinking-collapsed">
          <span className="thinking-collapsed-label">
            <StepDot active />
            {latest ? STEP_META[latest.step]?.label ?? latest.step : 'Active'}
          </span>
          <span className="thinking-collapsed-copy">
            {latest?.message ?? '实时追踪检索循环中...'}
          </span>
        </div>
      )}
    </section>
  );
}
