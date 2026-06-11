export type WelcomePanelProps = {
  examples: string[];
  streaming: boolean;
  onSubmitExample: (question: string) => void;
};

export function WelcomePanel({ examples, streaming, onSubmitExample }: WelcomePanelProps) {
  return (
    <div className="start-panel">
      <div className="start-copy">
        <span className="signal-line" />
        <p className="eyebrow">Evidence-first Q&A</p>
        <h2>把问题丢进知识库，让答案带着来源回来。</h2>
        <p>
          适合把零散资料放在同一张桌面上审阅：答案在中间生长，证据在旁边排队等待核验。
        </p>
      </div>
      <div className="example-grid">
        {examples.map((item) => (
          <button
            type="button"
            key={item}
            onClick={() => void onSubmitExample(item)}
            disabled={streaming}
          >
            {item}
          </button>
        ))}
      </div>
    </div>
  );
}
