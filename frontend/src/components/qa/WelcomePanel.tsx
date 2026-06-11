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
        <p className="eyebrow">证据优先问答</p>
        <h2>向知识库提问，观察智能代理收集证据。</h2>
        <p>
          上传笔记、报告或PDF，提出需要可靠答案的问题，而不是依赖模糊的记忆。
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
