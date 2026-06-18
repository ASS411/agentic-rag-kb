export type WelcomePanelProps = {
  examples: string[];
  streaming: boolean;
  onSubmitExample: (question: string) => void;
};

export function WelcomePanel({ examples, streaming, onSubmitExample }: WelcomePanelProps) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 px-4 py-16 text-center">
      <div className="max-w-xl">
        <span className="signal-line block mx-auto" />
        <p className="text-[11px] font-semibold tracking-wider uppercase text-muted-foreground mt-5 mb-2">
          证据优先问答
        </p>
        <h2 className="text-3xl sm:text-4xl font-bold text-foreground leading-tight mb-3">
          向知识库提问
        </h2>
        <p className="text-sm text-muted-foreground mb-8">
          上传笔记、报告或PDF，提出需要可靠答案的问题。
        </p>

        <div className="flex flex-wrap justify-center gap-2">
          {examples.map((item) => (
            <button
              type="button"
              key={item}
              className="px-3 py-1.5 rounded-full text-xs border border-[hsl(var(--border))] text-muted-foreground hover:bg-popover hover:text-foreground disabled:opacity-30 transition-colors"
              onClick={() => void onSubmitExample(item)}
              disabled={streaming}
            >
              {item}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
