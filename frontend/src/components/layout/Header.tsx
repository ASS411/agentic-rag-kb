import { CheckCircle2, PanelRight, Search } from 'lucide-react';

export type HeaderProps = {
  docCount: number;
  readyCount: number;
  onToggleSourcePanel: () => void;
};

export function Header({ docCount, readyCount, onToggleSourcePanel }: HeaderProps) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">证据工作台</p>
        <h2>知识问答工作区</h2>
      </div>
      <div className="status-pills">
        <span>
          <CheckCircle2 size={15} aria-hidden="true" />
          {docCount} 文档
        </span>
        <span>
          <Search size={15} aria-hidden="true" />
          {readyCount} 已索引
        </span>
        <button
          className="icon-button"
          type="button"
          onClick={onToggleSourcePanel}
          aria-label="切换来源面板"
        >
          <PanelRight size={17} aria-hidden="true" />
        </button>
      </div>
    </header>
  );
}
