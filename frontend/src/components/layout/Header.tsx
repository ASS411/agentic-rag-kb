import { CheckCircle2, PanelRight, Search } from 'lucide-react';

export type HeaderProps = {
  /** Total number of documents. */
  docCount: number;
  /** Number of documents with chunks (retrievable). */
  readyCount: number;
  /** Toggle the source panel visibility. */
  onToggleSourcePanel: () => void;
};

export function Header({ docCount, readyCount, onToggleSourcePanel }: HeaderProps) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Evidence Desk</p>
        <h2>知识问答工作台</h2>
      </div>
      <div className="status-pills">
        <span>
          <CheckCircle2 size={15} aria-hidden="true" />
          {docCount} 份文档
        </span>
        <span>
          <Search size={15} aria-hidden="true" />
          {readyCount} 份可检索
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
