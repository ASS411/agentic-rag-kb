import { Database } from 'lucide-react';
import type { ReactNode } from 'react';

export type SidebarProps = {
  children: ReactNode;
};

export function Sidebar({ children }: SidebarProps) {
  return (
    <aside className="left-rail">
      <div className="brand-lockup">
        <div className="brand-mark">
          <Database size={20} aria-hidden="true" />
        </div>
        <div>
          <p className="eyebrow">Agentic RAG</p>
          <h1>知识证据台</h1>
        </div>
      </div>

      {children}
    </aside>
  );
}
