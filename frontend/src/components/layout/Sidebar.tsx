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
          <p className="eyebrow">智能检索增强</p>
          <h1>证据库</h1>
        </div>
      </div>

      {children}
    </aside>
  );
}
