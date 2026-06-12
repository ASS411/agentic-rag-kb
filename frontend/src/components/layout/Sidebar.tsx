import { Database, PanelLeftClose, PanelLeftOpen, Upload, MessageSquare } from 'lucide-react';
import type { ReactNode } from 'react';
import clsx from 'clsx';

export type SidebarProps = {
  children: ReactNode;
  collapsed: boolean;
  onToggle: () => void;
  mobileOpen: boolean;
  onMobileClose: () => void;
  /** History list rendered inside sidebar when expanded */
  history?: ReactNode;
  /** Upload trigger rendered inside sidebar */
  onUploadClick?: () => void;
};

export function Sidebar({
  children,
  collapsed,
  onToggle,
  mobileOpen,
  onMobileClose,
  history,
  onUploadClick,
}: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {mobileOpen ? (
        <div
          className="fixed inset-0 z-20 bg-black/40 md:hidden"
          onClick={onMobileClose}
        />
      ) : null}

      <aside
        className={clsx(
          'left-rail relative',
          collapsed && 'collapsed',
          mobileOpen && 'mobile-open',
        )}
      >
        {/* Collapsed state: brand icon + toggle */}
        <div className="brand-icon">
          <Database size={20} aria-hidden="true" />
        </div>

        <button
          className="sidebar-toggle"
          type="button"
          onClick={onToggle}
          aria-label={collapsed ? '展开侧边栏' : '折叠侧边栏'}
        >
          {collapsed ? (
            <PanelLeftOpen size={18} />
          ) : (
            <PanelLeftClose size={18} />
          )}
        </button>

        {/* Brand lockup (visible when expanded) */}
        <div className="brand-lockup">
          <div className="brand-mark">
            <Database size={20} aria-hidden="true" />
          </div>
          <div>
            <p className="eyebrow">智能检索增强</p>
            <h1>证据库</h1>
          </div>
        </div>

        {/* Upload button (visible when expanded) */}
        {onUploadClick ? (
          <button
            className="flex items-center gap-2 w-full px-3 py-2 mt-2 rounded-md text-sm border border-dashed border-[hsl(var(--border))] text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--popover))] hover:text-[hsl(var(--foreground))] transition-colors"
            type="button"
            onClick={onUploadClick}
          >
            <Upload size={15} />
            上传文档
          </button>
        ) : null}

        {/* Document list section */}
        {children}

        {/* History list (visible when expanded) */}
        {history ? (
          <div className="history-section">
            <h3>
              <MessageSquare size={13} className="inline mr-1" />
              对话历史
            </h3>
            {history}
          </div>
        ) : null}
      </aside>
    </>
  );
}
