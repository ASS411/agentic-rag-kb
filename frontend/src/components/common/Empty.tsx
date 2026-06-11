import type { ReactNode } from 'react';

export type EmptyProps = {
  /** Icon element (lucide-react component or similar). */
  icon: ReactNode;
  /** Primary heading. */
  title: string;
  /** Secondary description. */
  description?: string;
  /** Optional action buttons or links. */
  actions?: ReactNode;
};

export function Empty({ icon, title, description, actions }: EmptyProps) {
  return (
    <div className="empty-copy" style={{ textAlign: 'center', padding: '2rem 0' }}>
      <div style={{ marginBottom: '0.75rem', opacity: 0.5 }}>{icon}</div>
      <strong style={{ display: 'block', marginBottom: '0.25rem' }}>{title}</strong>
      {description ? (
        <span style={{ display: 'block', fontSize: '0.85rem', opacity: 0.6 }}>
          {description}
        </span>
      ) : null}
      {actions ? <div style={{ marginTop: '1rem' }}>{actions}</div> : null}
    </div>
  );
}
