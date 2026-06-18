import type { ReactNode } from 'react';

export type EmptyProps = {
  icon: ReactNode;
  title: string;
  description?: string;
  actions?: ReactNode;
};

export function Empty({ icon, title, description, actions }: EmptyProps) {
  return (
    <div className="text-center py-8">
      <div className="mb-3 opacity-50">{icon}</div>
      <strong className="block mb-1">{title}</strong>
      {description ? (
        <span className="block text-sm opacity-60">{description}</span>
      ) : null}
      {actions ? <div className="mt-4">{actions}</div> : null}
    </div>
  );
}
