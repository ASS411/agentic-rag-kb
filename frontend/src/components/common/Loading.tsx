import { Loader2 } from 'lucide-react';

export type LoadingProps = {
  message?: string;
  size?: number;
  className?: string;
};

export function Loading({ message = '加载中...', size = 18, className = '' }: LoadingProps) {
  return (
    <div className={`quiet-state ${className}`}>
      <Loader2 className="spin" size={size} aria-hidden="true" />
      {message}
    </div>
  );
}
