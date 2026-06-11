import { Loader2 } from 'lucide-react';

export type LoadingProps = {
  /** Text label shown below the spinner. */
  message?: string;
  /** Spinner size in px. */
  size?: number;
  /** Additional class names. */
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
