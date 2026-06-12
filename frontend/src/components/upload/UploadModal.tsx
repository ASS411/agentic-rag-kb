import { X } from 'lucide-react';
import type { ReactNode } from 'react';

export type UploadModalProps = {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
};

export function UploadModal({ open, onClose, children }: UploadModalProps) {
  if (!open) return null;

  return (
    <div className="upload-modal-overlay" onClick={onClose}>
      <div className="upload-modal" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-foreground">上传文档</h2>
          <button
            type="button"
            className="p-1 rounded-md hover:bg-popover text-muted-foreground"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
