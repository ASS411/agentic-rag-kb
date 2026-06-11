import { Loader2, UploadCloud } from 'lucide-react';
import { type ChangeEvent, type DragEvent, useRef } from 'react';
import type { UploadState } from '../../types';

export type DropZoneProps = {
  upload: UploadState;
  onFiles: (files: FileList | File[]) => void;
};

export function DropZone({ upload, onFiles }: DropZoneProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const busy = upload.phase === 'uploading' || upload.phase === 'processing';

  const handleDragOver = (event: DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    void onFiles(event.dataTransfer.files);
  };

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) void onFiles(event.target.files);
    event.currentTarget.value = '';
  };

  return (
    <section
      className={`upload-pad ${upload.phase}`}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        className="sr-only"
        type="file"
        accept=".pdf,.md,.markdown,.txt,application/pdf,text/markdown,text/plain"
        onChange={handleChange}
      />
      <button
        className="upload-target"
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={busy}
      >
        {busy ? (
          <Loader2 className="spin" size={22} aria-hidden="true" />
        ) : (
          <UploadCloud size={24} aria-hidden="true" />
        )}
        <span>拖拽文件或点击上传</span>
        <small>{upload.message}</small>
      </button>
      <div className="progress-track" aria-label="上传进度">
        <span style={{ width: `${upload.progress}%` }} />
      </div>
    </section>
  );
}
