import { Loader2, UploadCloud } from 'lucide-react';
import { type ChangeEvent, type DragEvent } from 'react';
import type { UploadState } from '../../types';

export type DropZoneProps = {
  upload: UploadState;
  onFiles: (files: FileList | File[]) => void;
};

export function DropZone({ upload, onFiles }: DropZoneProps) {
  const busy = upload.phase === 'uploading' || upload.phase === 'processing';

  const handleDragOver = (event: DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    if (busy) return;
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
      <label
        className="upload-target"
        aria-disabled={busy}
      >
        <input
          className="upload-input"
          type="file"
          accept=".pdf,.md,.markdown,.txt,application/pdf,text/markdown,text/plain"
          onChange={handleChange}
          disabled={busy}
        />
        {busy ? (
          <Loader2 className="spin" size={22} aria-hidden="true" />
        ) : (
          <UploadCloud size={24} aria-hidden="true" />
        )}
        <span>拖拽文件或点击上传</span>
        <small>{upload.message}</small>
      </label>
      <div className="progress-track" aria-label="上传进度">
        <span style={{ width: `${upload.progress}%` }} />
      </div>
    </section>
  );
}
