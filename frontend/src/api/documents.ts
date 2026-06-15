import type {
  ApiEnvelope,
  DocumentItem,
  DocumentList,
  SearchChunk,
  SearchResponse,
} from '../types';
import { readJson } from '../lib/utils';

const API_BASE = '/api/v1';

// ---------------------------------------------------------------------------
// Document API
// ---------------------------------------------------------------------------

export async function fetchDocuments(): Promise<DocumentList> {
  const payload = await readJson<ApiEnvelope<DocumentList>>(
    await fetch(`${API_BASE}/documents?page=1&size=30`),
  );
  return payload.data;
}

export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${docId}`, { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as any).message || `删除失败 (${res.status})`);
  }
}

export async function fetchSearchChunks(question: string): Promise<SearchChunk[]> {
  const payload = await readJson<ApiEnvelope<SearchResponse>>(
    await fetch(`${API_BASE}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: question, top_k: 5 }),
    }),
  );
  return payload.data.results;
}

export function uploadFile(
  file: File,
  onProgress: (progress: number) => void,
): Promise<DocumentItem> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const form = new FormData();
    form.append('file', file);

    request.open('POST', `${API_BASE}/documents/upload`);

    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    request.onload = () => {
      try {
        const payload = JSON.parse(request.responseText) as ApiEnvelope<DocumentItem>;
        if (request.status >= 200 && request.status < 300 && payload.success) {
          resolve(payload.data);
          return;
        }
        reject(new Error(payload.message || '上传失败'));
      } catch {
        reject(new Error('无法解析上传响应'));
      }
    };

    request.onerror = () => reject(new Error('上传连接失败'));
    request.send(form);
  });
}
