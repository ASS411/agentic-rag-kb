import type { StreamEvent } from '../types';

/** Format bytes to a human-readable string. */
export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

/** Format an ISO date string to a compact Chinese locale string. */
export function compactDate(value: string): string {
  if (!value) return '刚刚';
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value));
}

/** Collapse whitespace in a string. */
export function trimSource(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

/** Read a JSON response, throwing on non-ok status. */
export async function readJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T;
  if (!response.ok) {
    const message =
      typeof payload === 'object' && payload !== null && 'message' in payload
        ? String((payload as { message?: unknown }).message)
        : response.statusText;
    throw new Error(message || '请求失败');
  }
  return payload;
}

/** Parse an SSE text block into a StreamEvent or null. */
export function parseSseBlock(block: string): StreamEvent | null {
  const data = block
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');

  if (!data) return null;

  try {
    const parsed = JSON.parse(data) as StreamEvent;
    if (
      parsed &&
      typeof parsed === 'object' &&
      'type' in parsed &&
      typeof parsed.type === 'string' &&
      [
        'agent-step',
        'answer-chunk',
        'answer-done',
        'token',
        'sources',
        'done',
        'error',
      ].includes(parsed.type)
    ) {
      return parsed;
    }
  } catch {
    return null;
  }

  return null;
}
