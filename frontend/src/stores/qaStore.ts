import { create } from 'zustand';
import type { ChatMessage, SearchChunk, ThinkingStep } from '../types';

export type QAState = {
  /** Chat message history (user + assistant turns). */
  messages: ChatMessage[];
  /** Source chunks from the last completed answer. */
  sources: SearchChunk[];
  /** Pre-formatted source note text. */
  sourcesNote: string;
  /** Agent thinking steps for the current turn. */
  thinkingSteps: ThinkingStep[];
  /** Currently selected (clicked) source chunk ID. */
  selectedSourceId: string | null;
  /** Whether the thinking panel is expanded. */
  thinkingOpen: boolean;

  // Actions
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  updateMessage: (id: string, content: string) => void;
  setSources: (sources: SearchChunk[], note: string) => void;
  addThinkingStep: (step: ThinkingStep) => void;
  clearThinkingSteps: () => void;
  selectSource: (chunkId: string | null) => void;
  toggleThinking: () => void;
  reset: () => void;
};

const initialState = {
  messages: [],
  sources: [],
  sourcesNote: '',
  thinkingSteps: [],
  selectedSourceId: null,
  thinkingOpen: true,
};

export const useQAStore = create<QAState>((set) => ({
  ...initialState,

  setMessages: (messages) => set({ messages }),

  addMessage: (message) =>
    set((s) => ({ messages: [...s.messages, message] })),

  updateMessage: (id, content) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + content } : m,
      ),
    })),

  setSources: (sources, note) =>
    set({
      sources,
      sourcesNote: note,
      selectedSourceId: sources[0]?.chunk_id ?? null,
    }),

  addThinkingStep: (step) =>
    set((s) => {
      const next = [...s.thinkingSteps, step];
      return { thinkingSteps: next.length > 20 ? next.slice(-20) : next };
    }),

  clearThinkingSteps: () => set({ thinkingSteps: [] }),

  selectSource: (chunkId) => set({ selectedSourceId: chunkId }),

  toggleThinking: () => set((s) => ({ thinkingOpen: !s.thinkingOpen })),

  reset: () => set(initialState),
}));
