import { create } from 'zustand';

export type UIState = {
  /** Sidebar open state (mobile responsiveness). */
  sidebarOpen: boolean;
  /** Whether the Agent thinking panel is expanded. */
  thinkingExpanded: boolean;
  /** Currently selected source chunk ID for highlight. */
  selectedSourceId: string | null;
  /** Whether the upload modal is open (future). */
  uploadModalOpen: boolean;

  // Actions ---------------------------------------------------------------

  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  toggleThinking: () => void;
  setThinkingExpanded: (expanded: boolean) => void;
  setSelectedSource: (id: string | null) => void;
  setUploadModalOpen: (open: boolean) => void;
};

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  thinkingExpanded: true,
  selectedSourceId: null,
  uploadModalOpen: false,

  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  toggleThinking: () => set((s) => ({ thinkingExpanded: !s.thinkingExpanded })),
  setThinkingExpanded: (expanded) => set({ thinkingExpanded: expanded }),
  setSelectedSource: (id) => set({ selectedSourceId: id }),
  setUploadModalOpen: (open) => set({ uploadModalOpen: open }),
}));
