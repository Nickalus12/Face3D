import { create } from "zustand";
import {
  listSessions,
  getSessionFiles,
  type Session,
} from "../lib/api";

export interface SessionFile {
  name: string;
  size: number;
}

interface SessionState {
  sessions: Session[];
  currentSession: Session | null;
  sessionFiles: SessionFile[];
  isLoading: boolean;
  error: string | null;

  fetchSessions: () => Promise<void>;
  selectSession: (session: Session) => Promise<void>;
  createSession: (name: string, contentDir: string) => void;
  clearError: () => void;
}

const useSessionStore = create<SessionState>((set, get) => ({
  sessions: [],
  currentSession: null,
  sessionFiles: [],
  isLoading: false,
  error: null,

  fetchSessions: async () => {
    set({ isLoading: true, error: null });
    try {
      const sessions = await listSessions();
      const current = get().currentSession;
      // Keep current selection if it still exists, otherwise auto-select first
      const stillExists = current
        ? sessions.find((s) => s.id === current.id)
        : null;
      const selected = stillExists ?? sessions[0] ?? null;

      set({
        sessions,
        isLoading: false,
        currentSession: selected,
      });

      // Load files for the selected session
      if (selected) {
        const files = await getSessionFiles(selected.id);
        set({
          sessionFiles: files.map(([name, size]) => ({ name, size })),
        });
      }
    } catch (e) {
      set({
        isLoading: false,
        error: e instanceof Error ? e.message : "Failed to fetch sessions",
      });
    }
  },

  selectSession: async (session: Session) => {
    set({ currentSession: session, sessionFiles: [] });
    try {
      const files = await getSessionFiles(session.id);
      set({
        sessionFiles: files.map(([name, size]) => ({ name, size })),
      });
    } catch (e) {
      console.error("Failed to fetch session files:", e);
    }
  },

  createSession: (name: string, _contentDir: string) => {
    // Creating a session is just a local placeholder until the pipeline
    // actually creates the output directory on disk.
    const id = name.trim().replace(/\s+/g, "_").toLowerCase();
    const newSession: Session = {
      id,
      name,
      has_gaussians: false,
      has_mesh: false,
      has_renders: false,
    };
    set((state) => ({
      sessions: [newSession, ...state.sessions],
      currentSession: newSession,
      sessionFiles: [],
    }));
  },

  clearError: () => set({ error: null }),
}));

export default useSessionStore;
