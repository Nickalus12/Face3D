import { create } from "zustand";
import { listSessions, type Session } from "../lib/tauri";

interface SessionState {
  sessions: Session[];
  currentSession: Session | null;
  isLoading: boolean;
  error: string | null;

  fetchSessions: () => Promise<void>;
  selectSession: (session: Session) => void;
  createSession: (name: string, contentDir: string) => void;
}

const useSessionStore = create<SessionState>((set, get) => ({
  sessions: [],
  currentSession: null,
  isLoading: false,
  error: null,

  fetchSessions: async () => {
    set({ isLoading: true, error: null });
    try {
      const sessions = await listSessions();
      const current = get().currentSession;
      set({
        sessions,
        isLoading: false,
        // Auto-select first session if none selected
        currentSession: current ?? sessions[0] ?? null,
      });
    } catch (e) {
      set({
        isLoading: false,
        error: e instanceof Error ? e.message : "Failed to fetch sessions",
      });
    }
  },

  selectSession: (session: Session) => {
    set({ currentSession: session });
  },

  createSession: (name: string, contentDir: string) => {
    const id = `session_${Date.now()}`;
    const newSession: Session = {
      id,
      name,
      created: new Date().toISOString(),
      stage: 0,
      contentDir,
    };
    set((state) => ({
      sessions: [newSession, ...state.sessions],
      currentSession: newSession,
    }));
  },
}));

export default useSessionStore;
