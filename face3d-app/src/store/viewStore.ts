import { create } from "zustand";

export type ViewId =
  | "home"
  | "pipeline"
  | "view"
  | "metrics"
  | "gallery"
  | "compare"
  | "sensors";

interface ViewState {
  activeView: ViewId;
  setActiveView: (view: ViewId) => void;
}

const useViewStore = create<ViewState>((set) => ({
  activeView: "view",
  setActiveView: (view) => set({ activeView: view }),
}));

export default useViewStore;
