import { create } from "zustand";
import {
  startPipeline as tauriStartPipeline,
  stopPipeline as tauriStopPipeline,
} from "../lib/tauri";

export interface StageStatus {
  id: number;
  name: string;
  status: "pending" | "running" | "complete" | "error" | "skipped";
  duration?: number;
}

export interface LogEntry {
  timestamp: string;
  level: "INFO" | "WARN" | "ERROR" | "DEBUG";
  message: string;
}

export interface MetricPoint {
  iteration: number;
  psnr?: number;
  loss?: number;
  gaussians?: number;
}

const PIPELINE_STAGES: { id: number; name: string }[] = [
  { id: 1, name: "Frame Extraction" },
  { id: 2, name: "Color Correction" },
  { id: 3, name: "Quality Filtering" },
  { id: 4, name: "IMU Parsing" },
  { id: 5, name: "Rotation Priors" },
  { id: 6, name: "COLMAP SfM" },
  { id: 7, name: "Depth Estimation" },
  { id: 8, name: "Depth Alignment" },
  { id: 9, name: "Face Landmarks" },
  { id: 10, name: "FLAME Fitting" },
  { id: 11, name: "Face Segmentation" },
  { id: 12, name: "Gaussian Init" },
  { id: 13, name: "Gaussian Training" },
  { id: 14, name: "Export & Render" },
];

interface PipelineState {
  stages: StageStatus[];
  isRunning: boolean;
  currentStage: number;
  logs: LogEntry[];
  metrics: MetricPoint[];
  contentDir: string;
  sessionName: string;

  setContentDir: (dir: string) => void;
  setSessionName: (name: string) => void;
  startPipeline: () => Promise<void>;
  stopPipeline: () => Promise<void>;
  addLog: (level: LogEntry["level"], message: string) => void;
  addMetric: (point: MetricPoint) => void;
  clearLogs: () => void;
  updateStage: (id: number, status: StageStatus["status"]) => void;
  setCurrentStage: (stage: number) => void;
}

const usePipelineStore = create<PipelineState>((set, get) => ({
  stages: PIPELINE_STAGES.map((s) => ({ ...s, status: "pending" as const })),
  isRunning: false,
  currentStage: 0,
  logs: [],
  metrics: [],
  contentDir: "",
  sessionName: "",

  setContentDir: (dir: string) => set({ contentDir: dir }),
  setSessionName: (name: string) => set({ sessionName: name }),

  startPipeline: async () => {
    const { contentDir, sessionName } = get();
    if (!contentDir) {
      get().addLog("ERROR", "No content directory selected");
      return;
    }

    set({
      isRunning: true,
      currentStage: 1,
      stages: PIPELINE_STAGES.map((s) => ({
        ...s,
        status: "pending" as const,
      })),
      metrics: [],
    });

    get().addLog("INFO", `Starting pipeline for session: ${sessionName || "unnamed"}`);
    get().addLog("INFO", `Content directory: ${contentDir}`);
    get().updateStage(1, "running");

    try {
      await tauriStartPipeline(contentDir, sessionName);
    } catch (e) {
      get().addLog(
        "ERROR",
        `Pipeline start failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  },

  stopPipeline: async () => {
    get().addLog("WARN", "Pipeline stop requested");
    try {
      await tauriStopPipeline();
    } catch {
      // Handled in dev mode
    }
    const { currentStage } = get();
    set((state) => ({
      isRunning: false,
      stages: state.stages.map((s) =>
        s.id === currentStage && s.status === "running"
          ? { ...s, status: "error" as const }
          : s,
      ),
    }));
    get().addLog("WARN", "Pipeline stopped");
  },

  addLog: (level, message) => {
    const entry: LogEntry = {
      timestamp: new Date().toISOString().slice(11, 23),
      level,
      message,
    };
    set((state) => ({ logs: [...state.logs, entry] }));
  },

  addMetric: (point) => {
    set((state) => ({ metrics: [...state.metrics, point] }));
  },

  clearLogs: () => set({ logs: [] }),

  updateStage: (id, status) => {
    set((state) => ({
      stages: state.stages.map((s) => (s.id === id ? { ...s, status } : s)),
    }));
  },

  setCurrentStage: (stage) => set({ currentStage: stage }),
}));

export default usePipelineStore;
