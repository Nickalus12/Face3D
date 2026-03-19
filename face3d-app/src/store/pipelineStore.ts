import { create } from "zustand";
import {
  startPipeline as apiStartPipeline,
  stopPipeline as apiStopPipeline,
  getGpuInfo as apiGetGpuInfo,
  type GpuInfo,
} from "../lib/api";

// ── Types ────────────────────────────────────────────────────────

export interface StageStatus {
  id: number;
  name: string;
  status: "pending" | "running" | "complete" | "error" | "skipped";
  elapsed?: number;
}

export interface LogEntry {
  timestamp: string;
  level: "info" | "warn" | "error" | "stderr" | "debug";
  message: string;
}

export interface MetricPoint {
  iter: number;
  loss?: number;
  psnr?: number;
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

export { PIPELINE_STAGES };

// ── Store ────────────────────────────────────────────────────────

interface PipelineState {
  status: "idle" | "running" | "stopping" | "complete" | "error";
  currentStage: number;
  stages: StageStatus[];
  logs: LogEntry[];
  metrics: MetricPoint[];
  gpuInfo: GpuInfo | null;
  contentDir: string;
  sessionName: string;
  startedAt: number | null;

  // Actions
  setContentDir: (dir: string) => void;
  setSessionName: (name: string) => void;
  startPipeline: (contentDir: string, session: string) => Promise<void>;
  stopPipeline: () => Promise<void>;
  fetchGpuInfo: () => Promise<void>;
  addLog: (line: string, level: string) => void;
  parseLogLine: (line: string, level: string) => void;
  clearLogs: () => void;
  resetPipeline: () => void;
  onPipelineComplete: (exitCode: number) => void;
}

const MAX_LOG_LINES = 1000;

const usePipelineStore = create<PipelineState>((set, get) => ({
  status: "idle",
  currentStage: 0,
  stages: PIPELINE_STAGES.map((s) => ({ ...s, status: "pending" as const })),
  logs: [],
  metrics: [],
  gpuInfo: null,
  contentDir: "",
  sessionName: "",
  startedAt: null,

  setContentDir: (dir) => set({ contentDir: dir }),
  setSessionName: (name) => set({ sessionName: name }),

  startPipeline: async (contentDir: string, session: string) => {
    // ── Optimistic update ──────────────────────────────────────
    // Immediately flip to "running" so the UI feels instant.
    set({
      status: "running",
      currentStage: 1,
      stages: PIPELINE_STAGES.map((s) => ({
        ...s,
        status: s.id === 1 ? ("running" as const) : ("pending" as const),
      })),
      metrics: [],
      logs: [],
      startedAt: Date.now(),
      contentDir,
      sessionName: session,
    });

    get().addLog(`Starting pipeline for session: ${session}`, "info");
    get().addLog(`Content directory: ${contentDir}`, "info");

    try {
      await apiStartPipeline(contentDir, session);
    } catch (e) {
      // ── Revert optimistic state on failure ─────────────────
      const msg = e instanceof Error ? e.message : String(e);
      get().addLog(`Pipeline start failed: ${msg}`, "error");
      set({
        status: "idle",
        currentStage: 0,
        stages: PIPELINE_STAGES.map((s) => ({
          ...s,
          status: "pending" as const,
        })),
        startedAt: null,
      });
    }
  },

  stopPipeline: async () => {
    set({ status: "stopping" });
    get().addLog("Pipeline stop requested", "warn");
    try {
      await apiStopPipeline();
    } catch {
      // May fail if not running — non-critical
    }
    const { currentStage } = get();
    set((state) => ({
      status: "idle",
      stages: state.stages.map((s) =>
        s.id === currentStage && s.status === "running"
          ? { ...s, status: "error" as const }
          : s,
      ),
    }));
    get().addLog("Pipeline stopped by user", "warn");
  },

  fetchGpuInfo: async () => {
    const info = await apiGetGpuInfo();
    if (info) {
      set({ gpuInfo: info });
    }
  },

  addLog: (line: string, level: string) => {
    const entry: LogEntry = {
      timestamp: new Date().toISOString().slice(11, 23),
      level: level as LogEntry["level"],
      message: line,
    };
    set((state) => {
      const logs = [...state.logs, entry];
      if (logs.length > MAX_LOG_LINES) {
        return { logs: logs.slice(logs.length - MAX_LOG_LINES) };
      }
      return { logs };
    });
  },

  parseLogLine: (line: string, level: string) => {
    const state = get();

    // Always add the log
    state.addLog(line, level);

    // Detect stage transitions: "Stage N completed" or "=== Stage N:"
    const stageCompleteMatch = line.match(
      /Stage\s+(\d+)\s+completed/i,
    );
    if (stageCompleteMatch) {
      const completedStage = parseInt(stageCompleteMatch[1], 10);
      const nextStage = completedStage + 1;
      set((s) => ({
        currentStage: nextStage <= 14 ? nextStage : completedStage,
        stages: s.stages.map((st) => {
          if (st.id === completedStage)
            return { ...st, status: "complete" as const };
          if (st.id === nextStage && nextStage <= 14)
            return { ...st, status: "running" as const };
          return st;
        }),
      }));
      return;
    }

    // Detect stage start: "=== Stage N:" or "Running stage N"
    const stageStartMatch = line.match(
      /(?:===\s*Stage|Running\s+stage)\s+(\d+)/i,
    );
    if (stageStartMatch) {
      const stageNum = parseInt(stageStartMatch[1], 10);
      set((s) => ({
        currentStage: stageNum,
        stages: s.stages.map((st) => {
          if (st.id === stageNum)
            return { ...st, status: "running" as const };
          if (st.id < stageNum && st.status !== "error" && st.status !== "skipped")
            return { ...st, status: "complete" as const };
          return st;
        }),
      }));
      return;
    }

    // Detect training metrics: "loss=X.XXX" or "psnr=X.XX" or "n_gs=NNNN"
    const lossMatch = line.match(/loss[=:]\s*([\d.]+)/i);
    const psnrMatch = line.match(/psnr[=:]\s*([\d.]+)/i);
    const gsMatch = line.match(/n_gs[=:]\s*([\d,]+)/i);
    const iterMatch = line.match(/(?:iter|iteration|step)[=:\s]*(\d+)/i);

    if (lossMatch || psnrMatch || gsMatch) {
      const point: MetricPoint = {
        iter: iterMatch
          ? parseInt(iterMatch[1], 10)
          : state.metrics.length,
        loss: lossMatch ? parseFloat(lossMatch[1]) : undefined,
        psnr: psnrMatch ? parseFloat(psnrMatch[1]) : undefined,
        gaussians: gsMatch
          ? parseInt(gsMatch[1].replace(/,/g, ""), 10)
          : undefined,
      };
      set((s) => ({ metrics: [...s.metrics, point] }));
    }

    // Detect errors
    if (level === "error" || line.includes("[ERROR]")) {
      set({ status: "error" });
    }

    // Detect pipeline completion
    if (/pipeline\s+complete/i.test(line)) {
      set((s) => ({
        status: "complete",
        stages: s.stages.map((st) =>
          st.status === "running" || st.status === "pending"
            ? { ...st, status: "complete" as const }
            : st,
        ),
      }));
    }
  },

  clearLogs: () => set({ logs: [] }),

  resetPipeline: () =>
    set({
      status: "idle",
      currentStage: 0,
      stages: PIPELINE_STAGES.map((s) => ({
        ...s,
        status: "pending" as const,
      })),
      logs: [],
      metrics: [],
      startedAt: null,
    }),

  onPipelineComplete: (exitCode: number) => {
    const state = get();
    if (exitCode === 0) {
      set((s) => ({
        status: "complete",
        stages: s.stages.map((st) =>
          st.status === "running" || st.status === "pending"
            ? { ...st, status: "complete" as const }
            : st,
        ),
      }));
      state.addLog("Pipeline completed successfully", "info");
    } else {
      set({ status: "error" });
      state.addLog(
        `Pipeline exited with code ${exitCode}`,
        "error",
      );
    }
  },
}));

export default usePipelineStore;
