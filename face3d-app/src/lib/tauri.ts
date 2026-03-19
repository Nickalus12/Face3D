/**
 * Tauri IPC compatibility shim.
 *
 * This file re-exports everything from the new type-safe API layer
 * so that existing component imports (`from "../lib/tauri"`) continue
 * to work without modification.
 *
 * New code should import from "../lib/api" directly.
 */

// ── Re-export all public functions and types from api.ts ────────
export {
  // Types
  type Session,
  type GpuInfo,
  type PipelineLogPayload,
  type PreviewFile,
  type SensorInfo,
  type SensorSummary,
  type PythonInfo,
  type SystemInfo,

  // Functions
  isTauri,
  listSessions,
  getSessionFiles,
  startPipeline,
  stopPipeline,
  isPipelineRunning,
  getGpuInfo,
  getModelPath,
  openFolder,
  listRenders,
  listPreviews,
  listInputFrames,
  getReportPath,
  getImageCounts,
  getSensorSummary,
  getSensorData,
  getPipelineConfig,
  savePipelineConfig,
  getPythonInfo,
  getSystemInfo,
  getSessionMetrics,
  invalidateSessionCaches,
  bindToast,
  getErrorHistory,
} from "./api";

// ── Re-export event helpers from events.ts ──────────────────────
export { eventManager, EVENTS } from "./events";
export type {
  StageChangePayload,
  TrainingMetricPayload,
  FileChangePayload,
} from "./events";

// ── Legacy event listener wrappers ──────────────────────────────
//
// The old tauri.ts exposed onPipelineLog / onPipelineComplete that
// returned Promise<UnlistenFn> | null.  We preserve that signature
// for backward compat, but new code should use eventManager directly.

import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { isTauri, type PipelineLogPayload } from "./api";

export function onPipelineLog(
  handler: (payload: PipelineLogPayload) => void,
): Promise<UnlistenFn> | null {
  if (!isTauri()) return null;
  return listen<PipelineLogPayload>("pipeline-log", (event) => {
    handler(event.payload);
  });
}

export function onPipelineComplete(
  handler: (exitCode: number) => void,
): Promise<UnlistenFn> | null {
  if (!isTauri()) return null;
  return listen<number>("pipeline-complete", (event) => {
    handler(event.payload);
  });
}

// ── selectContentDir ────────────────────────────────────────────
//
// This uses the Tauri dialog plugin, not invoke(), so it stays here.

import { open } from "@tauri-apps/plugin-dialog";

export async function selectContentDir(): Promise<string | null> {
  if (!isTauri()) {
    // Dev mode: return a fake path so the wizard flow works
    return "D:/Projects/3D/data/raw/dev_session";
  }
  try {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "Select Content Directory",
    });
    if (selected && typeof selected === "string") {
      return selected;
    }
    return null;
  } catch {
    return null;
  }
}
