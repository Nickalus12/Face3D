import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

// ── Types matching Rust backend ──────────────────────────────────

export interface Session {
  id: string;
  name: string;
  has_gaussians: boolean;
  has_mesh: boolean;
  has_renders: boolean;
}

export interface GpuInfo {
  name: string;
  memory_total: string;
  memory_used: string;
  utilization: string;
}

export interface PipelineLogPayload {
  line: string;
  level: string;
}

// ── Tauri detection ──────────────────────────────────────────────

function isTauri(): boolean {
  return typeof window !== "undefined" && !!(window as any).__TAURI_INTERNALS__;
}

// ── Session commands ─────────────────────────────────────────────

export async function listSessions(): Promise<Session[]> {
  if (!isTauri()) {
    return [];
  }
  try {
    return await invoke<Session[]>("list_sessions");
  } catch (e) {
    console.error("list_sessions failed:", e);
    return [];
  }
}

export async function getSessionFiles(
  sessionId: string,
): Promise<[string, number][]> {
  if (!isTauri()) {
    return [];
  }
  try {
    return await invoke<[string, number][]>("get_session_files", {
      sessionId,
    });
  } catch (e) {
    console.error("get_session_files failed:", e);
    return [];
  }
}

// ── Pipeline commands ────────────────────────────────────────────

export async function startPipeline(
  contentDir: string,
  session: string,
): Promise<void> {
  if (!isTauri()) {
    console.warn("[dev] startPipeline called outside Tauri");
    return;
  }
  await invoke("start_pipeline", { contentDir, session });
}

export async function stopPipeline(): Promise<void> {
  if (!isTauri()) {
    console.warn("[dev] stopPipeline called outside Tauri");
    return;
  }
  await invoke("stop_pipeline");
}

export async function isPipelineRunning(): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    return await invoke<boolean>("is_pipeline_running");
  } catch {
    return false;
  }
}

// ── GPU ──────────────────────────────────────────────────────────

export async function getGpuInfo(): Promise<GpuInfo | null> {
  if (!isTauri()) {
    return null;
  }
  try {
    return await invoke<GpuInfo>("get_gpu_info");
  } catch (e) {
    console.error("get_gpu_info failed:", e);
    return null;
  }
}

// ── Model path ──────────────────────────────────────────────────

export async function getModelPath(
  sessionId: string,
): Promise<string | null> {
  if (!isTauri()) {
    return null;
  }
  try {
    return await invoke<string>("get_model_path", { sessionId });
  } catch (e) {
    console.error("get_model_path failed:", e);
    return null;
  }
}

// ── File system helpers ──────────────────────────────────────────

export async function openFolder(path: string): Promise<void> {
  if (!isTauri()) return;
  try {
    await invoke("open_folder", { path });
  } catch (e) {
    console.error("open_folder failed:", e);
  }
}

export async function selectContentDir(): Promise<string | null> {
  if (!isTauri()) {
    return null;
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

// ── Gallery commands ────────────────────────────────────────────

export interface PreviewFile {
  name: string;
  preview_type: string;
  path: string;
}

export async function listRenders(sessionId: string): Promise<string[]> {
  if (!isTauri()) return [];
  try {
    return await invoke<string[]>("list_renders", { sessionId });
  } catch (e) {
    console.error("list_renders failed:", e);
    return [];
  }
}

export async function listPreviews(sessionId: string): Promise<PreviewFile[]> {
  if (!isTauri()) return [];
  try {
    return await invoke<PreviewFile[]>("list_previews", { sessionId });
  } catch (e) {
    console.error("list_previews failed:", e);
    return [];
  }
}

export async function listInputFrames(sessionId: string): Promise<string[]> {
  if (!isTauri()) return [];
  try {
    return await invoke<string[]>("list_input_frames", { sessionId });
  } catch (e) {
    console.error("list_input_frames failed:", e);
    return [];
  }
}

export async function getReportPath(
  sessionId: string,
): Promise<string | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<string | null>("get_report_path", { sessionId });
  } catch (e) {
    console.error("get_report_path failed:", e);
    return null;
  }
}

export async function getImageCounts(
  sessionId: string,
): Promise<[number, number, number]> {
  if (!isTauri()) return [0, 0, 0];
  try {
    return await invoke<[number, number, number]>("get_image_counts", {
      sessionId,
    });
  } catch (e) {
    console.error("get_image_counts failed:", e);
    return [0, 0, 0];
  }
}

// ── Sensor data ─────────────────────────────────────────────────

export interface SensorInfo {
  name: string;
  samples: number;
  duration: number;
}

export interface SensorSummary {
  device_name: string;
  recording_time: string;
  duration: number;
  sensors: SensorInfo[];
}

export async function getSensorSummary(
  sessionId: string,
): Promise<SensorSummary | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<SensorSummary>("get_sensor_summary", { sessionId });
  } catch (e) {
    console.error("get_sensor_summary failed:", e);
    return null;
  }
}

export async function getSensorData(
  sessionId: string,
  sensorType: string,
): Promise<number[][] | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<number[][]>("get_sensor_data", {
      sessionId,
      sensorType,
    });
  } catch (e) {
    console.error("get_sensor_data failed:", e);
    return null;
  }
}

// ── Settings commands ────────────────────────────────────────────

export interface PythonInfo {
  version: string;
  env_name: string;
  packages_count: number;
}

export interface SystemInfo {
  os_version: string;
  cpu: string;
  ram_gb: number;
  disk_free_gb: number;
  disk_total_gb: number;
  cuda_version: string;
}

export async function getPipelineConfig(): Promise<string | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<string>("get_pipeline_config");
  } catch (e) {
    console.error("get_pipeline_config failed:", e);
    return null;
  }
}

export async function savePipelineConfig(config: string): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    await invoke("save_pipeline_config", { config });
    return true;
  } catch (e) {
    console.error("save_pipeline_config failed:", e);
    return false;
  }
}

export async function getPythonInfo(): Promise<PythonInfo | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<PythonInfo>("get_python_info");
  } catch (e) {
    console.error("get_python_info failed:", e);
    return null;
  }
}

export async function getSystemInfo(): Promise<SystemInfo | null> {
  if (!isTauri()) return null;
  try {
    return await invoke<SystemInfo>("get_system_info");
  } catch (e) {
    console.error("get_system_info failed:", e);
    return null;
  }
}

// ── Session metrics ─────────────────────────────────────────────

export async function getSessionMetrics(
  sessionId: string,
): Promise<string> {
  if (!isTauri()) return "";
  try {
    return await invoke<string>("get_session_metrics", { sessionId });
  } catch (e) {
    console.error("get_session_metrics failed:", e);
    return "";
  }
}

// ── Event listeners ──────────────────────────────────────────────

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
