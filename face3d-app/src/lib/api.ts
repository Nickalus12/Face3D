/**
 * Type-safe IPC layer for Tauri commands.
 *
 * Every Rust #[tauri::command] is registered in the Commands interface so
 * the compiler catches mismatched argument names, types, and return types.
 *
 * Features:
 *   - Automatic dev-mode mock fallback (browser without Tauri runtime)
 *   - Retry with exponential back-off for transient failures
 *   - Centralised error logging with human-friendly messages
 *   - Optional TTL caching via IPCCache
 *   - Toast integration for user-visible errors
 */

import { invoke } from "@tauri-apps/api/core";
import {
  ipcCache,
  TTL_GPU,
  TTL_SESSION_LIST,
  TTL_SENSOR,
  TTL_MODEL_PATH,
  TTL_SESSION_FILES,
  TTL_SYSTEM_INFO,
} from "./cache";

// ── Re-export shared types ──────────────────────────────────────

export interface Session {
  id: string;
  name: string;
  has_gaussians: boolean;
  has_mesh: boolean;
  has_renders: boolean;
  // Extended fields from DB (optional for backward compat)
  quality_grade?: string;
  quality_score?: number;
  total_size_bytes?: number;
  status?: string;
  created_at?: string;
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

export interface PreviewFile {
  name: string;
  preview_type: string;
  path: string;
}

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

// ── Command registry ────────────────────────────────────────────

/**
 * Maps every Tauri command name to its argument shape and return type.
 * Argument keys MUST use camelCase (Tauri deserialises JS camelCase
 * to Rust snake_case automatically via serde rename_all).
 */
interface Commands {
  list_sessions:       { args: Record<string, never>; response: Session[] };
  get_session_files:   { args: { sessionId: string }; response: [string, number][] };
  start_pipeline:      { args: { contentDir: string; session: string }; response: void };
  stop_pipeline:       { args: Record<string, never>; response: void };
  is_pipeline_running: { args: Record<string, never>; response: boolean };
  get_gpu_info:        { args: Record<string, never>; response: GpuInfo };
  get_model_path:      { args: { sessionId: string }; response: string };
  open_folder:         { args: { path: string }; response: void };
  list_renders:        { args: { sessionId: string }; response: string[] };
  list_previews:       { args: { sessionId: string }; response: PreviewFile[] };
  list_input_frames:   { args: { sessionId: string }; response: string[] };
  get_report_path:     { args: { sessionId: string }; response: string | null };
  get_image_counts:    { args: { sessionId: string }; response: [number, number, number] };
  get_sensor_summary:  { args: { sessionId: string }; response: SensorSummary };
  get_sensor_data:     { args: { sessionId: string; sensorType: string }; response: number[][] };
  get_pipeline_config: { args: Record<string, never>; response: string };
  save_pipeline_config:{ args: { config: string }; response: void };
  get_python_info:     { args: Record<string, never>; response: PythonInfo };
  get_system_info:     { args: Record<string, never>; response: SystemInfo };
  get_session_metrics: { args: { sessionId: string }; response: string };
}

// ── Tauri runtime detection ─────────────────────────────────────

export function isTauri(): boolean {
  return (
    typeof window !== "undefined" &&
    !!(window as any).__TAURI_INTERNALS__
  );
}

// ── Error helpers ───────────────────────────────────────────────

/** Map raw Rust error strings to short user-facing messages. */
function friendlyMessage(command: string, raw: unknown): string {
  const msg = raw instanceof Error ? raw.message : String(raw);

  if (msg.includes("ENOENT") || msg.includes("not found") || msg.includes("NotFound")) {
    return `Resource not found (${command})`;
  }
  if (msg.includes("EACCES") || msg.includes("Permission")) {
    return `Permission denied (${command})`;
  }
  if (msg.includes("timeout") || msg.includes("Timeout")) {
    return `Operation timed out (${command})`;
  }
  if (msg.includes("already running")) {
    return "Pipeline is already running";
  }
  if (msg.includes("not running")) {
    return "No pipeline is running";
  }

  // Truncate very long Rust backtraces
  const short = msg.length > 200 ? msg.slice(0, 200) + "..." : msg;
  return `${command} failed: ${short}`;
}

/** Classify whether an error is likely transient (worth retrying). */
function isTransient(raw: unknown): boolean {
  const msg = raw instanceof Error ? raw.message : String(raw);
  return (
    msg.includes("timeout") ||
    msg.includes("Timeout") ||
    msg.includes("ECONNRESET") ||
    msg.includes("EPIPE") ||
    msg.includes("busy")
  );
}

// ── Error tracking ──────────────────────────────────────────────

interface ErrorRecord {
  command: string;
  message: string;
  timestamp: number;
}

const recentErrors: ErrorRecord[] = [];
const MAX_ERROR_HISTORY = 50;

function trackError(command: string, raw: unknown): void {
  const message = raw instanceof Error ? raw.message : String(raw);
  recentErrors.push({ command, message, timestamp: Date.now() });
  if (recentErrors.length > MAX_ERROR_HISTORY) {
    recentErrors.shift();
  }
}

/** Retrieve recent error log (useful for debug panels). */
export function getErrorHistory(): readonly ErrorRecord[] {
  return recentErrors;
}

// ── Toast integration (lazy-bound) ──────────────────────────────

type ToastFn = (message: string, type: "error" | "warning" | "info" | "success") => void;

let _showToast: ToastFn | null = null;

/**
 * Must be called once from React tree (e.g., in App.tsx useEffect)
 * to wire the toast store into the IPC layer.
 */
export function bindToast(fn: ToastFn): void {
  _showToast = fn;
}

function showErrorToast(message: string): void {
  if (_showToast) {
    _showToast(message, "error");
  }
}

// ── Core invoke wrapper ─────────────────────────────────────────

interface CallOptions {
  /** If true, suppress toast on error (caller handles UI). */
  silent?: boolean;
  /** Max retries for transient failures. Default 0. */
  retries?: number;
  /** Cache key. If set, result is cached with the given TTL. */
  cacheKey?: string;
  /** Cache TTL in ms (requires cacheKey). */
  cacheTtl?: number;
}

/**
 * Type-safe invoke wrapper.
 *
 * ```ts
 * const sessions = await api.call("list_sessions", {});
 * //    ^? Session[]
 * ```
 */
async function call<K extends keyof Commands>(
  command: K,
  args: Commands[K]["args"],
  options: CallOptions = {},
): Promise<Commands[K]["response"]> {
  const { silent = false, retries = 0, cacheKey, cacheTtl } = options;

  // Check cache first
  if (cacheKey) {
    const cached = ipcCache.get<Commands[K]["response"]>(cacheKey);
    if (cached !== null) return cached;
  }

  // Dev-mode fallback
  if (!isTauri()) {
    return getDevFallback(command);
  }

  let lastError: unknown;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const result = await invoke<Commands[K]["response"]>(command, args);

      // Populate cache on success
      if (cacheKey && cacheTtl) {
        ipcCache.set(cacheKey, result, cacheTtl);
      }

      return result;
    } catch (e) {
      lastError = e;
      console.error(
        `[IPC] ${command} attempt ${attempt + 1}/${retries + 1} failed:`,
        e,
      );
      trackError(command, e);

      // Only retry transient errors
      if (attempt < retries && isTransient(e)) {
        const delay = Math.min(1000 * 2 ** attempt, 5000);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      break;
    }
  }

  // All attempts failed
  if (!silent) {
    showErrorToast(friendlyMessage(command, lastError));
  }
  throw lastError;
}

// ── Dev-mode mock data ──────────────────────────────────────────

const MOCK_SESSIONS: Session[] = [
  {
    id: "demo_session",
    name: "Demo Session",
    has_gaussians: true,
    has_mesh: true,
    has_renders: true,
  },
  {
    id: "test_capture",
    name: "Test Capture",
    has_gaussians: false,
    has_mesh: false,
    has_renders: false,
  },
];

const MOCK_GPU: GpuInfo = {
  name: "NVIDIA RTX 3080 (Mock)",
  memory_total: "16384 MB",
  memory_used: "4096 MB",
  utilization: "35%",
};

const MOCK_SYSTEM: SystemInfo = {
  os_version: "Windows 11 (Mock)",
  cpu: "AMD Ryzen 9 5900X (Mock)",
  ram_gb: 32,
  disk_free_gb: 256,
  disk_total_gb: 1024,
  cuda_version: "12.1 (Mock)",
};

const MOCK_PYTHON: PythonInfo = {
  version: "3.10.12 (Mock)",
  env_name: "face3d",
  packages_count: 142,
};

function getDevFallback<K extends keyof Commands>(
  command: K,
): Commands[K]["response"] {
  const mocks: Record<string, unknown> = {
    list_sessions: MOCK_SESSIONS,
    get_session_files: [
      ["gaussians.ply", 52_428_800],
      ["mesh.obj", 12_582_912],
      ["training_log.txt", 102_400],
    ] as [string, number][],
    start_pipeline: undefined,
    stop_pipeline: undefined,
    is_pipeline_running: false,
    get_gpu_info: MOCK_GPU,
    get_model_path: "/mock/path/to/gaussians.ply",
    open_folder: undefined,
    list_renders: [
      "/mock/render_000.png",
      "/mock/render_001.png",
      "/mock/render_002.png",
    ],
    list_previews: [] as PreviewFile[],
    list_input_frames: ["/mock/frame_000.jpg", "/mock/frame_001.jpg"],
    get_report_path: null,
    get_image_counts: [80, 65, 12] as [number, number, number],
    get_sensor_summary: {
      device_name: "Samsung S25 Ultra (Mock)",
      recording_time: "2025-01-15T10:30:00",
      duration: 45.2,
      sensors: [
        { name: "accelerometer", samples: 4520, duration: 45.2 },
        { name: "gyroscope", samples: 4520, duration: 45.2 },
      ],
    } as SensorSummary,
    get_sensor_data: Array.from({ length: 100 }, (_, i) => [
      i * 0.01,
      Math.sin(i * 0.1),
      Math.cos(i * 0.1),
      Math.sin(i * 0.05),
    ]),
    get_pipeline_config: "# Mock pipeline config\nsplatting:\n  training:\n    iterations: 3000\n",
    save_pipeline_config: undefined,
    get_python_info: MOCK_PYTHON,
    get_system_info: MOCK_SYSTEM,
    get_session_metrics: '{"psnr": 28.5, "ssim": 0.92, "lpips": 0.08}',
  };

  if (command in mocks) {
    console.info(`[DEV] Mock response for ${command}`);
    return mocks[command] as Commands[K]["response"];
  }

  console.warn(`[DEV] No mock for command: ${command}`);
  return undefined as Commands[K]["response"];
}

// ── Public API (convenience wrappers) ───────────────────────────
//
// These preserve the exact same function signatures that components
// already import from "../lib/tauri", so migration is zero-effort.
// Internally they delegate to the typed `call()` with caching,
// retries, and error handling.
//

export async function listSessions(): Promise<Session[]> {
  try {
    return await call("list_sessions", {}, {
      retries: 1,
      cacheKey: "sessions",
      cacheTtl: TTL_SESSION_LIST,
    });
  } catch {
    return [];
  }
}

export async function getSessionFiles(
  sessionId: string,
): Promise<[string, number][]> {
  try {
    return await call("get_session_files", { sessionId }, {
      cacheKey: `session_files:${sessionId}`,
      cacheTtl: TTL_SESSION_FILES,
    });
  } catch {
    return [];
  }
}

export async function startPipeline(
  contentDir: string,
  session: string,
): Promise<void> {
  // Invalidate session caches — the pipeline will create new data
  ipcCache.invalidate("sessions");
  ipcCache.invalidate("session_files:");
  return call("start_pipeline", { contentDir, session });
}

export async function stopPipeline(): Promise<void> {
  return call("stop_pipeline", {});
}

export async function isPipelineRunning(): Promise<boolean> {
  try {
    return await call("is_pipeline_running", {}, { silent: true });
  } catch {
    return false;
  }
}

export async function getGpuInfo(): Promise<GpuInfo | null> {
  try {
    return await call("get_gpu_info", {}, {
      silent: true,
      cacheKey: "gpu_info",
      cacheTtl: TTL_GPU,
    });
  } catch {
    return null;
  }
}

export async function getModelPath(
  sessionId: string,
): Promise<string | null> {
  try {
    return await call("get_model_path", { sessionId }, {
      cacheKey: `model_path:${sessionId}`,
      cacheTtl: TTL_MODEL_PATH,
    });
  } catch {
    return null;
  }
}

export async function openFolder(path: string): Promise<void> {
  try {
    await call("open_folder", { path }, { silent: true });
  } catch {
    // swallow — not critical
  }
}

export async function listRenders(sessionId: string): Promise<string[]> {
  try {
    return await call("list_renders", { sessionId }, {
      cacheKey: `renders:${sessionId}`,
      cacheTtl: TTL_SESSION_FILES,
    });
  } catch {
    return [];
  }
}

export async function listPreviews(
  sessionId: string,
): Promise<PreviewFile[]> {
  try {
    return await call("list_previews", { sessionId }, {
      cacheKey: `previews:${sessionId}`,
      cacheTtl: TTL_SESSION_FILES,
    });
  } catch {
    return [];
  }
}

export async function listInputFrames(
  sessionId: string,
): Promise<string[]> {
  try {
    return await call("list_input_frames", { sessionId }, {
      cacheKey: `input_frames:${sessionId}`,
      cacheTtl: TTL_SESSION_FILES,
    });
  } catch {
    return [];
  }
}

export async function getReportPath(
  sessionId: string,
): Promise<string | null> {
  try {
    return await call("get_report_path", { sessionId }, { silent: true });
  } catch {
    return null;
  }
}

export async function getImageCounts(
  sessionId: string,
): Promise<[number, number, number]> {
  try {
    return await call("get_image_counts", { sessionId }, {
      cacheKey: `image_counts:${sessionId}`,
      cacheTtl: TTL_SESSION_FILES,
    });
  } catch {
    return [0, 0, 0];
  }
}

export async function getSensorSummary(
  sessionId: string,
): Promise<SensorSummary | null> {
  try {
    return await call("get_sensor_summary", { sessionId }, {
      cacheKey: `sensor_summary:${sessionId}`,
      cacheTtl: TTL_SENSOR,
    });
  } catch {
    return null;
  }
}

export async function getSensorData(
  sessionId: string,
  sensorType: string,
): Promise<number[][] | null> {
  try {
    return await call("get_sensor_data", { sessionId, sensorType }, {
      cacheKey: `sensor_data:${sessionId}:${sensorType}`,
      cacheTtl: TTL_SENSOR,
    });
  } catch {
    return null;
  }
}

export async function getPipelineConfig(): Promise<string | null> {
  try {
    return await call("get_pipeline_config", {});
  } catch {
    return null;
  }
}

export async function savePipelineConfig(
  config: string,
): Promise<boolean> {
  try {
    await call("save_pipeline_config", { config });
    return true;
  } catch {
    return false;
  }
}

export async function getPythonInfo(): Promise<PythonInfo | null> {
  try {
    return await call("get_python_info", {}, {
      cacheKey: "python_info",
      cacheTtl: TTL_SYSTEM_INFO,
    });
  } catch {
    return null;
  }
}

export async function getSystemInfo(): Promise<SystemInfo | null> {
  try {
    return await call("get_system_info", {}, {
      cacheKey: "system_info",
      cacheTtl: TTL_SYSTEM_INFO,
    });
  } catch {
    return null;
  }
}

export async function getSessionMetrics(
  sessionId: string,
): Promise<string> {
  try {
    return await call("get_session_metrics", { sessionId });
  } catch {
    return "";
  }
}

/**
 * Invalidate all cached data associated with sessions.
 * Call this after pipeline completes or session list changes.
 */
export function invalidateSessionCaches(): void {
  ipcCache.invalidate("sessions");
  ipcCache.invalidate("session_files:");
  ipcCache.invalidate("renders:");
  ipcCache.invalidate("previews:");
  ipcCache.invalidate("input_frames:");
  ipcCache.invalidate("image_counts:");
  ipcCache.invalidate("model_path:");
  ipcCache.invalidate("sensor_");
}
